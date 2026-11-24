"""Unit tests for MultiTimeframeEngine using standard unittest."""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scripts.engine.mtf_engine import MultiTimeframeEngine, MTFDecision


class TestMultiTimeframeEngine(unittest.TestCase):

    def setUp(self):
        self.mtf = MultiTimeframeEngine(min_alignment_score=65.0)

    def _create_df(self, n=50, **kwargs):
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        dates = [start + timedelta(hours=i) for i in range(n)]
        data = {
            'date': dates,
            'close': np.full(n, 100.0),
            'ema_20': np.full(n, 100.0),
            'ema_50': np.full(n, 100.0),
            'ema_200': np.full(n, 100.0),
            'adx_14': np.full(n, 20.0),
            'rsi_14': np.full(n, 50.0),
            'macd_hist': np.full(n, 0.0),
            'macd_hist_expanding': np.zeros(n, dtype=bool),
            'market_structure_trend': np.zeros(n, dtype=int),
            'supertrend_direction': np.ones(n, dtype=int),
            'ema_50_slope': np.zeros(n),
            'is_bullish_hammer': np.zeros(n, dtype=bool),
            'is_bullish_engulfing': np.zeros(n, dtype=bool),
            'is_resistance_breakout': np.zeros(n, dtype=bool),
            'is_ema20_pullback_bounce': np.zeros(n, dtype=bool),
            'is_ema50_pullback_bounce': np.zeros(n, dtype=bool)
        }
        for k, v in kwargs.items():
            data[k] = np.full(n, v) if isinstance(v, (int, float, bool)) else v
        return pd.DataFrame(data)

    def test_strong_bullish_alignment_triggers_buy(self):
        # 1D Bullish
        df_1d = self._create_df(
            n=50,
            close=110.0,
            ema_50=105.0,
            ema_200=95.0,
            rsi_14=58.0,
            market_structure_trend=1
        )
        # 4H Bullish
        df_4h = self._create_df(
            n=50,
            close=110.0,
            ema_50=105.0,
            ema_200=95.0,
            adx_14=28.0,
            supertrend_direction=1,
            ema_50_slope=0.10,
            market_structure_trend=1
        )
        # 1H Entry Trigger
        df_1h = self._create_df(
            n=50,
            close=101.0,
            ema_20=100.0,
            rsi_14=52.0,
            macd_hist=0.5,
            macd_hist_expanding=True,
            is_bullish_hammer=True
        )

        res = self.mtf.evaluate(df_1d, df_4h, df_1h, current_position_open=False)
        self.assertEqual(res.decision, MTFDecision.BUY)
        self.assertGreaterEqual(res.alignment_score, 65.0)
        self.assertEqual(len(res.conflicts), 0)

    def test_daily_bearish_and_4h_bearish_conflict(self):
        # Both Daily and 4H are BEARISH
        df_1d = self._create_df(n=50, close=85.0, ema_50=90.0, ema_200=105.0, rsi_14=35.0, market_structure_trend=-1)
        df_4h = self._create_df(n=50, close=85.0, ema_50=90.0, ema_200=105.0, adx_14=30.0, supertrend_direction=-1, market_structure_trend=-1)
        # Even if 1H shows a temporary bounce
        df_1h = self._create_df(n=50, close=86.0, ema_20=85.0, rsi_14=48.0, is_bullish_hammer=True)

        res = self.mtf.evaluate(df_1d, df_4h, df_1h, current_position_open=False)
        self.assertEqual(res.decision, MTFDecision.NO_TRADE)
        self.assertTrue(any("macro timeframe conflict" in c.lower() for c in res.conflicts))

    def test_daily_bullish_vs_4h_bearish_divergence(self):
        # Daily is Bullish
        df_1d = self._create_df(n=50, close=110.0, ema_50=105.0, ema_200=95.0, rsi_14=55.0)
        # 4H is Bearish
        df_4h = self._create_df(n=50, close=98.0, ema_50=102.0, ema_200=105.0, supertrend_direction=-1, market_structure_trend=-1)
        df_1h = self._create_df(n=50, close=99.0, ema_20=98.0, rsi_14=50.0)

        res = self.mtf.evaluate(df_1d, df_4h, df_1h, current_position_open=False)
        self.assertEqual(res.decision, MTFDecision.NO_TRADE)
        self.assertTrue(any("timeframe divergence" in c.lower() for c in res.conflicts))

    def test_open_position_hold_and_exit(self):
        # Healthy position -> HOLD
        df_1d = self._create_df(n=50, close=110.0, ema_50=105.0, ema_200=95.0, rsi_14=60.0)
        df_4h = self._create_df(n=50, close=110.0, ema_50=105.0, supertrend_direction=1)
        df_1h = self._create_df(n=50, close=110.0, ema_20=108.0, rsi_14=55.0)

        hold_res = self.mtf.evaluate(df_1d, df_4h, df_1h, current_position_open=True)
        self.assertEqual(hold_res.decision, MTFDecision.HOLD)

        # 4H Structure collapses to Bearish -> EXIT
        df_4h_breakdown = self._create_df(n=50, close=94.0, ema_50=105.0, supertrend_direction=-1, market_structure_trend=-1)
        exit_res = self.mtf.evaluate(df_1d, df_4h_breakdown, df_1h, current_position_open=True)
        self.assertEqual(exit_res.decision, MTFDecision.EXIT)


if __name__ == '__main__':
    unittest.main()

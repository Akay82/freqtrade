"""Unit tests for SignalScorer using standard unittest."""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scripts.engine.signal_scorer import SignalScorer, ScoringConfig


class TestSignalScorer(unittest.TestCase):

    def setUp(self):
        self.scorer = SignalScorer()

    def _create_df(self, n=50, **kwargs):
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        dates = [start + timedelta(hours=i) for i in range(n)]
        data = {
            'date': dates,
            'close': np.full(n, 100.0),
            'open': np.full(n, 99.5),
            'ema_20': np.full(n, 100.0),
            'ema_50': np.full(n, 100.0),
            'ema_200': np.full(n, 100.0),
            'adx_14': np.full(n, 20.0),
            'rsi_14': np.full(n, 50.0),
            'macd': np.full(n, 0.5),
            'macd_signal': np.full(n, 0.2),
            'macd_hist': np.full(n, 0.3),
            'macd_hist_expanding': np.ones(n, dtype=bool),
            'relative_volume': np.full(n, 1.0),
            'cmf_20': np.full(n, 0.05),
            'market_structure_trend': np.ones(n, dtype=int),
            'supertrend_direction': np.ones(n, dtype=int),
            'is_bullish_hammer': np.zeros(n, dtype=bool),
            'is_bullish_engulfing': np.zeros(n, dtype=bool),
            'is_resistance_breakout': np.zeros(n, dtype=bool),
            'is_ema20_pullback_bounce': np.zeros(n, dtype=bool),
            'is_ema50_pullback_bounce': np.zeros(n, dtype=bool)
        }
        for k, v in kwargs.items():
            data[k] = np.full(n, v) if isinstance(v, (int, float, bool)) else v
        return pd.DataFrame(data)

    def test_strong_candidate_score(self):
        # Perfect alignment
        df_1d = self._create_df(close=110.0, ema_50=105.0, ema_200=95.0, rsi_14=60.0)
        df_4h = self._create_df(close=110.0, ema_50=105.0, supertrend_direction=1, adx_14=26.0)
        df_1h = self._create_df(
            close=101.0,
            ema_20=100.0,
            rsi_14=55.0,
            macd_hist_expanding=True,
            relative_volume=1.4,
            cmf_20=0.08,
            is_bullish_hammer=True
        )

        res = self.scorer.score_signal(df_1d, df_4h, df_1h, strategy_name="TestTrend")
        self.assertGreaterEqual(res.total_score, 80.0)
        self.assertEqual(res.classification, "STRONG")
        self.assertTrue(res.is_approved)
        self.assertTrue(res.mandatory_passed)
        self.assertEqual(len(res.rejection_reasons), 0)

    def test_overextension_penalty_rejection(self):
        df_1d = self._create_df(close=110.0, ema_50=105.0, ema_200=95.0)
        df_4h = self._create_df(close=110.0, ema_50=105.0)
        # Price is overextended: Close = 104, EMA 20 = 100 (+4.0% > 2.8% max)
        df_1h = self._create_df(close=104.0, ema_20=100.0)

        res = self.scorer.score_signal(df_1d, df_4h, df_1h, strategy_name="TestOverextended")
        self.assertFalse(res.mandatory_passed)
        self.assertFalse(res.is_approved)
        self.assertEqual(res.classification, "REJECT")
        self.assertTrue(any("overextended" in r.lower() for r in res.rejection_reasons))

    def test_low_score_rejection(self):
        # Weak parameters across all timeframes
        df_1d = self._create_df(close=90.0, ema_50=95.0, ema_200=105.0, rsi_14=35.0, market_structure_trend=0)
        df_4h = self._create_df(close=90.0, ema_50=95.0, supertrend_direction=-1, adx_14=12.0)
        df_1h = self._create_df(close=90.0, ema_20=92.0, rsi_14=32.0, macd_hist=-0.5, macd_hist_expanding=False, relative_volume=0.6)

        res = self.scorer.score_signal(df_1d, df_4h, df_1h, strategy_name="TestWeak")
        self.assertLess(res.total_score, 65.0)
        self.assertEqual(res.classification, "REJECT")
        self.assertFalse(res.is_approved)
        self.assertTrue(any("below minimum approval" in r.lower() for r in res.rejection_reasons))


if __name__ == '__main__':
    unittest.main()

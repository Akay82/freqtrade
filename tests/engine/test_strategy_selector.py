"""Unit tests for StrategySelector using standard unittest."""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scripts.engine.strategy_selector import StrategySelector
from scripts.engine.regime_detector import MarketRegime


class TestStrategySelector(unittest.TestCase):

    def setUp(self):
        self.selector = StrategySelector()

    def _create_df(self, n=50, **kwargs):
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        dates = [start + timedelta(hours=i) for i in range(n)]
        data = {
            'date': dates,
            'close': np.full(n, 100.0),
            'open': np.full(n, 99.5),
            'high': np.full(n, 102.0),
            'low': np.full(n, 98.0),
            'ema_20': np.full(n, 100.0),
            'ema_50': np.full(n, 95.0),
            'ema_200': np.full(n, 90.0),
            'adx_14': np.full(n, 25.0),
            'rsi_14': np.full(n, 52.0),
            'macd': np.full(n, 0.5),
            'macd_signal': np.full(n, 0.2),
            'macd_hist': np.full(n, 0.3),
            'macd_hist_expanding': np.ones(n, dtype=bool),
            'relative_volume': np.full(n, 1.2),
            'cmf_20': np.full(n, 0.05),
            'atr_pct': np.full(n, 1.2),
            'atr_14': np.full(n, 1.5),
            'supertrend_direction': np.ones(n, dtype=int),
            'market_structure_trend': np.ones(n, dtype=int),
            'resistance_level': np.full(n, 105.0),
            'support_level': np.full(n, 95.0),
            'is_bullish_hammer': np.zeros(n, dtype=bool),
            'is_bullish_engulfing': np.zeros(n, dtype=bool),
            'is_resistance_breakout': np.zeros(n, dtype=bool),
            'is_ema20_pullback_bounce': np.zeros(n, dtype=bool),
            'is_ema50_pullback_bounce': np.zeros(n, dtype=bool),
            'ttm_squeeze_on': np.zeros(n, dtype=bool)
        }
        for k, v in kwargs.items():
            data[k] = np.full(n, v) if isinstance(v, (int, float, bool)) else v
        return pd.DataFrame(data)

    def test_selector_approves_best_setup_in_bullish_trend(self):
        df_1d = self._create_df(close=110.0, ema_50=105.0, ema_200=95.0, rsi_14=60.0)
        df_4h = self._create_df(close=110.0, ema_50=105.0, ema_200=95.0, adx_14=28.0, supertrend_direction=1)
        df_1h = self._create_df(
            close=100.8,
            open=100.0,
            ema_20=100.0,
            rsi_14=52.0,
            relative_volume=1.35,
            is_bullish_hammer=True
        )

        res = self.selector.select_and_evaluate(df_1d, df_4h, df_1h, current_position_open=False)
        self.assertEqual(res.final_action, "BUY")
        self.assertIsNotNone(res.selected_strategy)
        self.assertEqual(res.active_regime, MarketRegime.BULLISH_TREND)
        self.assertGreaterEqual(res.best_score.total_score, 65.0)
        self.assertIn("BUY Signal Approved", res.explanation)

    def test_selector_blocks_trades_during_bearish_regime(self):
        # Bearish on all counts
        df_1d = self._create_df(close=80.0, ema_50=85.0, ema_200=100.0, rsi_14=32.0, market_structure_trend=-1)
        df_4h = self._create_df(close=80.0, ema_50=85.0, ema_200=100.0, adx_14=30.0, supertrend_direction=-1, market_structure_trend=-1)
        df_1h = self._create_df(close=81.0, ema_20=82.0, rsi_14=42.0)

        res = self.selector.select_and_evaluate(df_1d, df_4h, df_1h, current_position_open=False)
        self.assertEqual(res.final_action, "NONE")
        self.assertEqual(res.active_regime, MarketRegime.BEARISH_TREND)
        self.assertIsNone(res.selected_strategy)
        self.assertTrue(len(res.rejections) > 0)

    def test_selector_position_management(self):
        df_1d = self._create_df(close=110.0, ema_50=105.0, ema_200=95.0)
        df_4h = self._create_df(close=110.0, ema_50=105.0, supertrend_direction=1)
        df_1h = self._create_df(close=110.0, ema_20=105.0)

        # Existing position in bullish trend -> HOLD
        hold_res = self.selector.select_and_evaluate(df_1d, df_4h, df_1h, current_position_open=True)
        self.assertEqual(hold_res.final_action, "HOLD")

        # 4H structure breaks down -> EXIT
        df_4h_down = self._create_df(close=90.0, ema_50=105.0, supertrend_direction=-1, market_structure_trend=-1)
        exit_res = self.selector.select_and_evaluate(df_1d, df_4h_down, df_1h, current_position_open=True)
        self.assertEqual(exit_res.final_action, "EXIT")


if __name__ == '__main__':
    unittest.main()

"""Unit tests for modular strategies A, B, C, D using standard unittest."""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scripts.engine.strategies import (
    StrategyA_TrendFollowing, StrategyB_Breakout,
    StrategyC_Pullback, StrategyD_RangeBound
)
from scripts.engine.regime_detector import MarketRegime


class TestStrategies(unittest.TestCase):

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
            'rsi_14': np.full(n, 50.0),
            'macd_hist': np.full(n, 0.2),
            'macd_hist_expanding': np.ones(n, dtype=bool),
            'relative_volume': np.full(n, 1.0),
            'atr_14': np.full(n, 1.5),
            'supertrend_direction': np.ones(n, dtype=int),
            'resistance_level': np.full(n, 105.0),
            'support_level': np.full(n, 95.0),
            'is_bullish_hammer': np.zeros(n, dtype=bool),
            'is_bullish_engulfing': np.zeros(n, dtype=bool),
            'is_resistance_breakout': np.zeros(n, dtype=bool),
            'is_ema20_pullback_bounce': np.zeros(n, dtype=bool),
            'is_ema50_pullback_bounce': np.zeros(n, dtype=bool)
        }
        for k, v in kwargs.items():
            data[k] = np.full(n, v) if isinstance(v, (int, float, bool)) else v
        return pd.DataFrame(data)

    def test_strategy_eligibility(self):
        strat_a = StrategyA_TrendFollowing()
        strat_b = StrategyB_Breakout()
        strat_d = StrategyD_RangeBound(is_enabled=False)

        self.assertTrue(strat_a.is_eligible(MarketRegime.BULLISH_TREND))
        self.assertFalse(strat_a.is_eligible(MarketRegime.SIDEWAYS))
        self.assertFalse(strat_a.is_eligible(MarketRegime.BEARISH_TREND))

        self.assertTrue(strat_b.is_eligible(MarketRegime.LOW_VOLATILITY))
        self.assertTrue(strat_b.is_eligible(MarketRegime.SIDEWAYS))

        # D is disabled by default
        self.assertFalse(strat_d.is_eligible(MarketRegime.SIDEWAYS))

    def test_strategy_a_trend_following_trigger(self):
        strat_a = StrategyA_TrendFollowing()
        df_1d = self._create_df()
        df_4h = self._create_df(supertrend_direction=1, adx_14=25.0)
        df_1h = self._create_df(
            close=100.8,
            open=100.0,
            ema_20=100.0,
            rsi_14=52.0,
            macd_hist_expanding=True
        )

        sig = strat_a.evaluate_setup(df_1d, df_4h, df_1h)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.action, "BUY")
        self.assertGreater(sig.suggested_take_profit, sig.entry_price)
        self.assertLess(sig.suggested_stop_loss, sig.entry_price)
        self.assertGreaterEqual(sig.risk_reward_ratio, 1.5)

    def test_strategy_b_breakout_trigger(self):
        strat_b = StrategyB_Breakout()
        df_1d = self._create_df()
        df_4h = self._create_df()
        df_1h = self._create_df(
            close=106.0,
            open=104.5,
            resistance_level=105.0,
            relative_volume=1.4,
            is_resistance_breakout=True
        )

        sig = strat_b.evaluate_setup(df_1d, df_4h, df_1h)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.action, "BUY")
        self.assertIn("breakout", sig.reasons[0].lower())

    def test_strategy_c_pullback_trigger(self):
        strat_c = StrategyC_Pullback()
        df_1d = self._create_df()
        df_4h = self._create_df()
        df_1h = self._create_df(
            close=100.2,
            open=99.8,
            low=99.9,
            ema_20=100.0,
            ema_50=95.0,
            rsi_14=46.0,
            is_bullish_hammer=True
        )

        sig = strat_c.evaluate_setup(df_1d, df_4h, df_1h)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.action, "BUY")
        self.assertTrue(any("pullback" in r.lower() for r in sig.reasons))


if __name__ == '__main__':
    unittest.main()

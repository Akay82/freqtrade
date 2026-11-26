"""Unit and integration tests for SwingBacktester and WalkForwardValidator."""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scripts.engine.backtester import (
    SwingBacktester,
    BacktestConfig,
    BacktestTrade,
    compute_metrics,
)
from scripts.engine.walk_forward import WalkForwardValidator


class TestBacktesterAndWalkForward(unittest.TestCase):

    def test_compute_metrics_all_12_fields(self):
        trades = [
            BacktestTrade(1, "StratA", "BULLISH_TREND", "2026-01-01", "2026-01-02", 100.0, 105.0, 1.0, 98.0, 106.0, 82.0, 0.2, 0.1, 15.0, 5.0, 2.0, 24.0, "TAKE_PROFIT"),
            BacktestTrade(2, "StratA", "BULLISH_TREND", "2026-01-03", "2026-01-04", 105.0, 103.0, 1.0, 103.0, 110.0, 75.0, 0.2, 0.1, -5.0, -2.0, -1.0, 12.0, "STOP_LOSS"),
            BacktestTrade(3, "StratA", "BULLISH_TREND", "2026-01-05", "2026-01-06", 103.0, 101.0, 1.0, 101.0, 108.0, 70.0, 0.2, 0.1, -5.0, -2.0, -1.0, 10.0, "STOP_LOSS"),
            BacktestTrade(4, "StratB", "BULLISH_TREND", "2026-01-07", "2026-01-09", 101.0, 108.0, 1.0, 99.0, 108.0, 88.0, 0.2, 0.1, 20.0, 7.0, 3.0, 48.0, "TAKE_PROFIT"),
        ]
        equity_series = [1000.0, 1015.0, 1010.0, 1005.0, 1025.0]
        m = compute_metrics(trades, equity_series, 1000.0)

        self.assertEqual(m.total_trades, 4)
        self.assertAlmostEqual(m.net_profit, 25.0, places=2)
        self.assertAlmostEqual(m.total_return_pct, 2.5, places=2)
        self.assertAlmostEqual(m.win_rate_pct, 50.0, places=1)
        self.assertAlmostEqual(m.profit_factor, 3.5, places=2)  # 35 / 10
        self.assertEqual(m.max_losing_streak, 2)
        self.assertAlmostEqual(m.expectancy, 6.25, places=2)
        self.assertGreater(m.max_drawdown_pct, 0.0)

    def test_backtester_on_real_btc_data(self):
        df_1d = pd.read_feather("user_data/data/binance/BTC_USDT-1d.feather")
        df_4h = pd.read_feather("user_data/data/binance/BTC_USDT-4h.feather")
        df_1h = pd.read_feather("user_data/data/binance/BTC_USDT-1h.feather")

        # Use last 1500 1h bars (~2 months) for fast deterministic test
        df_1h_sub = df_1h.iloc[-1500:].copy()
        cfg = BacktestConfig(initial_capital=1000.0, warmup_bars=200)
        res = SwingBacktester(cfg).run(df_1d, df_4h, df_1h_sub, run_id="test-real-btc")

        self.assertEqual(res.run_id, "test-real-btc")
        self.assertIn("btc_buy_hold_pct", res.benchmarks)
        self.assertIn("ema_crossover_pct", res.benchmarks)
        self.assertGreater(len(res.equity_curve), 10)

    def test_walk_forward_validation(self):
        df_1d = pd.read_feather("user_data/data/binance/BTC_USDT-1d.feather")
        df_4h = pd.read_feather("user_data/data/binance/BTC_USDT-4h.feather")
        df_1h = pd.read_feather("user_data/data/binance/BTC_USDT-1h.feather")

        # Use last 2200 1h bars with 1000 train, 500 test, 500 step -> 2 rolling windows
        df_1h_sub = df_1h.iloc[-2200:].copy()
        wf = WalkForwardValidator(
            train_bars_1h=1000,
            test_bars_1h=500,
            step_bars_1h=500,
            candidate_scores=[65.0],
        )
        summary = wf.run(df_1d, df_4h, df_1h_sub, BacktestConfig(initial_capital=1000.0, warmup_bars=200))
        self.assertEqual(summary.total_windows, 2)
        self.assertEqual(len(summary.windows), 2)


if __name__ == '__main__':
    unittest.main()

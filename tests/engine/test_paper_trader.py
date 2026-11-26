"""Integration tests for PaperTradingEngine restart recovery & attribution."""

import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.engine.paper_trader import PaperTradingEngine
from scripts.engine.storage import EngineStorage


class TestPaperTraderAndAttribution(unittest.TestCase):

    def test_restart_persistence_and_attribution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "paper_test.sqlite"
            storage1 = EngineStorage(db_path=db_path)
            engine1 = PaperTradingEngine(initial_capital=1000.0, storage=storage1)
            self.assertFalse(engine1.LIVE_TRADING_ENABLED)

            # Simulate opening a persisted position
            now_dt = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)
            pos = {
                "position_id": "pos-restart-01",
                "strategy_name": "StrategyA_TrendFollowing",
                "regime_at_entry": "BULLISH_TREND",
                "entry_time_utc": now_dt.isoformat(),
                "entry_price": 60000.0,
                "quantity": 0.01,
                "entry_fee": 0.6,
                "entry_cost": 600.6,
                "initial_stop_loss": 58800.0,
                "stop_loss": 58800.0,
                "take_profit": 63000.0,
                "partial_tp_price": 61800.0,
                "partial_taken": 0,
                "signal_score": 84.0,
                "ml_probability": 0.68,
                "entry_conditions_json": '["4H SuperTrend bullish"]',
            }
            with storage1._lock, storage1._connect() as conn:
                conn.execute(
                    """INSERT INTO paper_positions
                       (position_id, strategy_name, regime_at_entry, entry_time_utc,
                        entry_price, quantity, entry_fee, entry_cost, initial_stop_loss,
                        stop_loss, take_profit, partial_tp_price, partial_taken,
                        signal_score, ml_probability, entry_conditions_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
                    (
                        pos["position_id"], pos["strategy_name"], pos["regime_at_entry"],
                        pos["entry_time_utc"], pos["entry_price"], pos["quantity"],
                        pos["entry_fee"], pos["entry_cost"], pos["initial_stop_loss"],
                        pos["stop_loss"], pos["take_profit"], pos["partial_tp_price"],
                        pos["signal_score"], pos["ml_probability"], pos["entry_conditions_json"],
                    ),
                )
            engine1.cash_balance = 399.4
            engine1._persist_account()

            # Simulate process restart: new EngineStorage + new PaperTradingEngine
            storage2 = EngineStorage(db_path=db_path)
            engine2 = PaperTradingEngine(initial_capital=1000.0, storage=storage2)

            self.assertAlmostEqual(engine2.cash_balance, 399.4, places=2)
            self.assertEqual(len(engine2.open_positions), 1)
            self.assertEqual(engine2.open_positions[0]["position_id"], "pos-restart-01")

            # Close position at Take Profit and check attribution
            closed = engine2._close_position(
                engine2.open_positions[0],
                exit_price=63000.0,
                exit_reason="TAKE_PROFIT",
                now_utc=now_dt + timedelta(hours=20),
            )
            self.assertEqual(closed["exit_reason"], "TAKE_PROFIT")
            self.assertGreater(closed["net_pnl"], 0.0)
            self.assertEqual(len(engine2.open_positions), 0)

            attr = storage2.get_attribution_summary()
            self.assertEqual(attr["total_closed_trades"], 1)
            self.assertIn("StrategyA_TrendFollowing", attr["by_strategy"])
            self.assertIn("BULLISH_TREND", attr["by_regime"])
            self.assertIn("80-100 (Strong)", attr["by_score_range"])

    def test_paper_trader_step_on_real_btc_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = EngineStorage(db_path=Path(tmpdir) / "step_test.sqlite")
            engine = PaperTradingEngine(initial_capital=1000.0, storage=storage)

            df_1d = pd.read_feather("user_data/data/binance/BTC_USDT-1d.feather")
            df_4h = pd.read_feather("user_data/data/binance/BTC_USDT-4h.feather")
            df_1h = pd.read_feather("user_data/data/binance/BTC_USDT-1h.feather").iloc[-300:].copy()

            res = engine.step(df_1d, df_4h, df_1h, check_freshness=False)
            self.assertEqual(res["status"], "OK")
            self.assertIn(res["action"], ["BUY", "HOLD", "EXIT", "NONE"])


if __name__ == '__main__':
    unittest.main()

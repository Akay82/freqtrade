"""Unit tests for RiskManager using standard unittest."""

import unittest
from datetime import datetime, timezone, timedelta
from scripts.engine.risk_manager import RiskManager, RiskConfig


class TestRiskManager(unittest.TestCase):

    def setUp(self):
        self.config = RiskConfig(
            risk_per_trade_pct=0.01,
            max_open_positions=2,
            daily_loss_limit_pct=0.03,
            max_drawdown_limit_pct=0.10,
            fee_rate=0.001,
            slippage_bps=5.0,
            min_reward_risk_ratio=1.5,
        )
        self.rm = RiskManager(initial_equity=1000.0, config=self.config)

    def test_position_sizing_accounts_for_fees_and_slippage(self):
        # Equity = 1000, Risk = 1% -> $10 max risk
        # Entry = 60000, SL = 58800 (2% stop = 1200), TP = 63000 (5% = 3000, RR = 2.5)
        res = self.rm.evaluate_trade_risk(
            entry_price=60000.0,
            stop_loss_price=58800.0,
            take_profit_price=63000.0,
            available_balance=1000.0,
            open_positions_count=0,
            signal_key="sig-1",
        )
        self.assertTrue(res.approved)
        self.assertGreater(res.quantity, 0.0)
        self.assertLessEqual(res.notional_value, 1000.0)
        self.assertLessEqual(res.risk_amount, 10.01)  # Must never exceed 1% equity risk
        self.assertGreater(res.estimated_fees, 0.0)
        self.assertGreater(res.estimated_slippage, 0.0)
        self.assertAlmostEqual(res.reward_risk_ratio, 2.5, places=2)

    def test_atr_stops_calculation(self):
        sl, tp, partial = self.rm.calculate_atr_stops(
            entry_price=60000.0,
            atr=600.0,
            swing_low=58500.0,
            resistance=64000.0,
        )
        self.assertLess(sl, 60000.0)
        self.assertGreater(tp, 60000.0)
        self.assertGreater(partial, 60000.0)
        self.assertLessEqual(partial, tp)

    def test_invalid_stop_and_low_rr_rejected(self):
        # Stop above entry
        res_bad_sl = self.rm.evaluate_trade_risk(60000.0, 60500.0, 63000.0, 1000.0)
        self.assertFalse(res_bad_sl.approved)

        # R:R too low (1.0 < 1.5)
        res_low_rr = self.rm.evaluate_trade_risk(60000.0, 58800.0, 61200.0, 1000.0)
        self.assertFalse(res_low_rr.approved)
        self.assertTrue(any("reward/risk" in r.lower() for r in res_low_rr.rejection_reasons))

    def test_duplicate_signal_prevention_and_max_positions(self):
        res1 = self.rm.evaluate_trade_risk(60000.0, 58800.0, 63000.0, 1000.0, 0, signal_key="dup-1")
        self.assertTrue(res1.approved)

        # Same signal_key rejected
        res_dup = self.rm.evaluate_trade_risk(60000.0, 58800.0, 63000.0, 1000.0, 1, signal_key="dup-1")
        self.assertFalse(res_dup.approved)
        self.assertTrue(any("duplicate" in r.lower() for r in res_dup.rejection_reasons))

        # Max open positions (2) reached
        res_max = self.rm.evaluate_trade_risk(60000.0, 58800.0, 63000.0, 1000.0, 2, signal_key="new-1")
        self.assertFalse(res_max.approved)
        self.assertTrue(any("maximum open positions" in r.lower() for r in res_max.rejection_reasons))

    def test_daily_loss_lockout_and_utc_reset(self):
        now = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
        self.rm.update_equity(965.0, realized_pnl=-35.0, now_utc=now)  # -3.5% > 3% limit
        self.assertTrue(self.rm.daily_loss_lockout)

        res_locked = self.rm.evaluate_trade_risk(60000.0, 58800.0, 63000.0, 965.0)
        self.assertFalse(res_locked.approved)

        # Next UTC day resets daily loss lockout
        tomorrow = now + timedelta(days=1)
        self.rm.update_equity(965.0, realized_pnl=0.0, now_utc=tomorrow)
        self.assertFalse(self.rm.daily_loss_lockout)

    def test_max_drawdown_lockout_and_manual_kill_switch(self):
        now = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
        self.rm.update_equity(895.0, realized_pnl=-105.0, now_utc=now)  # -10.5% >= 10%
        self.assertTrue(self.rm.max_drawdown_lockout)

        # Even on next day, max drawdown stays locked until manually reset
        self.rm.update_equity(895.0, realized_pnl=0.0, now_utc=now + timedelta(days=1))
        self.assertTrue(self.rm.max_drawdown_lockout)

        self.rm.reset_max_drawdown_lockout()
        self.assertFalse(self.rm.max_drawdown_lockout)

        # Manual kill switch
        self.rm.set_manual_kill_switch(True, "Emergency test")
        res = self.rm.evaluate_trade_risk(60000.0, 58800.0, 63000.0, 895.0)
        self.assertFalse(res.approved)
        self.assertTrue(any("kill switch" in r.lower() for r in res.rejection_reasons))


if __name__ == '__main__':
    unittest.main()

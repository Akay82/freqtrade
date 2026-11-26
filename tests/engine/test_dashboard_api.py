"""Integration tests for EngineStorage and FastAPI Engine Router."""

import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from scripts.comparison_dashboard import app
from scripts.engine.storage import EngineStorage


class TestDashboardAndStorage(unittest.TestCase):

    def test_engine_storage_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_engine.sqlite"
            storage = EngineStorage(db_path=db_path)

            storage.save_regime_snapshot({
                "timestamp": "2026-11-26T00:00:00+00:00",
                "regime": "BULLISH_TREND",
                "confidence_score": 85.0,
                "data_quality_status": "HEALTHY",
                "summary_reason": "Test",
                "supporting_indicators": {"adx_14": 26.0},
            })
            storage.save_signal_evaluation({
                "timestamp": "2026-11-26T00:00:00+00:00",
                "final_action": "NONE",
                "active_regime": "BULLISH_TREND",
                "explanation": "No setup",
                "rejections": [{"strategy": "StrategyA", "reason": "Test"}],
            })
            sigs = storage.get_recent_signals(limit=5)
            self.assertEqual(len(sigs), 1)
            self.assertEqual(sigs[0]["regime"], "BULLISH_TREND")

    def test_fastapi_engine_endpoints(self):
        client = TestClient(app)

        r_mkt = client.get("/api/engine/market-overview")
        self.assertEqual(r_mkt.status_code, 200)
        mkt_json = r_mkt.json()
        self.assertEqual(mkt_json["pair"], "BTC/USDT")
        self.assertIn("regime", mkt_json)
        self.assertIn("mtf", mkt_json)

        r_sig = client.get("/api/engine/signals")
        self.assertEqual(r_sig.status_code, 200)
        sig_json = r_sig.json()
        self.assertIn("current", sig_json)
        self.assertEqual(len(sig_json["strategies"]), 4)

        r_risk = client.get("/api/engine/risk")
        self.assertEqual(r_risk.status_code, 200)
        self.assertFalse(r_risk.json()["manual_kill_switch"])

        # Toggle kill switch on then off
        r_ks_on = client.post("/api/engine/risk/kill-switch", json={"active": True, "reason": "Test"})
        self.assertEqual(r_ks_on.status_code, 200)
        self.assertTrue(r_ks_on.json()["manual_kill_switch"])
        r_ks_off = client.post("/api/engine/risk/kill-switch", json={"active": False, "reason": "Reset"})
        self.assertFalse(r_ks_off.json()["manual_kill_switch"])

        r_ml = client.get("/api/engine/ml-status")
        self.assertEqual(r_ml.status_code, 200)

        r_attr = client.get("/api/engine/attribution")
        self.assertEqual(r_attr.status_code, 200)

        r_home = client.get("/")
        self.assertEqual(r_home.status_code, 200)
        self.assertIn("Market &amp; Signals", r_home.text)


if __name__ == '__main__':
    unittest.main()

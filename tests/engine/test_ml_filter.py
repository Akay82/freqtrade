"""Unit tests for MLSignalFilter using standard unittest."""

import tempfile
import unittest
from pathlib import Path
import pandas as pd

from scripts.engine.data_validator import DataValidator
from scripts.engine.indicators import IndicatorEngine
from scripts.engine.price_action import PriceActionDetector
from scripts.engine.regime_detector import RegimeDetector
from scripts.engine.ml_filter import (
    MLSignalFilter,
    MLConfig,
    extract_features,
    generate_triple_barrier_labels,
    FEATURE_COLUMNS,
)
from scripts.engine.storage import EngineStorage


class TestMLFilter(unittest.TestCase):

    def setUp(self):
        df_1h = pd.read_feather("user_data/data/binance/BTC_USDT-1h.feather").iloc[-1200:].copy()
        df_1h = DataValidator().clean_and_normalize(df_1h, "1h", drop_incomplete=False)
        df_1h = IndicatorEngine().compute_all_indicators(df_1h)
        df_1h = PriceActionDetector().compute_all_price_action(df_1h)
        self.df_enriched = RegimeDetector().classify_series(df_1h)

    def test_feature_extraction_and_triple_barrier_labels(self):
        feats = extract_features(self.df_enriched)
        self.assertEqual(list(feats.columns), FEATURE_COLUMNS)
        self.assertEqual(len(feats), len(self.df_enriched))

        cfg = MLConfig(horizon_bars=24)
        labels = generate_triple_barrier_labels(self.df_enriched, cfg)
        # Last horizon_bars rows must be NaN (no future leakage)
        self.assertTrue(labels.iloc[-24:].isna().all())
        valid_labels = labels.dropna()
        self.assertTrue(set(valid_labels.unique()).issubset({0.0, 1.0}))

    def test_disabled_filter_always_passes(self):
        ml = MLSignalFilter(MLConfig(enabled=False))
        passed, prob, msg = ml.passes_filter(self.df_enriched, candle_idx=-1)
        self.assertTrue(passed)
        self.assertIsNone(prob)
        self.assertIn("disabled", msg.lower())

    def test_chronological_training_and_prediction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = EngineStorage(db_path=Path(tmpdir) / "ml_test.sqlite")
            cfg = MLConfig(enabled=True, min_probability=0.45, horizon_bars=24)
            ml = MLSignalFilter(config=cfg, storage=storage)

            metrics = ml.train_chronological(self.df_enriched, version="test_v1", save_model=True)
            self.assertEqual(metrics.version, "test_v1")
            self.assertGreater(metrics.train_samples, 100)
            self.assertGreater(metrics.test_samples, 20)
            self.assertTrue(0.0 <= metrics.brier_score <= 1.0)
            self.assertTrue(0.0 <= metrics.roc_auc <= 1.0)

            prob = ml.predict_success_probability(self.df_enriched, candle_idx=-1)
            self.assertIsNotNone(prob)
            self.assertTrue(0.0 <= prob <= 1.0)

            # Verify storage registration & reload
            ml2 = MLSignalFilter(config=cfg, storage=storage)
            self.assertTrue(ml2.load_latest())
            self.assertEqual(ml2.active_version, "test_v1")


if __name__ == '__main__':
    unittest.main()

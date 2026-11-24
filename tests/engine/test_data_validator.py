"""Unit tests for DataValidator using standard unittest."""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scripts.engine.data_validator import DataValidator, ValidationResult


class TestDataValidator(unittest.TestCase):

    def setUp(self):
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        dates = [start + timedelta(hours=i) for i in range(50)]
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(50) * 0.5)
        high = close + np.random.uniform(0.5, 1.5, size=50)
        low = close - np.random.uniform(0.5, 1.5, size=50)
        open_p = low + np.random.uniform(0.1, 0.8, size=50) * (high - low)
        volume = np.random.uniform(10, 100, size=50)

        self.sample_valid_ohlcv = pd.DataFrame({
            'date': dates,
            'open': open_p,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        self.validator = DataValidator()

    def test_validator_valid_dataframe(self):
        current_time = self.sample_valid_ohlcv['date'].iloc[-1] + timedelta(hours=2)
        res = self.validator.validate_dataframe(self.sample_valid_ohlcv, timeframe='1h', current_time=current_time)

        self.assertTrue(res.is_valid)
        self.assertEqual(res.total_candles, 50)
        self.assertEqual(res.duplicate_count, 0)
        self.assertEqual(res.invalid_ohlcv_count, 0)
        self.assertEqual(len(res.missing_gaps), 0)
        self.assertFalse(res.is_stale)

    def test_validator_empty_dataframe(self):
        res = self.validator.validate_dataframe(pd.DataFrame(), timeframe='1h')
        self.assertFalse(res.is_valid)
        self.assertIn("empty", res.errors[0].lower())

    def test_validator_missing_columns(self):
        bad_df = pd.DataFrame({'date': [datetime.now(timezone.utc)], 'open': [100.0]})
        res = self.validator.validate_dataframe(bad_df, timeframe='1h')
        self.assertFalse(res.is_valid)
        self.assertTrue(any("missing required" in err.lower() for err in res.errors))

    def test_validator_duplicate_timestamps(self):
        df_dup = pd.concat([self.sample_valid_ohlcv, self.sample_valid_ohlcv.iloc[[5]]]).sort_values('date')
        current_time = self.sample_valid_ohlcv['date'].iloc[-1] + timedelta(hours=2)

        res = self.validator.validate_dataframe(df_dup, timeframe='1h', current_time=current_time)
        self.assertEqual(res.duplicate_count, 1)
        self.assertEqual(res.total_candles, 50)

    def test_validator_invalid_ohlcv(self):
        df_inv = self.sample_valid_ohlcv.copy()
        df_inv.loc[10, 'high'] = 10.0
        df_inv.loc[10, 'low'] = 50.0
        df_inv.loc[20, 'volume'] = -5.0

        current_time = self.sample_valid_ohlcv['date'].iloc[-1] + timedelta(hours=2)
        res = self.validator.validate_dataframe(df_inv, timeframe='1h', current_time=current_time)

        self.assertEqual(res.invalid_ohlcv_count, 2)
        self.assertEqual(res.total_candles, 48)

    def test_validator_gap_detection(self):
        df_gap = self.sample_valid_ohlcv.drop(index=list(range(15, 20))).reset_index(drop=True)
        current_time = self.sample_valid_ohlcv['date'].iloc[-1] + timedelta(hours=2)

        res = self.validator.validate_dataframe(df_gap, timeframe='1h', current_time=current_time)
        self.assertEqual(len(res.missing_gaps), 1)
        start_gap, end_gap, count = res.missing_gaps[0]
        self.assertEqual(count, 5)

    def test_validator_staleness(self):
        validator = DataValidator(max_staleness_multiplier=2.0)
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        df = pd.DataFrame({
            'date': [start],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [102.0],
            'volume': [50.0]
        })
        current_time = start + timedelta(hours=10)
        res = validator.validate_dataframe(df, timeframe='1h', check_freshness=True, current_time=current_time)
        self.assertTrue(res.is_stale)
        self.assertTrue(any("stale" in w.lower() for w in res.warnings))

    def test_validator_drop_incomplete_candle(self):
        last_candle_time = self.sample_valid_ohlcv['date'].iloc[-1]
        current_time = last_candle_time + timedelta(minutes=30)

        res = self.validator.validate_dataframe(
            self.sample_valid_ohlcv,
            timeframe='1h',
            current_time=current_time,
            drop_incomplete_candle=True
        )
        self.assertEqual(res.total_candles, len(self.sample_valid_ohlcv) - 1)
        self.assertTrue(any("dropped incomplete" in w.lower() for w in res.warnings))


if __name__ == '__main__':
    unittest.main()

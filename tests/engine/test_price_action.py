"""Unit tests for PriceActionDetector using standard unittest."""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scripts.engine.price_action import PriceActionDetector


class TestPriceActionDetector(unittest.TestCase):

    def setUp(self):
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        dates = [start + timedelta(hours=i) for i in range(120)]
        t = np.linspace(0, 4 * np.pi, 120)
        base_price = 100.0 + (t * 2.0) + (np.sin(t) * 10.0)

        open_p = base_price - 0.5
        close = base_price + 0.5
        high = base_price + 2.0
        low = base_price - 2.0
        volume = np.full(120, 500.0)

        self.wave_candles = pd.DataFrame({
            'date': dates,
            'open': open_p,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        self.pa = PriceActionDetector()

    def test_candlestick_anatomy_and_patterns(self):
        dates = [datetime(2026, 1, 1, i, 0, tzinfo=timezone.utc) for i in range(4)]
        df = pd.DataFrame({
            'date': dates,
            'open':  [100.0, 100.0, 101.0, 98.0],
            'high':  [102.0, 100.2, 105.0, 102.0],
            'low':   [98.0,   95.0, 100.8, 97.5],
            'close': [101.0, 100.5, 101.2, 101.5],
            'volume': [100.0, 200.0, 150.0, 300.0]
        })
        res = self.pa.add_candlestick_structure(df)

        self.assertIn('is_bullish_hammer', res.columns)
        self.assertIn('is_shooting_star', res.columns)
        self.assertIn('is_bullish_engulfing', res.columns)

        # Row 1 should be detected as hammer
        self.assertTrue(res['is_bullish_hammer'].iloc[1])
        # Row 2 should be detected as shooting star
        self.assertTrue(res['is_shooting_star'].iloc[2])

    def test_swing_pivot_detection(self):
        pa = PriceActionDetector(pivot_window=4)
        res = pa.add_swing_pivots(self.wave_candles)

        self.assertIn('is_swing_high', res.columns)
        self.assertIn('is_swing_low', res.columns)
        self.assertIn('last_swing_high', res.columns)
        self.assertIn('last_swing_low', res.columns)

        self.assertGreaterEqual(res['is_swing_high'].sum(), 1)
        self.assertGreaterEqual(res['is_swing_low'].sum(), 1)

        # Verify no lookahead bias:
        confirmed_high_indices = np.where(res['is_swing_high'].values)[0]
        for idx in confirmed_high_indices:
            self.assertGreaterEqual(idx, 2 * pa.pivot_window)

    def test_market_structure_hh_hl(self):
        pa = PriceActionDetector(pivot_window=3)
        res = pa.compute_all_price_action(self.wave_candles)

        self.assertIn('is_higher_high', res.columns)
        self.assertIn('is_higher_low', res.columns)
        self.assertIn('market_structure_trend', res.columns)
        self.assertGreaterEqual(res['is_higher_high'].sum(), 1)

    def test_support_and_resistance(self):
        res = self.pa.add_support_resistance(self.wave_candles, window=20)

        self.assertIn('resistance_level', res.columns)
        self.assertIn('support_level', res.columns)
        self.assertIn('dist_to_resistance_pct', res.columns)
        self.assertIn('dist_to_support_pct', res.columns)

        valid_mask = ~res['resistance_level'].isna()
        self.assertTrue((res.loc[valid_mask, 'resistance_level'] >= res.loc[valid_mask, 'support_level']).all())


if __name__ == '__main__':
    unittest.main()

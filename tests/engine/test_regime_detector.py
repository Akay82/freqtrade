"""Unit tests for RegimeDetector using standard unittest."""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scripts.engine.regime_detector import RegimeDetector, MarketRegime, RegimeConfig


class TestRegimeDetector(unittest.TestCase):

    def setUp(self):
        self.detector = RegimeDetector()

    def _create_synthetic_df(self, n=60, **kwargs):
        """Helper to create synthetic enriched dataframe with default neutral columns."""
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        dates = [start + timedelta(hours=i) for i in range(n)]
        data = {
            'date': dates,
            'close': np.full(n, 100.0),
            'ema_20': np.full(n, 100.0),
            'ema_50': np.full(n, 100.0),
            'ema_200': np.full(n, 100.0),
            'adx_14': np.full(n, 15.0),
            'plus_di': np.full(n, 20.0),
            'minus_di': np.full(n, 20.0),
            'rsi_14': np.full(n, 50.0),
            'atr_pct': np.full(n, 1.0),
            'bb_width': np.full(n, 0.05),
            'market_structure_trend': np.zeros(n, dtype=int),
            'supertrend_direction': np.ones(n, dtype=int),
            'ema_50_slope': np.zeros(n),
            'ttm_squeeze_on': np.zeros(n, dtype=bool)
        }
        for k, v in kwargs.items():
            if isinstance(v, (int, float, bool)):
                data[k] = np.full(n, v)
            else:
                data[k] = v
        return pd.DataFrame(data)

    def test_insufficient_data(self):
        df_short = self._create_synthetic_df(n=15)
        res = self.detector.classify_candle(df_short, candle_idx=-1)
        self.assertEqual(res.regime, MarketRegime.UNCERTAIN)
        self.assertEqual(res.data_quality_status, "INSUFFICIENT_DATA")

    def test_high_volatility_detection(self):
        # Baseline ATR% is 1.0, current spike to 3.0
        df = self._create_synthetic_df(n=60, atr_pct=1.0)
        df.loc[59, 'atr_pct'] = 3.2

        res = self.detector.classify_candle(df, candle_idx=59)
        self.assertEqual(res.regime, MarketRegime.HIGH_VOLATILITY)
        self.assertGreaterEqual(res.confidence_score, 60.0)
        self.assertIn("spike", res.summary_reason.lower())

    def test_low_volatility_squeeze_detection(self):
        # Squeeze on with low ADX
        df = self._create_synthetic_df(n=60, ttm_squeeze_on=True, adx_14=12.0)
        res = self.detector.classify_candle(df, candle_idx=59)
        self.assertEqual(res.regime, MarketRegime.LOW_VOLATILITY)
        self.assertGreaterEqual(res.confidence_score, 65.0)

    def test_bullish_trend_detection(self):
        df = self._create_synthetic_df(
            n=60,
            close=110.0,
            ema_50=105.0,
            ema_200=95.0,
            ema_50_slope=0.15,
            adx_14=28.0,
            plus_di=30.0,
            minus_di=15.0,
            market_structure_trend=1,
            supertrend_direction=1
        )
        res = self.detector.classify_candle(df, candle_idx=59)
        self.assertEqual(res.regime, MarketRegime.BULLISH_TREND)
        self.assertGreaterEqual(res.confidence_score, 75.0)
        self.assertEqual(res.supporting_indicators['market_structure_trend'], 1)

    def test_bearish_trend_detection(self):
        df = self._create_synthetic_df(
            n=60,
            close=85.0,
            ema_50=92.0,
            ema_200=105.0,
            ema_50_slope=-0.18,
            adx_14=32.0,
            plus_di=12.0,
            minus_di=35.0,
            market_structure_trend=-1,
            supertrend_direction=-1
        )
        res = self.detector.classify_candle(df, candle_idx=59)
        self.assertEqual(res.regime, MarketRegime.BEARISH_TREND)
        self.assertGreaterEqual(res.confidence_score, 75.0)

    def test_sideways_range_detection(self):
        df = self._create_synthetic_df(
            n=60,
            close=100.0,
            ema_50=100.0,
            ema_200=100.0,
            ema_50_slope=0.001,
            adx_14=14.0,  # Below range threshold
            rsi_14=50.0,
            market_structure_trend=0
        )
        res = self.detector.classify_candle(df, candle_idx=59)
        self.assertEqual(res.regime, MarketRegime.SIDEWAYS)
        self.assertGreaterEqual(res.confidence_score, 60.0)

    def test_uncertain_regime(self):
        # Conflicting: EMA 50 > 200 (bullish) but negative slope, bearish structure and supertrend
        df = self._create_synthetic_df(
            n=60,
            ema_50=105.0,
            ema_200=95.0,
            ema_50_slope=-0.25,
            adx_14=26.0,
            plus_di=15.0,
            minus_di=30.0,
            market_structure_trend=-1,
            supertrend_direction=-1
        )
        res = self.detector.classify_candle(df, candle_idx=59)
        self.assertEqual(res.regime, MarketRegime.UNCERTAIN)
        self.assertTrue(len(res.conflicting_indicators) > 0)


if __name__ == '__main__':
    unittest.main()

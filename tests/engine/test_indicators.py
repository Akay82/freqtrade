"""Unit tests for IndicatorEngine using standard unittest."""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scripts.engine.indicators import IndicatorEngine


class TestIndicatorEngine(unittest.TestCase):

    def setUp(self):
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        dates = [start + timedelta(hours=i) for i in range(250)]
        np.random.seed(123)
        t = np.linspace(0, 10, 250)
        close = 100.0 + 3.0 * t + np.sin(t) * 5.0 + np.cumsum(np.random.randn(250) * 0.4)
        high = close + np.random.uniform(0.5, 2.0, size=250)
        low = close - np.random.uniform(0.5, 2.0, size=250)
        open_p = low + (high - low) * np.random.uniform(0.2, 0.8, size=250)
        volume = np.random.uniform(100, 1000, size=250)

        self.candle_series = pd.DataFrame({
            'date': dates,
            'open': open_p,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        self.ie = IndicatorEngine()

    def test_indicator_warmup_check(self):
        ie = IndicatorEngine(warmup_candles=200)
        self.assertTrue(ie.has_sufficient_warmup(self.candle_series))
        self.assertFalse(ie.has_sufficient_warmup(self.candle_series.iloc[:100]))

    def test_indicator_computation(self):
        df = self.ie.compute_all_indicators(self.candle_series)

        # 1. Trend indicators
        self.assertIn('ema_20', df.columns)
        self.assertIn('ema_50', df.columns)
        self.assertIn('ema_100', df.columns)
        self.assertIn('ema_200', df.columns)
        self.assertIn('adx_14', df.columns)
        self.assertIn('plus_di', df.columns)
        self.assertIn('minus_di', df.columns)
        self.assertIn('ema_20_slope', df.columns)

        # Verify EMA 200 is populated at the end
        self.assertFalse(np.isnan(df['ema_200'].iloc[-1]))

        # 2. Momentum indicators
        self.assertIn('rsi_14', df.columns)
        self.assertIn('macd', df.columns)
        self.assertIn('macd_signal', df.columns)
        self.assertIn('macd_hist', df.columns)
        self.assertIn('roc_14', df.columns)
        self.assertIn('stoch_rsi_k', df.columns)

        rsi_last = df['rsi_14'].iloc[-1]
        self.assertTrue(0 <= rsi_last <= 100)

        # 3. Volatility indicators
        self.assertIn('atr_14', df.columns)
        self.assertIn('atr_pct', df.columns)
        self.assertIn('bb_upper', df.columns)
        self.assertIn('bb_middle', df.columns)
        self.assertIn('bb_lower', df.columns)
        self.assertIn('bb_width', df.columns)
        self.assertIn('hist_volatility_20', df.columns)

        self.assertGreaterEqual(df['bb_upper'].iloc[-1], df['bb_middle'].iloc[-1])
        self.assertGreaterEqual(df['bb_middle'].iloc[-1], df['bb_lower'].iloc[-1])

        # 4. Volume indicators
        self.assertIn('volume_sma_20', df.columns)
        self.assertIn('relative_volume', df.columns)
        self.assertIn('obv', df.columns)
        self.assertIn('cmf_20', df.columns)
        self.assertGreater(df['relative_volume'].iloc[-1], 0)

        # 5. Breakout channel indicators
        self.assertIn('supertrend', df.columns)
        self.assertIn('supertrend_direction', df.columns)
        self.assertIn(df['supertrend_direction'].iloc[-1], [1, -1])
        self.assertIn('ttm_squeeze_on', df.columns)

    def test_indicator_multi_timeframe_merge(self):
        # Create 4h dataframe sampled from 1h
        df_4h = self.candle_series.set_index('date').resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()

        merged = self.ie.merge_informative_timeframes(self.candle_series, {'4h': df_4h})

        self.assertIn('4h_ema_50', merged.columns)
        self.assertIn('4h_rsi_14', merged.columns)
        self.assertIn('4h_supertrend_direction', merged.columns)
        self.assertEqual(len(merged), len(self.candle_series))


if __name__ == '__main__':
    unittest.main()

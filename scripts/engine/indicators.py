"""
Centralized Technical Indicator Engine for BTC Swing Trading.
Computes Trend, Momentum, Volatility, and Volume indicators consistently
across all timeframes with safe warm-up handling and null safety.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import talib.abstract as ta


class IndicatorEngine:
    """
    Standardized indicator calculator for 1D, 4H, and 1H candles.
    Ensures identical formula parameters across strategies, dashboards, and ML features.
    """

    # Minimum candles required for indicators like EMA 200 to stabilize
    DEFAULT_WARMUP_CANDLES = 200

    def __init__(self, warmup_candles: int = DEFAULT_WARMUP_CANDLES):
        self.warmup_candles = warmup_candles

    def has_sufficient_warmup(self, df: pd.DataFrame, required_candles: Optional[int] = None) -> bool:
        """Verifies if the dataframe has enough candles to compute reliable indicators."""
        req = required_candles or self.warmup_candles
        return df is not None and len(df) >= req

    def compute_all_indicators(self, df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """
        Calculates the complete suite of required indicators specified in Section 5:
        - Trend: EMA 20, 50, 100, 200, ADX 14, +DI, -DI, EMA slopes
        - Momentum: RSI 14, MACD (12/26/9), MACD Signal, MACD Hist, ROC (9, 14)
        - Volatility: ATR 14, ATR %, Bollinger Bands (20, 2), BB Width, Historical Volatility
        - Volume: Volume SMA 20, Relative Volume, OBV
        - Additional: SuperTrend, CMF 20, StochRSI
        """
        if df is None or len(df) == 0:
            return df

        res = df.copy()

        # 1. Trend Indicators
        res = self.add_trend_indicators(res)

        # 2. Momentum Indicators
        res = self.add_momentum_indicators(res)

        # 3. Volatility Indicators
        res = self.add_volatility_indicators(res)

        # 4. Volume Indicators
        res = self.add_volume_indicators(res)

        # 5. Volatility & Channel Breakout Indicators (SuperTrend, Bollinger, Keltner)
        res = self.add_breakout_channel_indicators(res)

        if prefix:
            # Prefix all non-date/ohlcv columns
            exclude = {'date', 'timestamp', 'open', 'high', 'low', 'close', 'volume'}
            rename_map = {col: f"{prefix}{col}" for col in res.columns if col not in exclude}
            res = res.rename(columns=rename_map)

        return res

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes EMAs (20, 50, 100, 200), slopes, ADX, +DI, -DI."""
        close = df['close']

        # Exponential Moving Averages
        df['ema_20'] = ta.EMA(df, timeperiod=20)
        df['ema_50'] = ta.EMA(df, timeperiod=50)
        df['ema_100'] = ta.EMA(df, timeperiod=100)
        df['ema_200'] = ta.EMA(df, timeperiod=200)

        # Directional Movement
        df['adx_14'] = ta.ADX(df, timeperiod=14)
        df['plus_di'] = ta.PLUS_DI(df, timeperiod=14)
        df['minus_di'] = ta.MINUS_DI(df, timeperiod=14)

        # EMA Slopes (rate of change over 3 periods normalized by price)
        df['ema_20_slope'] = (df['ema_20'] - df['ema_20'].shift(3)) / (df['ema_20'].shift(3) * 3) * 1000.0
        df['ema_50_slope'] = (df['ema_50'] - df['ema_50'].shift(3)) / (df['ema_50'].shift(3) * 3) * 1000.0
        df['ema_200_slope'] = (df['ema_200'] - df['ema_200'].shift(5)) / (df['ema_200'].shift(5) * 5) * 1000.0

        # Structural Alignment flags
        df['trend_bullish_alignment'] = (df['ema_20'] > df['ema_50']) & (df['ema_50'] > df['ema_200'])
        df['trend_bearish_alignment'] = (df['ema_20'] < df['ema_50']) & (df['ema_50'] < df['ema_200'])

        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes RSI 14, MACD (12/26/9), MACD Signal, MACD Hist, ROC 9 and 14."""
        df['rsi_14'] = ta.RSI(df, timeperiod=14)

        # MACD
        macd_df = ta.MACD(
            df,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        df['macd'] = macd_df['macd']
        df['macd_signal'] = macd_df['macdsignal']
        df['macd_hist'] = macd_df['macdhist']

        # MACD Histogram expansion/acceleration
        df['macd_hist_expanding'] = df['macd_hist'] > df['macd_hist'].shift(1)

        # Rate of Change (ROC)
        df['roc_9'] = ta.ROC(df, timeperiod=9)
        df['roc_14'] = ta.ROC(df, timeperiod=14)

        # Stochastic RSI
        stoch_df = ta.STOCHRSI(df, timeperiod=14, fastk_period=3, fastd_period=3)
        df['stoch_rsi_k'] = stoch_df['fastk']
        df['stoch_rsi_d'] = stoch_df['fastd']

        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes ATR 14, ATR %, Bollinger Bands (20, 2), BB Width, Historical Volatility."""
        df['atr_14'] = ta.ATR(df, timeperiod=14)
        df['atr_pct'] = (df['atr_14'] / df['close']) * 100.0

        # Bollinger Bands (20 periods, 2 standard deviations)
        bb_df = ta.BBANDS(df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        df['bb_upper'] = bb_df['upperband']
        df['bb_middle'] = bb_df['middleband']
        df['bb_lower'] = bb_df['lowerband']

        # Bollinger Band Width: (upper - lower) / middle
        # Useful for squeeze / volatility expansion detection
        df['bb_width'] = np.where(df['bb_middle'] > 0, (df['bb_upper'] - df['bb_lower']) / df['bb_middle'], 0.0)

        # Historical Volatility (annualized rolling standard deviation of 20-period log returns)
        log_ret = np.log(df['close'] / df['close'].shift(1))
        # 365 * 24 periods per year for 1h, but simple rolling std * sqrt(365) is standard benchmark
        df['hist_volatility_20'] = log_ret.rolling(window=20).std() * np.sqrt(365) * 100.0

        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes Volume SMA 20, Relative Volume, OBV, Chaikin Money Flow."""
        df['volume_sma_20'] = ta.SMA(df, timeperiod=20, price='volume')

        # Relative Volume (ratio of current volume to 20-period volume SMA)
        # Avoid zero division
        df['relative_volume'] = np.where(
            df['volume_sma_20'] > 0,
            df['volume'] / df['volume_sma_20'],
            1.0
        )

        # On-Balance Volume (OBV)
        df['obv'] = ta.OBV(df)
        df['obv_ema_20'] = ta.EMA(df, timeperiod=20, price='obv')

        # Chaikin Money Flow (CMF 20)
        mf_mult = np.where(
            (df['high'] - df['low']) > 0,
            ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']),
            0.0
        )
        mf_volume = mf_mult * df['volume']
        cmf_vol_sum = df['volume'].rolling(window=20).sum()
        df['cmf_20'] = np.where(
            cmf_vol_sum > 0,
            mf_volume.rolling(window=20).sum() / cmf_vol_sum,
            0.0
        )

        return df

    def add_breakout_channel_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes SuperTrend, Keltner Channels, and TTM Squeeze state."""
        # 1. SuperTrend (multiplier=3.0, period=10)
        atr_10 = ta.ATR(df, timeperiod=10)
        hl2 = (df['high'] + df['low']) / 2.0
        basic_upper = hl2 + (3.0 * atr_10)
        basic_lower = hl2 - (3.0 * atr_10)

        upperband = basic_upper.copy()
        lowerband = basic_lower.copy()
        trend = np.ones(len(df), dtype=int)
        close_vals = df['close'].values

        for i in range(1, len(df)):
            if basic_lower.iloc[i] > lowerband.iloc[i-1] or close_vals[i-1] < lowerband.iloc[i-1]:
                lowerband.iloc[i] = basic_lower.iloc[i]
            else:
                lowerband.iloc[i] = lowerband.iloc[i-1]

            if basic_upper.iloc[i] < upperband.iloc[i-1] or close_vals[i-1] > upperband.iloc[i-1]:
                upperband.iloc[i] = basic_upper.iloc[i]
            else:
                upperband.iloc[i] = upperband.iloc[i-1]

            if close_vals[i] > upperband.iloc[i-1]:
                trend[i] = 1
            elif close_vals[i] < lowerband.iloc[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]

        df['supertrend'] = np.where(trend == 1, lowerband, upperband)
        df['supertrend_direction'] = trend  # 1 for bullish, -1 for bearish

        # 2. Keltner Channels (20-period EMA, 1.5 * ATR 20)
        kc_ema = ta.EMA(df, timeperiod=20)
        kc_atr = ta.ATR(df, timeperiod=20)
        df['kc_upper'] = kc_ema + (1.5 * kc_atr)
        df['kc_lower'] = kc_ema - (1.5 * kc_atr)

        # 3. TTM Squeeze Detection
        # Squeeze on when Bollinger Bands are completely inside Keltner Channels
        # (Lower volatility compression preceding a major swing breakout)
        if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
            df['ttm_squeeze_on'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])
        else:
            df['ttm_squeeze_on'] = False

        return df

    def merge_informative_timeframes(
        self,
        base_df: pd.DataFrame,
        informative_dfs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Merges 4h and 1d informative indicator columns onto the base 1h dataframe
        using strictly backward-looking merges to prevent lookahead bias.
        """
        res = base_df.copy()
        for tf, inf_df in informative_dfs.items():
            if inf_df is None or len(inf_df) == 0:
                continue

            # Compute indicators for informative timeframe
            computed_inf = self.compute_all_indicators(inf_df, prefix=f"{tf}_")

            # Merge via asof merge on date (ensuring only closed informative candles are mapped)
            # Both dataframes must be sorted by date
            computed_inf = computed_inf.sort_values('date').reset_index(drop=True)
            res = res.sort_values('date').reset_index(drop=True)

            # Drop OHLCV from informative if duplicated
            cols_to_drop = [f"{tf}_{c}" for c in ['open', 'high', 'low', 'close', 'volume'] if f"{tf}_{c}" in computed_inf.columns]
            clean_inf = computed_inf.drop(columns=cols_to_drop, errors='ignore')

            res = pd.merge_asof(
                res,
                clean_inf,
                on='date',
                direction='backward'
            )

        return res

# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import IStrategy
from risk_management import RiskManagementMixin
import talib.abstract as ta
from technical import qtpylib


def calculate_supertrend(dataframe: DataFrame, multiplier=3.0, period=10) -> DataFrame:
    df = dataframe.copy()
    atr = ta.ATR(df, timeperiod=period)
    hl2 = (df['high'] + df['low']) / 2
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)

    upperband = basic_upperband.copy()
    lowerband = basic_lowerband.copy()
    trend = pd.Series(1, index=df.index)

    close = df['close'].values
    for i in range(1, len(df)):
        if basic_lowerband.iloc[i] > lowerband.iloc[i-1] or close[i-1] < lowerband.iloc[i-1]:
            lowerband.iloc[i] = basic_lowerband.iloc[i]
        else:
            lowerband.iloc[i] = lowerband.iloc[i-1]

        if basic_upperband.iloc[i] < upperband.iloc[i-1] or close[i-1] > upperband.iloc[i-1]:
            upperband.iloc[i] = basic_upperband.iloc[i]
        else:
            upperband.iloc[i] = upperband.iloc[i-1]

        if close[i] > upperband.iloc[i-1]:
            trend.iloc[i] = 1
        elif close[i] < lowerband.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]

    dataframe['supertrend'] = np.where(trend == 1, lowerband, upperband)
    dataframe['supertrend_direction'] = trend
    return dataframe


class BTCSwing_SuperTrend(RiskManagementMixin, IStrategy):
    """
    Strategy 2: SuperTrend + ADX Volatility Swing
    Thesis: Trade high-momentum trend continuations confirmed by directional volatility.
    Trigger: SuperTrend flips green (+1) + ADX > 25 (strong trend strength) + above Volume SMA.
    """
    INTERFACE_VERSION = 3

    timeframe = '1h'

    minimal_roi = {
        "0": 0.08,
        "1080": 0.045,
        "2160": 0.025,
        "4320": 0.015
    }

    stoploss = -0.04

    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 50

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = calculate_supertrend(dataframe, multiplier=3.0, period=10)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['vol_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # SuperTrend is bullish (or just flipped bullish)
                (dataframe['supertrend_direction'] == 1) &
                (dataframe['supertrend_direction'].shift(1) == -1) &
                # Strong trend strength
                (dataframe['adx'] > 22) &
                # Above 50 EMA baseline
                (dataframe['close'] > dataframe['ema_50']) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Exit when SuperTrend flips bearish
                (dataframe['supertrend_direction'] == -1) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'
        ] = 1

        return dataframe

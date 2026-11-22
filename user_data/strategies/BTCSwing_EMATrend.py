# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    informative,
    DecimalParameter,
    IntParameter,
)
import talib.abstract as ta
from technical import qtpylib
from risk_management import RiskManagementMixin


class BTCSwing_EMATrend(RiskManagementMixin, IStrategy):
    """
    Strategy 1: Multi-Timeframe EMA Trend Pullback
    Thesis: Trade 1h swings strictly in the direction of the 4h macro trend.
    Macro Filter (4h): EMA 50 > EMA 200 (Bullish structural regime).
    Trigger (1h): Pullback below EMA 20 or RSI < 42 with upward hook.
    """
    INTERFACE_VERSION = 3

    timeframe = '1h'

    minimal_roi = {
        "0": 0.06,
        "720": 0.035,
        "1440": 0.02,
        "2880": 0.012
    }

    stoploss = -0.045

    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.022
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 200

    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1h Trend Indicators
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        # Oscillators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=7)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # 4h Macro Bullish Filter
                (dataframe['ema_50_4h'] > dataframe['ema_200_4h']) &
                # 1h Trend structure: close is above 1h EMA 50
                (dataframe['close'] > dataframe['ema_50']) &
                # 1h Pullback: RSI is oversold/pullback level (< 42) and turning up
                (dataframe['rsi'] < 42) &
                (dataframe['rsi_fast'] > dataframe['rsi_fast'].shift(1)) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Exit when 1h RSI gets overbought
                (dataframe['rsi'] > 72) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'
        ] = 1

        return dataframe

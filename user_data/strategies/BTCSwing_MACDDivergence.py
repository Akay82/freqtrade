# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import IStrategy
from risk_management import RiskManagementMixin
import talib.abstract as ta
from technical import qtpylib


class BTCSwing_MACDDivergence(RiskManagementMixin, IStrategy):
    """
    Strategy 3: MACD Momentum Cross & Acceleration Swing
    Thesis: Identify momentum pivot points where MACD line crosses above signal line
            with accelerating histogram bars above the 100-period baseline.
    """
    INTERFACE_VERSION = 3

    timeframe = '1h'

    minimal_roi = {
        "0": 0.07,
        "720": 0.04,
        "1440": 0.025,
        "4320": 0.012
    }

    stoploss = -0.05

    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 120

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # MACD (12, 26, 9)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Trend filter
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # MACD cross above Signal
                (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])) &
                # Histogram is positive and expanding
                (dataframe['macdhist'] > 0) &
                (dataframe['macdhist'] > dataframe['macdhist'].shift(1)) &
                # Price is at or above EMA 100 baseline
                (dataframe['close'] >= dataframe['ema_100'] * 0.985) &
                # RSI not overbought
                (dataframe['rsi'] < 65) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Exit when MACD crosses below signal line or RSI overbought exhaustion
                (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])) |
                (dataframe['rsi'] > 75)
            ) & (dataframe['volume'] > 0),
            'exit_long'
        ] = 1

        return dataframe

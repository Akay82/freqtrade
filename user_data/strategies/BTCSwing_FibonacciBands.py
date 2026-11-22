# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import IStrategy
from risk_management import RiskManagementMixin
import talib.abstract as ta
from technical import qtpylib


class BTCSwing_FibonacciBands(RiskManagementMixin, IStrategy):
    """
    Strategy 10: Fibonacci Volatility Band Swing
    Thesis: Use Fibonacci multipliers of Average True Range (1.618, 2.618) around an EMA baseline.
            Enters when price hits extreme Fibonacci discount levels and prints a reversal candle.
    """
    INTERFACE_VERSION = 3

    timeframe = '1h'

    minimal_roi = {
        "0": 0.08,
        "720": 0.045,
        "1440": 0.025,
        "2880": 0.015
    }

    stoploss = -0.045

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
        # Base EMA (34 period Fibonacci number)
        dataframe['ema_base'] = ta.EMA(dataframe, timeperiod=34)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Fibonacci bands
        dataframe['fib_lower_1618'] = dataframe['ema_base'] - (1.618 * dataframe['atr'])
        dataframe['fib_upper_1618'] = dataframe['ema_base'] + (1.618 * dataframe['atr'])
        dataframe['fib_upper_2618'] = dataframe['ema_base'] + (2.618 * dataframe['atr'])

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Price dipped to or below the 1.618 lower Fibonacci band
                (dataframe['low'] <= dataframe['fib_lower_1618']) &
                # Reversal confirmation: Close is green and closes above the 1.618 band
                (dataframe['close'] > dataframe['fib_lower_1618']) &
                (dataframe['close'] > dataframe['open']) &
                # RSI not overbought
                (dataframe['rsi'] < 48) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Exit when price reaches or exceeds the upper Fibonacci band or RSI > 72
                (dataframe['close'] >= dataframe['fib_upper_1618']) |
                (dataframe['rsi'] > 72)
            ) & (dataframe['volume'] > 0),
            'exit_long'
        ] = 1

        return dataframe

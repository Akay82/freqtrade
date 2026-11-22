# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import IStrategy
from risk_management import RiskManagementMixin
import talib.abstract as ta
from technical import qtpylib


class BTCSwing_IchimokuCloud(RiskManagementMixin, IStrategy):
    """
    Strategy 6: Ichimoku Kinko Hyo Cloud Breakout Swing
    Thesis: Identify strong equilibrium breakouts when price surges above the Kumo Cloud
            with Tenkan/Kijun bullish alignment.
    """
    INTERFACE_VERSION = 3

    timeframe = '1h'

    minimal_roi = {
        "0": 0.09,
        "1440": 0.05,
        "2880": 0.03,
        "5760": 0.015
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
    startup_candle_count: int = 100

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        nine_period_high = dataframe['high'].rolling(window=9).max()
        nine_period_low = dataframe['low'].rolling(window=9).min()
        dataframe['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        twenty_six_high = dataframe['high'].rolling(window=26).max()
        twenty_six_low = dataframe['low'].rolling(window=26).min()
        dataframe['kijun_sen'] = (twenty_six_high + twenty_six_low) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        dataframe['senkou_span_a'] = ((dataframe['tenkan_sen'] + dataframe['kijun_sen']) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        fifty_two_high = dataframe['high'].rolling(window=52).max()
        fifty_two_low = dataframe['low'].rolling(window=52).min()
        dataframe['senkou_span_b'] = ((fifty_two_high + fifty_two_low) / 2).shift(26)

        # Cloud top and bottom
        dataframe['cloud_top'] = dataframe[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        dataframe['cloud_bottom'] = dataframe[['senkou_span_a', 'senkou_span_b']].min(axis=1)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Price is strictly above the Kumo Cloud
                (dataframe['close'] > dataframe['cloud_top']) &
                # Tenkan-sen is above Kijun-sen (Bullish alignment)
                (dataframe['tenkan_sen'] > dataframe['kijun_sen']) &
                # Cross above or bounce off Kijun-sen
                (dataframe['close'] > dataframe['kijun_sen']) &
                (dataframe['rsi'] > 45) &
                (dataframe['rsi'] < 68) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Exit when price drops back below Kijun-sen or below Cloud Top
                (dataframe['close'] < dataframe['kijun_sen']) |
                (dataframe['close'] < dataframe['cloud_top'])
            ) & (dataframe['volume'] > 0),
            'exit_long'
        ] = 1

        return dataframe

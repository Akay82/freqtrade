# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import IStrategy
from risk_management import RiskManagementMixin
import talib.abstract as ta
from technical import qtpylib


class BTCSwing_SqueezeBreakout(RiskManagementMixin, IStrategy):
    """
    Strategy 4: Bollinger & Keltner Squeeze Breakout (TTM Model)
    Thesis: Periods of low volatility (Bollinger inside Keltner Channel) lead to violent explosive breakouts.
    Entry: The squeeze fires (BB expands outside KC) with positive momentum.
    """
    INTERFACE_VERSION = 3

    timeframe = '1h'

    minimal_roi = {
        "0": 0.08,
        "720": 0.045,
        "1440": 0.025,
        "2880": 0.015
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
        # Bollinger Bands (20, 2.0)
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2.0)
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_mid'] = bollinger['mid']

        # Keltner Channel (20, 1.5 ATR)
        atr = ta.ATR(dataframe, timeperiod=20)
        dataframe['kc_mid'] = ta.EMA(dataframe['close'], timeperiod=20)
        dataframe['kc_upper'] = dataframe['kc_mid'] + (1.5 * atr)
        dataframe['kc_lower'] = dataframe['kc_mid'] - (1.5 * atr)

        # Squeeze Condition: True when BB is inside KC
        dataframe['squeeze_on'] = (dataframe['bb_lower'] > dataframe['kc_lower']) & (dataframe['bb_upper'] < dataframe['kc_upper'])

        # Momentum: Linear regression / rate of change over midpoint
        dataframe['mom'] = ta.MOM(dataframe['close'], timeperiod=12)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Squeeze was ON previously, and now FIRES (squeeze_on turns False)
                (dataframe['squeeze_on'].shift(1) == True) &
                (dataframe['squeeze_on'] == False) &
                # Momentum is positive and expanding
                (dataframe['mom'] > 0) &
                (dataframe['close'] > dataframe['bb_mid']) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Exit when momentum weakens or turns negative
                (dataframe['mom'] < 0) |
                (dataframe['rsi'] > 75)
            ) & (dataframe['volume'] > 0),
            'exit_long'
        ] = 1

        return dataframe

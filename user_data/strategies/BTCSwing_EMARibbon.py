# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import IStrategy
from risk_management import RiskManagementMixin
import talib.abstract as ta
from technical import qtpylib


class BTCSwing_EMARibbon(RiskManagementMixin, IStrategy):
    """
    Strategy 8: Guppy / EMA Ribbon Trend Surfer
    Thesis: Identify powerful sustained swings when the short-term ribbon (EMAs 8, 13, 21)
            fans out cleanly above the long-term ribbon (EMAs 55, 89, 144).
    """
    INTERFACE_VERSION = 3

    timeframe = '1h'

    minimal_roi = {
        "0": 0.10,
        "1440": 0.05,
        "2880": 0.03,
        "5760": 0.015
    }

    stoploss = -0.05

    trailing_stop = True
    trailing_stop_positive = 0.018
    trailing_stop_positive_offset = 0.028
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 150

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Fast Ribbon (Trader sentiment)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)

        # Slow Ribbon (Investor baseline)
        dataframe['ema_55'] = ta.EMA(dataframe, timeperiod=55)
        dataframe['ema_89'] = ta.EMA(dataframe, timeperiod=89)
        dataframe['ema_144'] = ta.EMA(dataframe, timeperiod=144)

        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Fast ribbon cleanly ordered (8 > 13 > 21)
                (dataframe['ema_8'] > dataframe['ema_13']) &
                (dataframe['ema_13'] > dataframe['ema_21']) &
                # Fast ribbon is above slow ribbon (21 > 55)
                (dataframe['ema_21'] > dataframe['ema_55']) &
                # Slow ribbon ordered (55 > 89)
                (dataframe['ema_55'] > dataframe['ema_89']) &
                # Ribbon expansion trigger (ema_8 crossed above or expanding)
                (dataframe['ema_8'] > dataframe['ema_8'].shift(1)) &
                (dataframe['adx'] > 20) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Exit when fast EMA crosses below EMA 21 (momentum collapse)
                (dataframe['ema_8'] < dataframe['ema_21']) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'
        ] = 1

        return dataframe

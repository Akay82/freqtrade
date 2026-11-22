# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import IStrategy
from risk_management import RiskManagementMixin
import talib.abstract as ta


class BTCSwing_DonchianTurtle(RiskManagementMixin, IStrategy):
    """
    Strategy 5: Donchian Channel / Turtle Breakout Swing
    Thesis: Classic Richard Dennis trend following adapted for crypto.
            Buys when price breaks out of the 20-period highest high.
            Exits when price breaks below the 10-period lowest low or hits trailing stop.
    """
    INTERFACE_VERSION = 3

    timeframe = '1h'

    minimal_roi = {
        "0": 0.12,
        "1440": 0.06,
        "2880": 0.035,
        "5760": 0.02
    }

    stoploss = -0.055

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 50

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Donchian Channels (20 High, 10 Low)
        dataframe['donchian_high_20'] = dataframe['high'].rolling(window=20).max().shift(1)
        dataframe['donchian_low_10'] = dataframe['low'].rolling(window=10).min().shift(1)

        # Long term trend anchor
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Close breaks above the previous 20-candle highest high
                (dataframe['close'] > dataframe['donchian_high_20']) &
                # Above the 100 EMA trend anchor
                (dataframe['close'] > dataframe['ema_100']) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Turtle Exit: price drops below the 10-period lowest low
                (dataframe['close'] < dataframe['donchian_low_10']) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'
        ] = 1

        return dataframe

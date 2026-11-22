# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import IStrategy
from risk_management import RiskManagementMixin
import talib.abstract as ta
from technical import qtpylib


class BTCSwing_RSI_VWAP(RiskManagementMixin, IStrategy):
    """
    Strategy 7: Volume-Weighted Average Price (VWAP) & RSI Mean Reversion
    Thesis: Identify institutional discount zones when price dips below rolling VWAP bands
            with oversold RSI exhaustion and reclaims support.
    """
    INTERFACE_VERSION = 3

    timeframe = '1h'

    minimal_roi = {
        "0": 0.06,
        "720": 0.035,
        "1440": 0.02,
        "2880": 0.012
    }

    stoploss = -0.04

    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.022
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 50

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Rolling VWAP over 24 1-hour candles (daily rolling institutional benchmark)
        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        vol_tp = typical_price * dataframe['volume']
        rolling_vol = dataframe['volume'].rolling(window=24).sum()
        dataframe['vwap_24'] = vol_tp.rolling(window=24).sum() / rolling_vol.replace(0, np.nan)

        # Standard deviation bands around VWAP
        rolling_std = typical_price.rolling(window=24).std()
        dataframe['vwap_lower'] = dataframe['vwap_24'] - (1.5 * rolling_std)
        dataframe['vwap_upper'] = dataframe['vwap_24'] + (1.5 * rolling_std)

        # Oscillators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Reclaiming from discount: Close was below or touched lower VWAP band and now crosses back up
                (dataframe['close'] > dataframe['vwap_lower']) &
                (dataframe['low'].shift(1) <= dataframe['vwap_lower'].shift(1)) &
                # RSI oversold recovery
                (dataframe['rsi'] < 42) &
                (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Exit when price reaches upper VWAP band or overbought RSI
                (dataframe['close'] >= dataframe['vwap_upper']) |
                (dataframe['rsi'] > 72)
            ) & (dataframe['volume'] > 0),
            'exit_long'
        ] = 1

        return dataframe

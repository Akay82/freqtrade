# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import IStrategy
from risk_management import RiskManagementMixin
import talib.abstract as ta
from technical import qtpylib


def calculate_cmf(dataframe: DataFrame, period=20) -> pd.Series:
    hl_diff = (dataframe['high'] - dataframe['low']).replace(0, np.nan)
    mf_multiplier = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / hl_diff
    mf_volume = mf_multiplier * dataframe['volume']
    cmf = mf_volume.rolling(window=period).sum() / dataframe['volume'].rolling(window=period).sum().replace(0, np.nan)
    return cmf.fillna(0)


class BTCSwing_StochRSI_CMF(RiskManagementMixin, IStrategy):
    """
    Strategy 9: Chaikin Money Flow + Stochastic RSI Swing
    Thesis: Enter swings when institutional volume accumulation (CMF > 0.05)
            aligns with oversold Stochastic RSI momentum turns.
    """
    INTERFACE_VERSION = 3

    timeframe = '1h'

    minimal_roi = {
        "0": 0.07,
        "1080": 0.04,
        "2160": 0.025,
        "4320": 0.012
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
        # Chaikin Money Flow
        dataframe['cmf'] = calculate_cmf(dataframe, period=20)

        # Stochastic RSI
        stoch_fast = ta.STOCHF(dataframe, fastk_period=14, fastd_period=3)
        dataframe['stoch_k'] = stoch_fast['fastk']
        dataframe['stoch_d'] = stoch_fast['fastd']

        # Trend filter
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Institutional money inflow positive
                (dataframe['cmf'] > 0.03) &
                # StochRSI oversold crossover
                (qtpylib.crossed_above(dataframe['stoch_k'], dataframe['stoch_d'])) &
                (dataframe['stoch_k'] < 30) &
                # Trend check: above or near 50 EMA
                (dataframe['close'] > dataframe['ema_50'] * 0.985) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Exit when StochRSI is overbought and crosses below D
                (qtpylib.crossed_below(dataframe['stoch_k'], dataframe['stoch_d'])) &
                (dataframe['stoch_k'] > 75) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'
        ] = 1

        return dataframe

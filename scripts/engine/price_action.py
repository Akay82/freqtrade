"""
Price Action and Market Structure Feature Engine for BTC Swing Trading.
Detects Swing Highs/Lows, HH/HL/LH/LL market structure, dynamic Support/Resistance,
Breakout & Pullback confirmations, and Candlestick anatomy.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class PriceActionDetector:
    """
    Computes price action features strictly on historical closed candles,
    avoiding lookahead bias.
    """

    def __init__(self, pivot_window: int = 5):
        """
        :param pivot_window: number of candles on each side to define a local swing high/low.
                             To avoid lookahead bias, pivot at candle [i - pivot_window] is only
                             confirmed and usable at candle [i]!
        """
        self.pivot_window = pivot_window

    def compute_all_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes all market structure and price action features on dataframe."""
        if df is None or len(df) == 0:
            return df

        res = df.copy()

        # 1. Candlestick anatomy & patterns
        res = self.add_candlestick_structure(res)

        # 2. Swing Highs & Lows (Pivots with lag to prevent lookahead)
        res = self.add_swing_pivots(res)

        # 3. Market Structure (HH, HL, LH, LL)
        res = self.add_market_structure(res)

        # 4. Support and Resistance zones
        res = self.add_support_resistance(res)

        # 5. Breakout & Pullback confirmation
        res = self.add_breakout_and_pullback(res)

        return res

    def add_candlestick_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes candlestick anatomy features:
        - Body size, upper shadow, lower shadow, total range
        - Body-to-range ratio
        - Bullish Hammer / Pin Bar
        - Shooting Star / Bearish Pin Bar
        - Bullish & Bearish Engulfing
        """
        open_p = df['open']
        high = df['high']
        low = df['low']
        close = df['close']

        candle_range = (high - low).replace(0, np.nan)
        body = (close - open_p).abs()
        upper_shadow = high - np.maximum(open_p, close)
        lower_shadow = np.minimum(open_p, close) - low

        df['candle_body'] = body
        df['candle_range'] = candle_range
        df['body_ratio'] = body / candle_range
        df['upper_shadow_ratio'] = upper_shadow / candle_range
        df['lower_shadow_ratio'] = lower_shadow / candle_range
        df['is_bullish_candle'] = close > open_p
        df['is_bearish_candle'] = close < open_p

        # Bullish Hammer / Pin Bar:
        # Long lower shadow (>= 60% of candle range), small upper shadow (<= 20%), small body
        df['is_bullish_hammer'] = (
            (df['lower_shadow_ratio'] >= 0.55) &
            (df['upper_shadow_ratio'] <= 0.20) &
            (df['body_ratio'] <= 0.35)
        )

        # Shooting Star / Bearish Pin Bar:
        # Long upper shadow (>= 55% of range), small lower shadow (<= 20%), small body
        df['is_shooting_star'] = (
            (df['upper_shadow_ratio'] >= 0.55) &
            (df['lower_shadow_ratio'] <= 0.20) &
            (df['body_ratio'] <= 0.35)
        )

        # Bullish Engulfing:
        # Previous candle was bearish, current candle is bullish and completely engulfs previous body
        prev_open = open_p.shift(1)
        prev_close = close.shift(1)
        df['is_bullish_engulfing'] = (
            (prev_close < prev_open) &
            (close > open_p) &
            (open_p <= prev_close) &
            (close >= prev_open)
        )

        # Bearish Engulfing:
        # Previous candle was bullish, current candle is bearish and completely engulfs previous body
        df['is_bearish_engulfing'] = (
            (prev_close > prev_open) &
            (close < open_p) &
            (open_p >= prev_close) &
            (close <= prev_open)
        )

        return df

    def add_swing_pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies swing highs and swing lows.
        CRITICAL NO-LOOKAHEAD BIAS:
        A swing high at index [i - pivot_window] is only confirmed when candle [i] closes!
        Therefore, we shift the confirmed pivot forward by pivot_window candles so that at
        any candle time t, only pivots confirmed at or before t are known.
        """
        w = self.pivot_window
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)

        is_swing_high = np.zeros(n, dtype=bool)
        is_swing_low = np.zeros(n, dtype=bool)
        swing_high_price = np.full(n, np.nan)
        swing_low_price = np.full(n, np.nan)

        # Find raw local extrema
        for i in range(w, n - w):
            # Check high
            if highs[i] == np.max(highs[i - w: i + w + 1]):
                # Confirmed at candle (i + w)
                confirm_idx = i + w
                if confirm_idx < n:
                    is_swing_high[confirm_idx] = True
                    swing_high_price[confirm_idx] = highs[i]

            # Check low
            if lows[i] == np.min(lows[i - w: i + w + 1]):
                confirm_idx = i + w
                if confirm_idx < n:
                    is_swing_low[confirm_idx] = True
                    swing_low_price[confirm_idx] = lows[i]

        df['is_swing_high'] = is_swing_high
        df['is_swing_low'] = is_swing_low

        # Forward fill the most recently confirmed swing high and swing low levels
        df['last_swing_high'] = pd.Series(swing_high_price, index=df.index).ffill()
        df['last_swing_low'] = pd.Series(swing_low_price, index=df.index).ffill()

        return df

    def add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects structural market trend state:
        - Higher High (HH): New confirmed swing high > previous confirmed swing high
        - Higher Low (HL): New confirmed swing low > previous confirmed swing low
        - Lower High (LH): New confirmed swing high < previous confirmed swing high
        - Lower Low (LL): New confirmed swing low < previous confirmed swing low
        - Structural trend: 1 = Bullish (HH + HL), -1 = Bearish (LH + LL), 0 = Sideways/Mixed
        """
        n = len(df)
        structure_trend = np.zeros(n, dtype=int)
        is_hh = np.zeros(n, dtype=bool)
        is_hl = np.zeros(n, dtype=bool)
        is_lh = np.zeros(n, dtype=bool)
        is_ll = np.zeros(n, dtype=bool)

        high_pivots = []  # list of (index, price)
        low_pivots = []   # list of (index, price)

        last_high_relation = 0  # 1 for HH, -1 for LH
        last_low_relation = 0   # 1 for HL, -1 for LL

        for i in range(n):
            if df['is_swing_high'].iloc[i]:
                current_high = df['last_swing_high'].iloc[i]
                if high_pivots:
                    prev_high = high_pivots[-1][1]
                    if current_high > prev_high:
                        is_hh[i] = True
                        last_high_relation = 1
                    elif current_high < prev_high:
                        is_lh[i] = True
                        last_high_relation = -1
                high_pivots.append((i, current_high))

            if df['is_swing_low'].iloc[i]:
                current_low = df['last_swing_low'].iloc[i]
                if low_pivots:
                    prev_low = low_pivots[-1][1]
                    if current_low > prev_low:
                        is_hl[i] = True
                        last_low_relation = 1
                    elif current_low < prev_low:
                        is_ll[i] = True
                        last_low_relation = -1
                low_pivots.append((i, current_low))

            # Trend classification based on latest structure
            if last_high_relation == 1 and last_low_relation == 1:
                structure_trend[i] = 1   # Bullish Structure (HH + HL)
            elif last_high_relation == -1 and last_low_relation == -1:
                structure_trend[i] = -1  # Bearish Structure (LH + LL)
            else:
                structure_trend[i] = 0   # Mixed / Range Structure

        df['is_higher_high'] = is_hh
        df['is_higher_low'] = is_hl
        df['is_lower_high'] = is_lh
        df['is_lower_low'] = is_ll
        df['market_structure_trend'] = structure_trend

        return df

    def add_support_resistance(self, df: pd.DataFrame, window: int = 40) -> pd.DataFrame:
        """
        Calculates dynamic support and resistance levels based on rolling swing points.
        Support = highest swing low within the recent window (below current close)
        Resistance = lowest swing high within the recent window (above current close)
        """
        # Dynamic rolling high of recent 20 candles
        df['resistance_level'] = df['high'].rolling(window=window).max().shift(1)
        df['support_level'] = df['low'].rolling(window=window).min().shift(1)

        # Distance to resistance and support in percentage
        df['dist_to_resistance_pct'] = (df['resistance_level'] - df['close']) / df['close'] * 100.0
        df['dist_to_support_pct'] = (df['close'] - df['support_level']) / df['close'] * 100.0

        return df

    def add_breakout_and_pullback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies Breakout and Pullback confirmation events:
        - Breakout: Close crosses above resistance_level with above-average volume
        - Pullback: In an uptrend (EMA 50 > EMA 200), low retests EMA 20 or EMA 50, but close holds above it.
        """
        close = df['close']
        open_p = df['open']
        high = df['high']
        low = df['low']
        resistance = df['resistance_level']
        support = df['support_level']

        # Volume condition (relative volume >= 1.2 if available, else True)
        vol_confirm = df['relative_volume'] >= 1.2 if 'relative_volume' in df.columns else True

        # Breakout: Previous close below resistance, current close cleanly breaks above
        df['is_resistance_breakout'] = (
            (close.shift(1) <= resistance.shift(1)) &
            (close > resistance) &
            (close > open_p) &
            vol_confirm
        )

        # Breakdown: Close breaks below support
        df['is_support_breakdown'] = (
            (close.shift(1) >= support.shift(1)) &
            (close < support) &
            (close < open_p) &
            vol_confirm
        )

        # Pullback confirmation:
        # Requires EMA 20 and EMA 50 to be present
        if 'ema_20' in df.columns and 'ema_50' in df.columns:
            uptrend = df['ema_20'] > df['ema_50']
            # Price pulled back near or touched EMA 20, but closed above it with bullish candle
            ema_20_touch = (low <= df['ema_20'] * 1.003) & (close >= df['ema_20'])
            ema_50_touch = (low <= df['ema_50'] * 1.005) & (close >= df['ema_50'])
            
            df['is_ema20_pullback_bounce'] = uptrend & ema_20_touch & (close > open_p)
            df['is_ema50_pullback_bounce'] = uptrend & ema_50_touch & (close > open_p)
        else:
            df['is_ema20_pullback_bounce'] = False
            df['is_ema50_pullback_bounce'] = False

        return df

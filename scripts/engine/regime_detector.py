"""
Market Regime Detection Engine for BTC Swing Trading.
Classifies market conditions into 6 distinct regimes:
1. Bullish Trend
2. Bearish Trend
3. Sideways / Range-bound
4. High Volatility
5. Low Volatility
6. Uncertain / No Clear Regime
Provides a confidence score (0-100), supporting factor breakdown, and strict no-lookahead processing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class MarketRegime(str, Enum):
    BULLISH_TREND = "BULLISH_TREND"
    BEARISH_TREND = "BEARISH_TREND"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class RegimeConfig:
    """Configurable thresholds for regime classification."""
    adx_trend_threshold: float = 22.0          # ADX above this implies trending market
    adx_range_threshold: float = 18.0          # ADX below this implies range-bound
    ema_slope_bull_min: float = 0.02           # Minimum normalized slope for bullish confirmation
    ema_slope_bear_max: float = -0.02          # Maximum normalized slope for bearish confirmation
    high_vol_atr_multiplier: float = 1.55      # Multiplier over 50-period mean ATR% to trigger High Volatility
    low_vol_bb_width_percentile: float = 20.0  # BB Width percentile below which is Low Volatility compression


@dataclass
class RegimeClassification:
    """Detailed output of market regime detection."""
    regime: MarketRegime
    confidence_score: float                    # 0 to 100
    supporting_indicators: Dict[str, Any]      # Values that supported this decision
    conflicting_indicators: List[str]          # Any conflicting evidence
    timestamp: str                             # ISO timestamp of candle
    data_quality_status: str                   # 'HEALTHY' | 'WARNING' | 'INSUFFICIENT_DATA'
    summary_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence_score": round(self.confidence_score, 1),
            "supporting_indicators": self.supporting_indicators,
            "conflicting_indicators": self.conflicting_indicators,
            "timestamp": self.timestamp,
            "data_quality_status": self.data_quality_status,
            "summary_reason": self.summary_reason
        }


class RegimeDetector:
    """
    Evaluates price structure, moving average relationships, momentum,
    and volatility to classify the current BTC market regime.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()

    def classify_candle(self, df: pd.DataFrame, candle_idx: int = -1) -> RegimeClassification:
        """
        Classifies the market regime for a specific candle index (defaults to the latest).
        Expects a DataFrame enriched with indicators from IndicatorEngine and PriceActionDetector.
        """
        if df is None or len(df) == 0:
            return RegimeClassification(
                regime=MarketRegime.UNCERTAIN,
                confidence_score=0.0,
                supporting_indicators={},
                conflicting_indicators=["Empty dataframe"],
                timestamp=datetime.now(timezone.utc).isoformat(),
                data_quality_status="INSUFFICIENT_DATA",
                summary_reason="Cannot classify empty data"
            )

        if candle_idx < 0:
            candle_idx = len(df) + candle_idx

        if candle_idx < 30:
            return RegimeClassification(
                regime=MarketRegime.UNCERTAIN,
                confidence_score=0.0,
                supporting_indicators={},
                conflicting_indicators=["Insufficient candle history (<30 candles)"],
                timestamp=str(df['date'].iloc[candle_idx]) if 'date' in df.columns else datetime.now(timezone.utc).isoformat(),
                data_quality_status="INSUFFICIENT_DATA",
                summary_reason="Insufficient warm-up candles for regime detection"
            )

        row = df.iloc[candle_idx]
        ts_str = str(row['date']) if 'date' in df.columns else datetime.now(timezone.utc).isoformat()

        # Extract primary metrics with safe fallbacks
        close = float(row.get('close', 0.0))
        ema_50 = float(row.get('ema_50', np.nan))
        ema_200 = float(row.get('ema_200', np.nan))
        ema_20 = float(row.get('ema_20', np.nan))
        adx = float(row.get('adx_14', np.nan))
        plus_di = float(row.get('plus_di', np.nan))
        minus_di = float(row.get('minus_di', np.nan))
        rsi = float(row.get('rsi_14', 50.0))
        atr_pct = float(row.get('atr_pct', np.nan))
        bb_width = float(row.get('bb_width', np.nan))
        structure = int(row.get('market_structure_trend', 0))
        supertrend_dir = int(row.get('supertrend_direction', 0))
        ema_50_slope = float(row.get('ema_50_slope', 0.0))

        # Check for historical baseline volatility (rolling 50-period mean ATR%)
        hist_slice = df['atr_pct'].iloc[max(0, candle_idx - 50): candle_idx] if 'atr_pct' in df.columns else pd.Series()
        mean_atr_pct = float(hist_slice.mean()) if len(hist_slice) > 10 else 1.2

        # Check historical BB width percentile (rolling 50-period)
        bb_slice = df['bb_width'].iloc[max(0, candle_idx - 50): candle_idx] if 'bb_width' in df.columns else pd.Series()
        bb_percentile = float(pd.Series(bb_slice).rank(pct=True).iloc[-1] * 100.0) if len(bb_slice) > 10 else 50.0

        support_ind = {
            "close": round(close, 2),
            "ema_50": round(ema_50, 2) if not np.isnan(ema_50) else None,
            "ema_200": round(ema_200, 2) if not np.isnan(ema_200) else None,
            "adx_14": round(adx, 1) if not np.isnan(adx) else None,
            "rsi_14": round(rsi, 1),
            "atr_pct": round(atr_pct, 2) if not np.isnan(atr_pct) else None,
            "mean_atr_pct": round(mean_atr_pct, 2),
            "bb_width": round(bb_width, 4) if not np.isnan(bb_width) else None,
            "bb_percentile": round(bb_percentile, 1),
            "market_structure_trend": structure,
            "supertrend_direction": supertrend_dir,
            "ema_50_slope": round(ema_50_slope, 3)
        }

        conflicts = []

        # =========================================================================
        # 1. Check for Extreme Volatility (High Volatility Regime)
        # =========================================================================
        # If ATR% is dramatically higher than historical baseline (> high_vol_atr_multiplier)
        if not np.isnan(atr_pct) and mean_atr_pct > 0:
            if atr_pct >= mean_atr_pct * self.config.high_vol_atr_multiplier and atr_pct > 1.8:
                vol_confidence = min(95.0, 60.0 + (atr_pct / mean_atr_pct - 1.0) * 40.0)
                return RegimeClassification(
                    regime=MarketRegime.HIGH_VOLATILITY,
                    confidence_score=vol_confidence,
                    supporting_indicators=support_ind,
                    conflicting_indicators=conflicts,
                    timestamp=ts_str,
                    data_quality_status="HEALTHY",
                    summary_reason=f"Volatility spike: ATR% ({atr_pct:.2f}%) is {atr_pct/mean_atr_pct:.2f}x above baseline ({mean_atr_pct:.2f}%)"
                )

        # =========================================================================
        # 2. Check for Extreme Volatility Compression (Low Volatility Regime)
        # =========================================================================
        # If BB Width is in the lowest 20th percentile and ADX is low
        ttm_squeeze = bool(row.get('ttm_squeeze_on', False))
        if (bb_percentile <= self.config.low_vol_bb_width_percentile or ttm_squeeze) and (np.isnan(adx) or adx < 20.0):
            low_vol_confidence = 75.0 if ttm_squeeze else 65.0
            return RegimeClassification(
                regime=MarketRegime.LOW_VOLATILITY,
                confidence_score=low_vol_confidence,
                supporting_indicators=support_ind,
                conflicting_indicators=conflicts,
                timestamp=ts_str,
                data_quality_status="HEALTHY",
                summary_reason=f"Volatility squeeze: BB Width in {bb_percentile:.1f} percentile with low ADX ({adx:.1f})"
            )

        # =========================================================================
        # 3. Check for Strong Trending Markets (Bullish vs Bearish)
        # =========================================================================
        has_ema_200 = not np.isnan(ema_200)
        ema_trend_bullish = (ema_50 > ema_200) if has_ema_200 else (close > ema_50)
        ema_trend_bearish = (ema_50 < ema_200) if has_ema_200 else (close < ema_50)

        slope_bullish = ema_50_slope > self.config.ema_slope_bull_min
        slope_bearish = ema_50_slope < self.config.ema_slope_bear_max

        adx_trending = not np.isnan(adx) and adx >= self.config.adx_trend_threshold
        di_bullish = not np.isnan(plus_di) and plus_di > minus_di
        di_bearish = not np.isnan(minus_di) and minus_di > plus_di

        # Evaluate Bullish Trend
        bullish_score = 0.0
        bullish_conflicts = []
        if ema_trend_bullish:
            bullish_score += 25.0
        else:
            bullish_conflicts.append("EMA 50 below EMA 200")

        if slope_bullish:
            bullish_score += 20.0
        elif slope_bearish:
            bullish_conflicts.append("EMA slope is negative")

        if structure == 1:
            bullish_score += 20.0
        elif structure == -1:
            bullish_conflicts.append("Market structure is bearish (LH/LL)")

        if adx_trending:
            bullish_score += 15.0
            if di_bullish:
                bullish_score += 10.0
        elif not np.isnan(adx) and adx < self.config.adx_range_threshold:
            bullish_conflicts.append(f"ADX ({adx:.1f}) is low/flat")

        if supertrend_dir == 1:
            bullish_score += 10.0

        # Evaluate Bearish Trend
        bearish_score = 0.0
        bearish_conflicts = []
        if ema_trend_bearish:
            bearish_score += 25.0
        else:
            bearish_conflicts.append("EMA 50 above EMA 200 (Macro Golden Cross)")

        if slope_bearish:
            bearish_score += 20.0
        elif slope_bullish:
            bearish_conflicts.append("EMA slope is positive")

        if structure == -1:
            bearish_score += 20.0
        elif structure == 1:
            bearish_conflicts.append("Market structure is bullish (HH/HL)")

        if adx_trending:
            bearish_score += 15.0
            if di_bearish:
                bearish_score += 10.0
        elif not np.isnan(adx) and adx < self.config.adx_range_threshold:
            bearish_conflicts.append(f"ADX ({adx:.1f}) is low/flat")

        if supertrend_dir == -1:
            bearish_score += 10.0

        # If Bullish criteria strongly dominate
        if bullish_score >= 65.0 and ema_trend_bullish and len(bullish_conflicts) <= 1:
            return RegimeClassification(
                regime=MarketRegime.BULLISH_TREND,
                confidence_score=min(95.0, bullish_score),
                supporting_indicators=support_ind,
                conflicting_indicators=bullish_conflicts,
                timestamp=ts_str,
                data_quality_status="HEALTHY",
                summary_reason=f"Bullish trend confirmed (Score: {bullish_score:.0f}/100, ADX: {adx:.1f}, Structure: {structure})"
            )

        # If Bearish criteria strongly dominate
        if bearish_score >= 65.0 and ema_trend_bearish and len(bearish_conflicts) <= 1:
            return RegimeClassification(
                regime=MarketRegime.BEARISH_TREND,
                confidence_score=min(95.0, bearish_score),
                supporting_indicators=support_ind,
                conflicting_indicators=bearish_conflicts,
                timestamp=ts_str,
                data_quality_status="HEALTHY",
                summary_reason=f"Bearish trend confirmed (Score: {bearish_score:.0f}/100, ADX: {adx:.1f}, Structure: {structure})"
            )

        # =========================================================================
        # 4. Check for Sideways / Range-Bound Market
        # =========================================================================
        # Low ADX (< range_threshold), flat EMA slopes, oscillating RSI around 40-60
        adx_flat = not np.isnan(adx) and adx < self.config.adx_range_threshold
        slope_flat = abs(ema_50_slope) < self.config.ema_slope_bull_min
        rsi_middle = 40.0 <= rsi <= 60.0

        if adx_flat or (slope_flat and rsi_middle and structure == 0):
            range_confidence = 60.0
            if adx_flat:
                range_confidence += 15.0
            if slope_flat:
                range_confidence += 15.0
            return RegimeClassification(
                regime=MarketRegime.SIDEWAYS,
                confidence_score=min(90.0, range_confidence),
                supporting_indicators=support_ind,
                conflicting_indicators=conflicts,
                timestamp=ts_str,
                data_quality_status="HEALTHY",
                summary_reason=f"Sideways range: ADX ({adx:.1f}) is below trend threshold, flat EMA slope ({ema_50_slope:.3f})"
            )

        # =========================================================================
        # 5. Default: Uncertain / Mixed Indicators
        # =========================================================================
        all_conflicts = list(set(bullish_conflicts + bearish_conflicts))
        return RegimeClassification(
            regime=MarketRegime.UNCERTAIN,
            confidence_score=45.0,
            supporting_indicators=support_ind,
            conflicting_indicators=all_conflicts or ["Conflicting trend and oscillator signals"],
            timestamp=ts_str,
            data_quality_status="HEALTHY",
            summary_reason="Uncertain regime: Indicators conflict or lack strong directional momentum"
        )

    def classify_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies every candle in a historical DataFrame.
        Adds columns: 'regime' and 'regime_confidence'.
        """
        res = df.copy()
        regimes = []
        confidences = []

        for i in range(len(res)):
            classification = self.classify_candle(res, candle_idx=i)
            regimes.append(classification.regime.value)
            confidences.append(classification.confidence_score)

        res['market_regime'] = regimes
        res['regime_confidence'] = confidences
        return res

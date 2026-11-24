"""
Multi-Timeframe Analysis and Confirmation Engine for BTC Swing Trading.
Analyzes 1D (Macro Trend), 4H (Setup Structure), and 1H (Entry Refinement).
Resolves timeframe conflicts and outputs definitive decisions: BUY, HOLD, EXIT, or NO_TRADE.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from scripts.engine.regime_detector import MarketRegime, RegimeDetector


class MTFDecision(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    EXIT = "EXIT"
    NO_TRADE = "NO_TRADE"


@dataclass
class TimeframeAnalysis:
    """Evaluation of an individual timeframe."""
    timeframe: str
    bias: str                     # 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    score: float                  # 0 to 100
    key_indicators: Dict[str, Any]
    reasons: List[str] = field(default_factory=list)


@dataclass
class MTFAnalysisResult:
    """Consolidated multi-timeframe analysis and signal decision."""
    decision: MTFDecision
    alignment_score: float        # 0 to 100 weighted multi-timeframe score
    daily: TimeframeAnalysis
    h4: TimeframeAnalysis
    h1: TimeframeAnalysis
    conflicts: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "alignment_score": round(self.alignment_score, 1),
            "daily_bias": self.daily.bias,
            "daily_score": round(self.daily.score, 1),
            "h4_bias": self.h4.bias,
            "h4_score": round(self.h4.score, 1),
            "h1_bias": self.h1.bias,
            "h1_score": round(self.h1.score, 1),
            "conflicts": self.conflicts,
            "reasons": self.reasons,
            "timestamp": self.timestamp,
            "details": {
                "daily": {
                    "bias": self.daily.bias,
                    "score": self.daily.score,
                    "indicators": self.daily.key_indicators,
                    "reasons": self.daily.reasons
                },
                "h4": {
                    "bias": self.h4.bias,
                    "score": self.h4.score,
                    "indicators": self.h4.key_indicators,
                    "reasons": self.h4.reasons
                },
                "h1": {
                    "bias": self.h1.bias,
                    "score": self.h1.score,
                    "indicators": self.h1.key_indicators,
                    "reasons": self.h1.reasons
                }
            }
        }


class MultiTimeframeEngine:
    """
    Coordinates multi-timeframe analysis and enforces strict alignment rules:
    - 1D: Overall market direction and macro filter
    - 4H: Structural setup and regime validation
    - 1H: Precision entry timing and momentum confirmation
    """

    def __init__(self, min_alignment_score: float = 65.0):
        self.min_alignment_score = min_alignment_score
        self.regime_detector = RegimeDetector()

    def evaluate_daily(self, df_1d: pd.DataFrame, idx: int = -1) -> TimeframeAnalysis:
        """
        Analyzes 1D candle for overall market direction.
        Criteria:
        - EMA 50 vs EMA 200 (Golden / Death Cross)
        - Price relative to EMA 50
        - RSI 14 level (> 45 for constructive structure)
        - EMA 50 slope
        """
        if df_1d is None or len(df_1d) < 30:
            return TimeframeAnalysis(
                timeframe="1d",
                bias="NEUTRAL",
                score=40.0,
                key_indicators={},
                reasons=["Insufficient 1D data history"]
            )

        if idx < 0:
            idx = len(df_1d) + idx

        row = df_1d.iloc[idx]
        close = float(row.get('close', 0.0))
        ema_50 = float(row.get('ema_50', np.nan))
        ema_200 = float(row.get('ema_200', np.nan))
        rsi = float(row.get('rsi_14', 50.0))
        structure = int(row.get('market_structure_trend', 0))
        slope = float(row.get('ema_50_slope', 0.0))

        score = 0.0
        reasons = []

        # 1. EMA 50 vs 200
        if not np.isnan(ema_200):
            if ema_50 > ema_200:
                score += 35.0
                reasons.append("Daily EMA 50 > EMA 200 (Macro Bullish)")
            else:
                reasons.append("Daily EMA 50 < EMA 200 (Macro Bearish)")
        else:
            if close > ema_50:
                score += 30.0

        # 2. Price vs EMA 50
        if not np.isnan(ema_50) and close > ema_50:
            score += 25.0
            reasons.append("Price trading above Daily EMA 50")

        # 3. Daily RSI
        if rsi >= 50.0:
            score += 20.0
            reasons.append(f"Daily RSI constructive ({rsi:.1f})")
        elif rsi < 40.0:
            reasons.append(f"Daily RSI depressed ({rsi:.1f})")

        # 4. Market structure
        if structure == 1:
            score += 20.0
            reasons.append("Daily market structure is Bullish (HH/HL)")
        elif structure == -1:
            reasons.append("Daily market structure is Bearish (LH/LL)")

        if score >= 60.0:
            bias = "BULLISH"
        elif score <= 35.0:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        return TimeframeAnalysis(
            timeframe="1d",
            bias=bias,
            score=score,
            key_indicators={
                "close": close,
                "ema_50": ema_50 if not np.isnan(ema_50) else None,
                "ema_200": ema_200 if not np.isnan(ema_200) else None,
                "rsi_14": rsi,
                "structure": structure,
                "ema_50_slope": slope
            },
            reasons=reasons
        )

    def evaluate_4h(self, df_4h: pd.DataFrame, idx: int = -1) -> TimeframeAnalysis:
        """
        Analyzes 4H candle for setup identification and regime confirmation.
        Criteria:
        - Regime: Bullish trend vs Sideways breakout vs Bearish
        - SuperTrend direction
        - ADX trend strength (> 20)
        - Price relative to EMA 20 and EMA 50
        """
        if df_4h is None or len(df_4h) < 30:
            return TimeframeAnalysis(
                timeframe="4h",
                bias="NEUTRAL",
                score=40.0,
                key_indicators={},
                reasons=["Insufficient 4H data history"]
            )

        if idx < 0:
            idx = len(df_4h) + idx

        row = df_4h.iloc[idx]
        close = float(row.get('close', 0.0))
        ema_20 = float(row.get('ema_20', np.nan))
        ema_50 = float(row.get('ema_50', np.nan))
        adx = float(row.get('adx_14', np.nan))
        supertrend_dir = int(row.get('supertrend_direction', 0))
        regime_cls = self.regime_detector.classify_candle(df_4h, candle_idx=idx)

        score = 0.0
        reasons = []

        # 1. Regime contribution
        if regime_cls.regime == MarketRegime.BULLISH_TREND:
            score += 40.0
            reasons.append("4H confirmed Bullish Trend regime")
        elif regime_cls.regime == MarketRegime.SIDEWAYS:
            score += 15.0
            reasons.append("4H in Sideways regime (requires breakout confirmation)")
        elif regime_cls.regime == MarketRegime.BEARISH_TREND:
            reasons.append("4H confirmed Bearish Trend regime")

        # 2. SuperTrend
        if supertrend_dir == 1:
            score += 25.0
            reasons.append("4H SuperTrend is Green (+1)")
        else:
            reasons.append("4H SuperTrend is Red (-1)")

        # 3. ADX Strength
        if not np.isnan(adx) and adx >= 22.0:
            score += 20.0
            reasons.append(f"4H ADX shows strong trend momentum ({adx:.1f})")

        # 4. Price > EMA 50
        if not np.isnan(ema_50) and close > ema_50:
            score += 15.0
            reasons.append("4H Close is above EMA 50")

        if score >= 60.0:
            bias = "BULLISH"
        elif score <= 35.0:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        return TimeframeAnalysis(
            timeframe="4h",
            bias=bias,
            score=score,
            key_indicators={
                "close": close,
                "ema_20": ema_20 if not np.isnan(ema_20) else None,
                "ema_50": ema_50 if not np.isnan(ema_50) else None,
                "adx_14": adx if not np.isnan(adx) else None,
                "supertrend_direction": supertrend_dir,
                "regime": regime_cls.regime.value
            },
            reasons=reasons
        )

    def evaluate_1h(self, df_1h: pd.DataFrame, idx: int = -1) -> TimeframeAnalysis:
        """
        Analyzes 1H candle for entry refinement.
        Criteria:
        - Momentum: RSI in recovery zone (40 - 68, not overbought)
        - MACD histogram expanding / recovering
        - Distance to EMA 20 (not overextended: within 2.5%)
        - Bullish candle reversal / bounce structure
        """
        if df_1h is None or len(df_1h) < 30:
            return TimeframeAnalysis(
                timeframe="1h",
                bias="NEUTRAL",
                score=40.0,
                key_indicators={},
                reasons=["Insufficient 1H data history"]
            )

        if idx < 0:
            idx = len(df_1h) + idx

        row = df_1h.iloc[idx]
        close = float(row.get('close', 0.0))
        ema_20 = float(row.get('ema_20', np.nan))
        rsi = float(row.get('rsi_14', 50.0))
        macd_hist = float(row.get('macd_hist', 0.0))
        macd_expanding = bool(row.get('macd_hist_expanding', False))
        is_hammer = bool(row.get('is_bullish_hammer', False))
        is_engulfing = bool(row.get('is_bullish_engulfing', False))
        is_breakout = bool(row.get('is_resistance_breakout', False))
        is_bounce = bool(row.get('is_ema20_pullback_bounce', False) or row.get('is_ema50_pullback_bounce', False))

        score = 0.0
        reasons = []

        # 1. RSI entry zone: 42 to 65 is sweet spot for swing entries
        if 42.0 <= rsi <= 68.0:
            score += 30.0
            reasons.append(f"1H RSI in optimal entry zone ({rsi:.1f})")
        elif rsi > 70.0:
            reasons.append(f"1H RSI overbought ({rsi:.1f}) - entry penalised")
        elif rsi < 35.0:
            score += 15.0
            reasons.append(f"1H RSI deeply oversold ({rsi:.1f})")

        # 2. MACD histogram
        if macd_expanding or macd_hist > 0:
            score += 25.0
            reasons.append("1H MACD Histogram positive / expanding")

        # 3. Not overextended from EMA 20
        if not np.isnan(ema_20) and ema_20 > 0:
            dist_pct = (close - ema_20) / ema_20 * 100.0
            if -1.5 <= dist_pct <= 2.2:
                score += 25.0
                reasons.append(f"1H Close well-positioned near EMA 20 ({dist_pct:+.2f}%)")
            elif dist_pct > 3.0:
                reasons.append(f"1H Price overextended above EMA 20 ({dist_pct:+.2f}%)")

        # 4. Price action trigger (Hammer, Engulfing, Breakout, or EMA Bounce)
        if is_hammer or is_engulfing or is_breakout or is_bounce:
            score += 20.0
            triggers = []
            if is_hammer: triggers.append("Hammer")
            if is_engulfing: triggers.append("Bullish Engulfing")
            if is_breakout: triggers.append("Resistance Breakout")
            if is_bounce: triggers.append("EMA Pullback Bounce")
            reasons.append(f"1H Price Action Trigger confirmed: {', '.join(triggers)}")

        if score >= 60.0:
            bias = "BULLISH"
        elif score <= 35.0:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        return TimeframeAnalysis(
            timeframe="1h",
            bias=bias,
            score=score,
            key_indicators={
                "close": close,
                "ema_20": ema_20 if not np.isnan(ema_20) else None,
                "rsi_14": rsi,
                "macd_hist": macd_hist,
                "price_action_trigger": is_hammer or is_engulfing or is_breakout or is_bounce
            },
            reasons=reasons
        )

    def evaluate(
        self,
        df_1d: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        current_position_open: bool = False
    ) -> MTFAnalysisResult:
        """
        Consolidates 1D, 4H, and 1H timeframes into an actionable trading decision:
        BUY, HOLD, EXIT, or NO_TRADE.
        """
        daily_res = self.evaluate_daily(df_1d)
        h4_res = self.evaluate_4h(df_4h)
        h1_res = self.evaluate_1h(df_1h)

        # Weighted alignment score: Daily (35%), 4H (40%), 1H (25%)
        alignment_score = (
            daily_res.score * 0.35 +
            h4_res.score * 0.40 +
            h1_res.score * 0.25
        )

        conflicts = []
        overall_reasons = []

        # =========================================================================
        # Conflict Detection Matrix
        # =========================================================================
        # Rule 1: Daily Bearish + 4H Bearish -> Strictly NO_TRADE for long swing
        if daily_res.bias == "BEARISH" and h4_res.bias == "BEARISH":
            conflicts.append("Macro timeframe conflict: Both Daily and 4H are BEARISH")

        # Rule 2: Daily Bullish vs 4H Bearish
        elif daily_res.bias == "BULLISH" and h4_res.bias == "BEARISH":
            conflicts.append("Timeframe divergence: Daily is Bullish but 4H is Bearish")

        # Rule 3: Daily Bearish vs 4H Bullish
        elif daily_res.bias == "BEARISH" and h4_res.bias == "BULLISH":
            conflicts.append("Counter-trend warning: 4H bounce against Bearish Daily trend")

        # =========================================================================
        # Decision Synthesis
        # =========================================================================
        if current_position_open:
            # When position is already open, check exit conditions:
            # 1. 4H flips BEARISH
            # 2. Both 1D and 4H are unsupportive
            if h4_res.bias == "BEARISH" or alignment_score < 35.0:
                decision = MTFDecision.EXIT
                overall_reasons.append("Exit triggered: 4H structure broke down or alignment collapsed")
            else:
                decision = MTFDecision.HOLD
                overall_reasons.append("Hold position: Multi-timeframe trend remains constructive")
        else:
            # When looking for new BUY entry:
            if conflicts:
                decision = MTFDecision.NO_TRADE
                overall_reasons.extend(conflicts)
            elif alignment_score >= self.min_alignment_score and daily_res.bias == "BULLISH" and h4_res.bias == "BULLISH":
                decision = MTFDecision.BUY
                overall_reasons.append(
                    f"Strong multi-timeframe alignment ({alignment_score:.1f}/100): "
                    "Daily Bullish + 4H Setup Confirmed + 1H Entry Trigger"
                )
            else:
                decision = MTFDecision.NO_TRADE
                if alignment_score < self.min_alignment_score:
                    overall_reasons.append(
                        f"Alignment score ({alignment_score:.1f}/100) below required threshold ({self.min_alignment_score:.0f})"
                    )
                else:
                    overall_reasons.append("Insufficient multi-timeframe confirmation for BUY")

        return MTFAnalysisResult(
            decision=decision,
            alignment_score=alignment_score,
            daily=daily_res,
            h4=h4_res,
            h1=h1_res,
            conflicts=conflicts,
            reasons=overall_reasons
        )

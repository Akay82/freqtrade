"""
Signal Quality Scoring Engine for BTC Swing Trading.
Implements a 100-point multi-factor rubric:
- Daily Trend Alignment: 20 pts
- 4H Trend Alignment: 20 pts
- RSI Momentum Confirmation: 15 pts
- MACD Acceleration Confirmation: 15 pts
- Volume Confirmation: 15 pts
- Price Action / Candlestick Structure: 15 pts
Enforces mandatory conditions, tracks contributing factors, and records explicit rejection reasons.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class ScoringConfig:
    """Configurable scoring thresholds and weights."""
    strong_threshold: float = 80.0
    moderate_threshold: float = 65.0
    max_daily_points: float = 20.0
    max_h4_points: float = 20.0
    max_rsi_points: float = 15.0
    max_macd_points: float = 15.0
    max_volume_points: float = 15.0
    max_price_action_points: float = 15.0
    max_overextension_pct: float = 2.8


@dataclass
class SignalScoreResult:
    """Detailed output of the signal scoring evaluation."""
    total_score: float
    classification: str                    # 'STRONG' | 'MODERATE' | 'REJECT'
    is_approved: bool
    mandatory_passed: bool
    daily_score: float
    h4_score: float
    rsi_score: float
    macd_score: float
    volume_score: float
    price_action_score: float
    factor_breakdown: Dict[str, str]       # Factor -> Status ('Confirmed', 'Weak', 'Neutral')
    contributing_factors: List[str]
    rejection_reasons: List[str]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": round(self.total_score, 1),
            "classification": self.classification,
            "is_approved": self.is_approved,
            "mandatory_passed": self.mandatory_passed,
            "daily_score": round(self.daily_score, 1),
            "h4_score": round(self.h4_score, 1),
            "rsi_score": round(self.rsi_score, 1),
            "macd_score": round(self.macd_score, 1),
            "volume_score": round(self.volume_score, 1),
            "price_action_score": round(self.price_action_score, 1),
            "factor_breakdown": self.factor_breakdown,
            "contributing_factors": self.contributing_factors,
            "rejection_reasons": self.rejection_reasons,
            "explanation": self.explanation
        }


class SignalScorer:
    """
    Evaluates candidate trade signals against a systematic 100-point rubric.
    Ensures that weak, unconfirmed, or overextended signals are rejected before reaching risk controls.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()

    def score_signal(
        self,
        df_1d: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        strategy_name: str = "GenericSwing",
        candle_idx: int = -1
    ) -> SignalScoreResult:
        """
        Calculates the multi-factor signal quality score for the specified candle.
        """
        rejection_reasons = []
        contributing_factors = []
        factor_breakdown = {}
        mandatory_passed = True

        idx_1d = candle_idx if candle_idx >= 0 else len(df_1d) + candle_idx if df_1d is not None and len(df_1d) > 0 else 0
        idx_4h = candle_idx if candle_idx >= 0 else len(df_4h) + candle_idx if df_4h is not None and len(df_4h) > 0 else 0
        idx_1h = candle_idx if candle_idx >= 0 else len(df_1h) + candle_idx if df_1h is not None and len(df_1h) > 0 else 0

        # Safe row extraction
        row_1d = df_1d.iloc[idx_1d] if df_1d is not None and len(df_1d) > 0 else pd.Series()
        row_4h = df_4h.iloc[idx_4h] if df_4h is not None and len(df_4h) > 0 else pd.Series()
        row_1h = df_1h.iloc[idx_1h] if df_1h is not None and len(df_1h) > 0 else pd.Series()

        # =========================================================================
        # 1. Daily Trend Alignment (Max: 20 pts)
        # =========================================================================
        daily_score = 0.0
        close_1d = float(row_1d.get('close', 0.0))
        ema_50_1d = float(row_1d.get('ema_50', np.nan))
        ema_200_1d = float(row_1d.get('ema_200', np.nan))
        rsi_1d = float(row_1d.get('rsi_14', 50.0))
        structure_1d = int(row_1d.get('market_structure_trend', 0))

        # Golden Cross check
        if not np.isnan(ema_200_1d):
            if ema_50_1d > ema_200_1d:
                daily_score += 10.0
                contributing_factors.append("Daily EMA 50 > EMA 200 (Golden Cross regime)")
            else:
                # Mandatory penalty: if deeply in death cross and structure is lower lows
                if structure_1d == -1:
                    rejection_reasons.append("Mandatory check failed: Daily timeframe in Death Cross with Bearish structure")
                    mandatory_passed = False

        if not np.isnan(ema_50_1d) and close_1d > ema_50_1d:
            daily_score += 5.0
            contributing_factors.append("Daily close above EMA 50")

        if rsi_1d >= 48.0:
            daily_score += 5.0
            contributing_factors.append(f"Daily RSI constructive ({rsi_1d:.1f})")

        factor_breakdown["Daily trend"] = "Confirmed" if daily_score >= 15.0 else ("Moderate" if daily_score >= 10.0 else "Weak")

        # =========================================================================
        # 2. 4H Trend Alignment (Max: 20 pts)
        # =========================================================================
        h4_score = 0.0
        close_4h = float(row_4h.get('close', 0.0))
        ema_50_4h = float(row_4h.get('ema_50', np.nan))
        supertrend_4h = int(row_4h.get('supertrend_direction', 0))
        adx_4h = float(row_4h.get('adx_14', np.nan))
        structure_4h = int(row_4h.get('market_structure_trend', 0))

        if supertrend_4h == 1:
            h4_score += 8.0
            contributing_factors.append("4H SuperTrend is Green (+1)")

        if not np.isnan(ema_50_4h) and close_4h > ema_50_4h:
            h4_score += 6.0
            contributing_factors.append("4H close above EMA 50")

        if not np.isnan(adx_4h) and adx_4h >= 22.0:
            h4_score += 6.0
            contributing_factors.append(f"4H ADX shows strong trend momentum ({adx_4h:.1f})")

        factor_breakdown["4H trend"] = "Confirmed" if h4_score >= 14.0 else ("Moderate" if h4_score >= 8.0 else "Weak")

        # =========================================================================
        # 3. RSI Momentum Confirmation (Max: 15 pts)
        # =========================================================================
        rsi_score = 0.0
        rsi_1h = float(row_1h.get('rsi_14', 50.0))
        rsi_prev_1h = float(df_1h['rsi_14'].iloc[idx_1h - 1]) if df_1h is not None and len(df_1h) > 1 else rsi_1h

        if 42.0 <= rsi_1h <= 65.0:
            rsi_score += 10.0
            contributing_factors.append(f"1H RSI in optimal entry pocket ({rsi_1h:.1f})")
        elif rsi_1h < 38.0:
            rsi_score += 7.0
            contributing_factors.append(f"1H RSI in deep value/oversold pocket ({rsi_1h:.1f})")
        elif rsi_1h > 72.0:
            rejection_reasons.append(f"1H RSI overbought ({rsi_1h:.1f})")

        # RSI upward hook
        if rsi_1h > rsi_prev_1h:
            rsi_score += 5.0
            contributing_factors.append("1H RSI momentum hooked upward")

        factor_breakdown["RSI"] = "Confirmed" if rsi_score >= 12.0 else ("Moderate" if rsi_score >= 7.0 else "Weak")

        # =========================================================================
        # 4. MACD Momentum Acceleration (Max: 15 pts)
        # =========================================================================
        macd_score = 0.0
        macd_hist_1h = float(row_1h.get('macd_hist', 0.0))
        macd_expanding = bool(row_1h.get('macd_hist_expanding', False))
        macd_line = float(row_1h.get('macd', 0.0))
        macd_sig = float(row_1h.get('macd_signal', 0.0))

        if macd_expanding:
            macd_score += 8.0
            contributing_factors.append("1H MACD histogram expanding / accelerating")

        if macd_line > macd_sig:
            macd_score += 7.0
            contributing_factors.append("1H MACD line above signal line")

        factor_breakdown["MACD"] = "Confirmed" if macd_score >= 12.0 else ("Moderate" if macd_score >= 7.0 else "Weak")

        # =========================================================================
        # 5. Volume Confirmation (Max: 15 pts)
        # =========================================================================
        volume_score = 0.0
        rel_vol = float(row_1h.get('relative_volume', 1.0))
        cmf = float(row_1h.get('cmf_20', 0.0))

        if rel_vol >= 1.25:
            volume_score += 10.0
            contributing_factors.append(f"Volume confirms breakout participation ({rel_vol:.2f}x average)")
        elif rel_vol >= 0.90:
            volume_score += 6.0
            contributing_factors.append(f"Volume healthy ({rel_vol:.2f}x average)")
        else:
            contributing_factors.append(f"Volume is below average ({rel_vol:.2f}x)")

        if cmf > 0.03:
            volume_score += 5.0
            contributing_factors.append(f"Institutional accumulation confirmed: CMF > 0 ({cmf:+.3f})")

        factor_breakdown["Volume"] = "Confirmed" if volume_score >= 11.0 else ("Moderate" if volume_score >= 6.0 else "Weak")

        # =========================================================================
        # 6. Price Action / Candlestick Structure (Max: 15 pts)
        # =========================================================================
        pa_score = 0.0
        close_1h = float(row_1h.get('close', 0.0))
        ema_20_1h = float(row_1h.get('ema_20', np.nan))
        is_hammer = bool(row_1h.get('is_bullish_hammer', False))
        is_engulfing = bool(row_1h.get('is_bullish_engulfing', False))
        is_breakout = bool(row_1h.get('is_resistance_breakout', False))
        is_bounce = bool(row_1h.get('is_ema20_pullback_bounce', False) or row_1h.get('is_ema50_pullback_bounce', False))

        # Check overextension penalty
        if not np.isnan(ema_20_1h) and ema_20_1h > 0:
            dist_pct = (close_1h - ema_20_1h) / ema_20_1h * 100.0
            if dist_pct > self.config.max_overextension_pct:
                rejection_reasons.append(f"Price overextended from 1H EMA 20 ({dist_pct:+.2f}% > {self.config.max_overextension_pct}%)")
                mandatory_passed = False
            elif -1.0 <= dist_pct <= 1.8:
                pa_score += 5.0
                contributing_factors.append("Controlled pullback near EMA 20 baseline")

        # Check candle reversal triggers
        triggers = []
        if is_hammer: triggers.append("Hammer")
        if is_engulfing: triggers.append("Bullish Engulfing")
        if is_breakout: triggers.append("Resistance Breakout")
        if is_bounce: triggers.append("EMA Bounce")

        if triggers:
            pa_score += 10.0
            contributing_factors.append(f"Price action structure confirmed: {', '.join(triggers)}")
        else:
            pa_score += 3.0

        factor_breakdown["Price action"] = "Confirmed" if pa_score >= 10.0 else ("Moderate" if pa_score >= 5.0 else "Weak")

        # =========================================================================
        # Aggregate & Classification
        # =========================================================================
        total_score = daily_score + h4_score + rsi_score + macd_score + volume_score + pa_score

        if total_score >= self.config.strong_threshold and mandatory_passed:
            classification = "STRONG"
            is_approved = True
        elif total_score >= self.config.moderate_threshold and mandatory_passed:
            classification = "MODERATE"
            is_approved = True
        else:
            classification = "REJECT"
            is_approved = False
            if total_score < self.config.moderate_threshold:
                rejection_reasons.append(f"Score ({total_score:.1f}/100) below minimum approval threshold ({self.config.moderate_threshold:.0f})")

        # Build human-readable dashboard explanation
        explanation_lines = [
            f"Strategy: {strategy_name}",
            f"Score: {total_score:.0f}/100 ({classification})",
            f"Daily trend: {factor_breakdown['Daily trend']} ({daily_score:.0f}/20)",
            f"4H trend: {factor_breakdown['4H trend']} ({h4_score:.0f}/20)",
            f"RSI: {factor_breakdown['RSI']} ({rsi_score:.0f}/15)",
            f"MACD: {factor_breakdown['MACD']} ({macd_score:.0f}/15)",
            f"Volume: {factor_breakdown['Volume']} ({volume_score:.0f}/15)",
            f"Price action: {factor_breakdown['Price action']} ({pa_score:.0f}/15)"
        ]
        if not is_approved:
            explanation_lines.append(f"Rejection reason: {'; '.join(rejection_reasons)}")

        explanation = "\n".join(explanation_lines)

        return SignalScoreResult(
            total_score=total_score,
            classification=classification,
            is_approved=is_approved,
            mandatory_passed=mandatory_passed,
            daily_score=daily_score,
            h4_score=h4_score,
            rsi_score=rsi_score,
            macd_score=macd_score,
            volume_score=volume_score,
            price_action_score=pa_score,
            factor_breakdown=factor_breakdown,
            contributing_factors=contributing_factors,
            rejection_reasons=rejection_reasons,
            explanation=explanation
        )

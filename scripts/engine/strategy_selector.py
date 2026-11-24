"""
Regime-Aware Strategy Selection and Orchestration Engine.
Coordinates Market Regime Detection, Multi-Timeframe Alignment,
Independent Strategy Evaluation, and Signal Quality Scoring.
Selects the single highest-quality setup and logs all rejection reasons.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from scripts.engine.regime_detector import MarketRegime, RegimeDetector, RegimeClassification
from scripts.engine.mtf_engine import MultiTimeframeEngine, MTFDecision, MTFAnalysisResult
from scripts.engine.signal_scorer import SignalScorer, SignalScoreResult
from scripts.engine.strategies import (
    BaseSwingStrategy,
    StrategySignal,
    StrategyA_TrendFollowing,
    StrategyB_Breakout,
    StrategyC_Pullback,
    StrategyD_RangeBound
)


@dataclass
class StrategySelectionResult:
    """Consolidated outcome of the strategy selection and signal generation process."""
    final_action: str                     # 'BUY' | 'HOLD' | 'EXIT' | 'NONE'
    selected_strategy: Optional[str]
    active_regime: MarketRegime
    regime_confidence: float
    eligible_strategies: List[str]
    ineligible_strategies: List[str]
    candidate_signals: List[StrategySignal]
    best_signal: Optional[StrategySignal]
    best_score: Optional[SignalScoreResult]
    rejections: List[Dict[str, Any]]      # [{'strategy': name, 'reasons': [...]}]
    mtf_decision: MTFDecision
    explanation: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_action": self.final_action,
            "selected_strategy": self.selected_strategy,
            "active_regime": self.active_regime.value,
            "regime_confidence": round(self.regime_confidence, 1),
            "eligible_strategies": self.eligible_strategies,
            "ineligible_strategies": self.ineligible_strategies,
            "candidate_count": len(self.candidate_signals),
            "best_signal": self.best_signal.to_dict() if self.best_signal else None,
            "best_score": self.best_score.to_dict() if self.best_score else None,
            "rejections": self.rejections,
            "mtf_decision": self.mtf_decision.value,
            "explanation": self.explanation,
            "timestamp": self.timestamp
        }


class StrategySelector:
    """
    Selects and coordinates trading strategies based on detected market conditions.
    Never forces trades when conditions conflict or setup scores fall below quality thresholds.
    """

    def __init__(
        self,
        strategies: Optional[List[BaseSwingStrategy]] = None,
        regime_detector: Optional[RegimeDetector] = None,
        mtf_engine: Optional[MultiTimeframeEngine] = None,
        signal_scorer: Optional[SignalScorer] = None
    ):
        self.strategies = strategies or [
            StrategyA_TrendFollowing(),
            StrategyB_Breakout(),
            StrategyC_Pullback(),
            StrategyD_RangeBound(is_enabled=False)  # Disabled until independently tested
        ]
        self.regime_detector = regime_detector or RegimeDetector()
        self.mtf_engine = mtf_engine or MultiTimeframeEngine()
        self.signal_scorer = signal_scorer or SignalScorer()

    def select_and_evaluate(
        self,
        df_1d: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        candle_idx: int = -1,
        current_position_open: bool = False
    ) -> StrategySelectionResult:
        """
        Executes the systematic strategy selection pipeline:
        1. Classifies active market regime.
        2. Evaluates multi-timeframe alignment & conflict handling.
        3. Filters eligible strategy modules by regime.
        4. Gathers candidate setups and scores them against the 100-pt rubric.
        5. Selects the single highest-scoring setup and explains the decision.
        """
        rejections = []
        candidate_signals = []

        # Step 1: Detect Market Regime (evaluated primarily on 4H structure)
        regime_res = self.regime_detector.classify_candle(df_4h, candle_idx=candle_idx)
        active_regime = regime_res.regime
        regime_conf = regime_res.confidence_score

        # Step 2: Multi-Timeframe Alignment
        mtf_res = self.mtf_engine.evaluate(df_1d, df_4h, df_1h, current_position_open=current_position_open)

        # Handle existing position management (HOLD / EXIT)
        if current_position_open:
            if mtf_res.decision == MTFDecision.EXIT:
                return StrategySelectionResult(
                    final_action="EXIT",
                    selected_strategy=None,
                    active_regime=active_regime,
                    regime_confidence=regime_conf,
                    eligible_strategies=[],
                    ineligible_strategies=[s.name for s in self.strategies],
                    candidate_signals=[],
                    best_signal=None,
                    best_score=None,
                    rejections=[],
                    mtf_decision=mtf_res.decision,
                    explanation=f"EXIT position: {'; '.join(mtf_res.reasons)}"
                )
            else:
                return StrategySelectionResult(
                    final_action="HOLD",
                    selected_strategy=None,
                    active_regime=active_regime,
                    regime_confidence=regime_conf,
                    eligible_strategies=[],
                    ineligible_strategies=[s.name for s in self.strategies],
                    candidate_signals=[],
                    best_signal=None,
                    best_score=None,
                    rejections=[],
                    mtf_decision=mtf_res.decision,
                    explanation=f"HOLD active position: {'; '.join(mtf_res.reasons)}"
                )

        # Step 3: Check for Macro Timeframe Conflicts before evaluating entries
        if mtf_res.decision == MTFDecision.NO_TRADE:
            return StrategySelectionResult(
                final_action="NONE",
                selected_strategy=None,
                active_regime=active_regime,
                regime_confidence=regime_conf,
                eligible_strategies=[],
                ineligible_strategies=[s.name for s in self.strategies],
                candidate_signals=[],
                best_signal=None,
                best_score=None,
                rejections=[{
                    "strategy": "ALL",
                    "reason": f"Multi-timeframe filter blocked entries: {'; '.join(mtf_res.reasons)}"
                }],
                mtf_decision=mtf_res.decision,
                explanation=f"NO TRADE: Timeframe conflict or unsupportive macro structure ({'; '.join(mtf_res.reasons)})"
            )

        # Step 4: Determine Strategy Eligibility by Regime
        eligible = []
        ineligible = []
        for strat in self.strategies:
            if not strat.is_enabled:
                ineligible.append(f"{strat.name} (Disabled)")
                continue

            if strat.is_eligible(active_regime):
                eligible.append(strat.name)
            else:
                ineligible.append(f"{strat.name} (Incompatible with {active_regime.value})")

        # Step 5: Evaluate Setups on Eligible Strategies
        approved_candidates: List[Tuple[StrategySignal, SignalScoreResult]] = []

        for strat in self.strategies:
            if strat.name not in eligible:
                continue

            sig = strat.evaluate_setup(df_1d, df_4h, df_1h, candle_idx=candle_idx)
            if sig is None or sig.action != "BUY":
                rejections.append({
                    "strategy": strat.name,
                    "reason": "Strategy setup conditions not met on current candle"
                })
                continue

            candidate_signals.append(sig)

            # Score the candidate signal
            score_res = self.signal_scorer.score_signal(
                df_1d, df_4h, df_1h,
                strategy_name=strat.name,
                candle_idx=candle_idx
            )

            if score_res.is_approved:
                approved_candidates.append((sig, score_res))
            else:
                rejections.append({
                    "strategy": strat.name,
                    "score": score_res.total_score,
                    "reasons": score_res.rejection_reasons
                })

        # Step 6: Select Highest Quality Candidate
        if approved_candidates:
            # Sort by total score descending
            approved_candidates.sort(key=lambda pair: pair[1].total_score, reverse=True)
            best_sig, best_sc = approved_candidates[0]

            explanation_lines = [
                f"BUY Signal Approved: {best_sig.strategy_name}",
                f"Market Regime: {active_regime.value} (Confidence: {regime_conf:.0f}%)",
                f"Signal Quality Score: {best_sc.total_score:.0f}/100 ({best_sc.classification})",
                f"Entry Price: ${best_sig.entry_price:,.2f}",
                f"Stop Loss: ${best_sig.suggested_stop_loss:,.2f}",
                f"Take Profit: ${best_sig.suggested_take_profit:,.2f}",
                f"Reward/Risk: {best_sig.risk_reward_ratio:.2f}:1",
                f"Supporting Factors: {'; '.join(best_sig.reasons)}"
            ]

            return StrategySelectionResult(
                final_action="BUY",
                selected_strategy=best_sig.strategy_name,
                active_regime=active_regime,
                regime_confidence=regime_conf,
                eligible_strategies=eligible,
                ineligible_strategies=ineligible,
                candidate_signals=candidate_signals,
                best_signal=best_sig,
                best_score=best_sc,
                rejections=rejections,
                mtf_decision=mtf_res.decision,
                explanation="\n".join(explanation_lines)
            )

        # No approved setup
        return StrategySelectionResult(
            final_action="NONE",
            selected_strategy=None,
            active_regime=active_regime,
            regime_confidence=regime_conf,
            eligible_strategies=eligible,
            ineligible_strategies=ineligible,
            candidate_signals=candidate_signals,
            best_signal=None,
            best_score=None,
            rejections=rejections,
            mtf_decision=mtf_res.decision,
            explanation=f"NO TRADE: No strategy generated a qualifying setup in {active_regime.value} regime"
        )

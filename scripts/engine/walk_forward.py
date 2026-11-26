"""
Walk-Forward Validation Engine (Section 13).
Splits historical data into chronological Train -> Validate -> Out-of-Sample Test
rolling windows to evaluate parameter generalization across market regimes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd

from scripts.engine.backtester import SwingBacktester, BacktestConfig, BacktestMetrics


@dataclass
class WalkForwardWindowResult:
    """Results for a single rolling walk-forward window."""
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    selected_min_score: float
    in_sample_return_pct: float
    oos_metrics: BacktestMetrics
    oos_buy_hold_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_index": self.window_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "selected_min_score": self.selected_min_score,
            "in_sample_return_pct": round(self.in_sample_return_pct, 2),
            "oos_metrics": self.oos_metrics.to_dict(),
            "oos_buy_hold_pct": round(self.oos_buy_hold_pct, 2),
        }


@dataclass
class WalkForwardSummary:
    """Aggregated out-of-sample walk-forward validation report."""
    total_windows: int
    avg_oos_return_pct: float
    avg_oos_win_rate_pct: float
    avg_oos_sharpe: float
    total_oos_trades: int
    positive_windows: int
    windows: List[WalkForwardWindowResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_windows": self.total_windows,
            "avg_oos_return_pct": round(self.avg_oos_return_pct, 2),
            "avg_oos_win_rate_pct": round(self.avg_oos_win_rate_pct, 1),
            "avg_oos_sharpe": round(self.avg_oos_sharpe, 2),
            "total_oos_trades": self.total_oos_trades,
            "positive_windows": self.positive_windows,
            "windows": [w.to_dict() for w in self.windows],
        }


class WalkForwardValidator:
    """
    Runs chronological rolling walk-forward validation without tuning on final test data.
    """

    def __init__(
        self,
        train_bars_1h: int = 2400,    # ~100 days
        test_bars_1h: int = 720,      # ~30 days
        step_bars_1h: int = 720,      # ~30 days step
        candidate_scores: Optional[List[float]] = None
    ):
        self.train_bars_1h = train_bars_1h
        self.test_bars_1h = test_bars_1h
        self.step_bars_1h = step_bars_1h
        self.candidate_scores = candidate_scores or [65.0, 72.0]

    def run(
        self,
        df_1d: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        base_config: Optional[BacktestConfig] = None
    ) -> WalkForwardSummary:
        cfg_base = base_config or BacktestConfig()
        df_1h_sorted = df_1h.sort_values('date').reset_index(drop=True)
        n = len(df_1h_sorted)

        windows: List[WalkForwardWindowResult] = []
        idx = 0
        win_num = 1

        while idx + self.train_bars_1h + self.test_bars_1h <= n:
            train_slice_1h = df_1h_sorted.iloc[idx: idx + self.train_bars_1h].copy()
            # Include warmup overlap for test window so indicators don't start cold,
            # while only evaluating trades inside the test range
            warmup_overlap = min(cfg_base.warmup_bars, self.train_bars_1h)
            test_start_idx = idx + self.train_bars_1h - warmup_overlap
            test_end_idx = idx + self.train_bars_1h + self.test_bars_1h
            test_slice_1h = df_1h_sorted.iloc[test_start_idx: test_end_idx].copy()

            train_end_dt = train_slice_1h['date'].iloc[-1]
            test_end_dt = test_slice_1h['date'].iloc[-1]

            # Filter 1D and 4H strictly up to each window's end date (zero future leakage)
            train_1d = df_1d[df_1d['date'] <= train_end_dt].copy()
            train_4h = df_4h[df_4h['date'] <= train_end_dt].copy()
            test_1d = df_1d[df_1d['date'] <= test_end_dt].copy()
            test_4h = df_4h[df_4h['date'] <= test_end_dt].copy()

            # 1. Select best parameter on Train+Validate window
            best_score_param = self.candidate_scores[0]
            best_is_return = -9999.0
            for cand in self.candidate_scores:
                is_cfg = BacktestConfig(
                    initial_capital=cfg_base.initial_capital,
                    fee_rate=cfg_base.fee_rate,
                    slippage_bps=cfg_base.slippage_bps,
                    risk_per_trade_pct=cfg_base.risk_per_trade_pct,
                    min_signal_score=cand,
                    warmup_bars=cfg_base.warmup_bars,
                    strategy_filter=cfg_base.strategy_filter,
                )
                is_res = SwingBacktester(is_cfg).run(train_1d, train_4h, train_slice_1h, run_id=f"wf-is-{win_num}")
                if is_res.metrics.total_return_pct > best_is_return:
                    best_is_return = is_res.metrics.total_return_pct
                    best_score_param = cand

            # 2. Evaluate strictly Out-Of-Sample on Test window
            oos_cfg = BacktestConfig(
                initial_capital=cfg_base.initial_capital,
                fee_rate=cfg_base.fee_rate,
                slippage_bps=cfg_base.slippage_bps,
                risk_per_trade_pct=cfg_base.risk_per_trade_pct,
                min_signal_score=best_score_param,
                warmup_bars=warmup_overlap,
                strategy_filter=cfg_base.strategy_filter,
            )
            oos_res = SwingBacktester(oos_cfg).run(test_1d, test_4h, test_slice_1h, run_id=f"wf-oos-{win_num}")

            windows.append(WalkForwardWindowResult(
                window_index=win_num,
                train_start=str(train_slice_1h['date'].iloc[0]),
                train_end=str(train_end_dt),
                test_start=str(df_1h_sorted['date'].iloc[idx + self.train_bars_1h]),
                test_end=str(test_end_dt),
                selected_min_score=best_score_param,
                in_sample_return_pct=best_is_return,
                oos_metrics=oos_res.metrics,
                oos_buy_hold_pct=oos_res.benchmarks.get("btc_buy_hold_pct", 0.0),
            ))

            idx += self.step_bars_1h
            win_num += 1

        total_w = len(windows)
        if total_w == 0:
            return WalkForwardSummary(0, 0.0, 0.0, 0.0, 0, 0, [])

        avg_ret = sum(w.oos_metrics.total_return_pct for w in windows) / total_w
        avg_wr = sum(w.oos_metrics.win_rate_pct for w in windows) / total_w
        avg_sh = sum(w.oos_metrics.sharpe_ratio for w in windows) / total_w
        tot_trades = sum(w.oos_metrics.total_trades for w in windows)
        pos_w = sum(1 for w in windows if w.oos_metrics.total_return_pct >= 0)

        return WalkForwardSummary(
            total_windows=total_w,
            avg_oos_return_pct=avg_ret,
            avg_oos_win_rate_pct=avg_wr,
            avg_oos_sharpe=avg_sh,
            total_oos_trades=tot_trades,
            positive_windows=pos_w,
            windows=windows,
        )

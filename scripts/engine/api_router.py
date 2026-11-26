"""
FastAPI Router for Systematic BTC Swing Engine Endpoints (Section 17).
Exposes market overview, regime classification, signal analysis, risk monitor,
background backtesting, ML model status, and performance attribution.
"""

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from scripts.engine.data_validator import DataValidator
from scripts.engine.indicators import IndicatorEngine
from scripts.engine.price_action import PriceActionDetector
from scripts.engine.regime_detector import RegimeDetector
from scripts.engine.mtf_engine import MultiTimeframeEngine
from scripts.engine.strategy_selector import StrategySelector
from scripts.engine.risk_manager import RiskManager, RiskConfig
from scripts.engine.backtester import SwingBacktester, BacktestConfig
from scripts.engine.storage import EngineStorage


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "user_data" / "data" / "binance"

engine_router = APIRouter(prefix="/api/engine", tags=["Systematic Engine"])

# Shared singletons
_storage = EngineStorage()
_risk_manager = RiskManager(initial_equity=1000.0, config=RiskConfig())
_backtest_lock = threading.Lock()
_backtest_jobs: Dict[str, str] = {}


class KillSwitchRequest(BaseModel):
    active: bool
    reason: str = "Dashboard operator toggle"


class BacktestRunRequest(BaseModel):
    strategy_filter: Optional[str] = None
    initial_capital: float = 1000.0
    risk_per_trade_pct: float = 0.01
    min_signal_score: float = 65.0


def _load_prepared_mtf():
    """Loads local BTC/USDT 1d, 4h, 1h feather files and computes indicators & price action."""
    validator = DataValidator()
    ie = IndicatorEngine()
    pa = PriceActionDetector()

    p_1d = DATA_DIR / "BTC_USDT-1d.feather"
    p_4h = DATA_DIR / "BTC_USDT-4h.feather"
    p_1h = DATA_DIR / "BTC_USDT-1h.feather"

    if not (p_1d.exists() and p_4h.exists() and p_1h.exists()):
        raise HTTPException(status_code=503, detail="Historical BTC/USDT 1d/4h/1h candle files not found")

    df_1d = pa.compute_all_price_action(
        ie.compute_all_indicators(validator.clean_and_normalize(pd.read_feather(p_1d), "1d", drop_incomplete=False))
    )
    df_4h = pa.compute_all_price_action(
        ie.compute_all_indicators(validator.clean_and_normalize(pd.read_feather(p_4h), "4h", drop_incomplete=False))
    )
    df_1h = pa.compute_all_price_action(
        ie.compute_all_indicators(validator.clean_and_normalize(pd.read_feather(p_1h), "1h", drop_incomplete=False))
    )
    return df_1d, df_4h, df_1h


@engine_router.get("/market-overview")
def get_market_overview() -> Dict[str, Any]:
    """Section 18A: Current BTC price, 24h change, Market Regime, 1D/4H/1H indicators."""
    df_1d, df_4h, df_1h = _load_prepared_mtf()
    regime_cls = RegimeDetector().classify_candle(df_4h, candle_idx=-1)
    mtf_res = MultiTimeframeEngine().evaluate(df_1d, df_4h, df_1h)

    latest_1h = df_1h.iloc[-1]
    prev_24h = df_1h.iloc[-25] if len(df_1h) >= 25 else df_1h.iloc[0]
    price = float(latest_1h["close"])
    change_24h_pct = ((price - float(prev_24h["close"])) / float(prev_24h["close"])) * 100.0

    _storage.save_regime_snapshot(regime_cls.to_dict())

    return {
        "pair": "BTC/USDT",
        "price": round(price, 2),
        "change_24h_pct": round(change_24h_pct, 2),
        "candle_time_utc": str(latest_1h["date"]),
        "regime": regime_cls.to_dict(),
        "mtf": mtf_res.to_dict(),
        "indicators_1h": {
            "rsi_14": round(float(latest_1h.get("rsi_14", 50.0)), 1),
            "macd_hist": round(float(latest_1h.get("macd_hist", 0.0)), 2),
            "atr_14": round(float(latest_1h.get("atr_14", 0.0)), 2),
            "atr_pct": round(float(latest_1h.get("atr_pct", 0.0)), 2),
            "relative_volume": round(float(latest_1h.get("relative_volume", 1.0)), 2),
            "adx_14": round(float(latest_1h.get("adx_14", 0.0)), 1),
        },
    }


@engine_router.get("/signals")
def get_signal_analysis() -> Dict[str, Any]:
    """Section 18B & 18C: Strategy Monitor, Signal Analysis, and Rejection History."""
    df_1d, df_4h, df_1h = _load_prepared_mtf()
    selector = StrategySelector()
    sel_res = selector.select_and_evaluate(df_1d, df_4h, df_1h, candle_idx=-1, current_position_open=False)
    eval_dict = sel_res.to_dict()
    _storage.save_signal_evaluation(eval_dict)

    return {
        "current": eval_dict,
        "strategies": [
            {
                "name": s.name,
                "type": s.strategy_type,
                "enabled": s.is_enabled,
                "eligible": s.is_eligible(sel_res.active_regime),
                "compatible_regimes": [r.value for r in s.compatible_regimes],
            }
            for s in selector.strategies
        ],
        "recent_history": _storage.get_recent_signals(limit=15),
    }


@engine_router.get("/risk")
def get_risk_status() -> Dict[str, Any]:
    """Section 18G: Risk Monitor telemetry & active lockouts."""
    return _risk_manager.get_status(open_exposure=0.0)


@engine_router.post("/risk/kill-switch")
def toggle_kill_switch(req: KillSwitchRequest) -> Dict[str, Any]:
    """Toggles the manual emergency trading disable switch."""
    _risk_manager.set_manual_kill_switch(req.active, req.reason)
    return {"status": "ok", "manual_kill_switch": _risk_manager.manual_kill_switch, "risk": _risk_manager.get_status()}


def _run_backtest_job(run_id: str, req: BacktestRunRequest) -> None:
    try:
        _backtest_jobs[run_id] = "RUNNING"
        df_1d, df_4h, df_1h = _load_prepared_mtf()
        cfg = BacktestConfig(
            initial_capital=req.initial_capital,
            risk_per_trade_pct=req.risk_per_trade_pct,
            min_signal_score=req.min_signal_score,
            strategy_filter=req.strategy_filter,
        )
        res = SwingBacktester(cfg).run(df_1d, df_4h, df_1h, run_id=run_id)
        _storage.save_backtest_run(res.to_dict())
        _backtest_jobs[run_id] = "COMPLETED"
    except Exception as exc:
        _backtest_jobs[run_id] = f"FAILED: {exc}"


@engine_router.post("/backtest/run")
def start_backtest(req: BacktestRunRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Starts an asynchronous historical backtest job (Section 17)."""
    run_id = f"bt-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    _backtest_jobs[run_id] = "QUEUED"
    background_tasks.add_task(_run_backtest_job, run_id, req)
    return {"run_id": run_id, "status": "QUEUED"}


@engine_router.get("/backtest/history")
def get_backtest_history() -> Dict[str, Any]:
    """Returns completed backtest runs and active job statuses."""
    return {
        "jobs": _backtest_jobs,
        "runs": _storage.get_backtest_history(limit=10),
    }


@engine_router.get("/ml-status")
def get_ml_status() -> Dict[str, Any]:
    """Section 18H: Active ML model metadata and validation metrics."""
    active_model = _storage.get_active_ml_model()
    return {
        "implemented": active_model is not None,
        "active_model": active_model,
    }


@engine_router.get("/attribution")
def get_performance_attribution() -> Dict[str, Any]:
    """Section 15: Segmented paper-trading attribution by strategy, regime, and score."""
    return _storage.get_attribution_summary()

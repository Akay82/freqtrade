"""
SQLite Persistence Layer for the BTC Systematic Swing Engine (Section 16).
Stores regime snapshots, signal evaluations & rejections, backtest runs,
risk events, ML model metadata, and persistent paper-trading state/attributions.
"""

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT / "user_data" / "db" / "engine_state.sqlite"


class EngineStorage:
    """Thread-safe SQLite storage with indexes and transactional helpers."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS regime_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    data_quality_status TEXT NOT NULL,
                    summary_reason TEXT,
                    indicators_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_regime_ts ON regime_snapshots(timestamp_utc);

                CREATE TABLE IF NOT EXISTS signal_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    action TEXT NOT NULL,
                    selected_strategy TEXT,
                    regime TEXT NOT NULL,
                    score REAL,
                    classification TEXT,
                    ml_probability REAL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    explanation TEXT,
                    rejections_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_signal_ts ON signal_evaluations(timestamp_utc);

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    total_return_pct REAL,
                    win_rate_pct REAL,
                    profit_factor REAL,
                    max_drawdown_pct REAL,
                    sharpe_ratio REAL,
                    total_trades INTEGER,
                    result_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_bt_created ON backtest_runs(created_at_utc);

                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_risk_ts ON risk_events(timestamp_utc);

                CREATE TABLE IF NOT EXISTS ml_models (
                    version TEXT PRIMARY KEY,
                    trained_at_utc TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    precision_score REAL,
                    recall_score REAL,
                    brier_score REAL,
                    roc_auc REAL,
                    baseline_accuracy REAL,
                    feature_names_json TEXT
                );

                CREATE TABLE IF NOT EXISTS paper_account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    initial_capital REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    peak_equity REAL NOT NULL,
                    daily_start_equity REAL NOT NULL,
                    current_utc_date TEXT NOT NULL,
                    manual_kill_switch INTEGER NOT NULL DEFAULT 0,
                    updated_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_positions (
                    position_id TEXT PRIMARY KEY,
                    strategy_name TEXT NOT NULL,
                    regime_at_entry TEXT NOT NULL,
                    entry_time_utc TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    entry_fee REAL NOT NULL,
                    entry_cost REAL NOT NULL,
                    initial_stop_loss REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    partial_tp_price REAL NOT NULL,
                    partial_taken INTEGER NOT NULL DEFAULT 0,
                    signal_score REAL NOT NULL,
                    ml_probability REAL,
                    entry_conditions_json TEXT
                );

                CREATE TABLE IF NOT EXISTS paper_closed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    regime_at_entry TEXT NOT NULL,
                    entry_time_utc TEXT NOT NULL,
                    exit_time_utc TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    signal_score REAL NOT NULL,
                    ml_probability REAL,
                    fees REAL NOT NULL,
                    slippage_cost REAL NOT NULL,
                    net_pnl REAL NOT NULL,
                    return_pct REAL NOT NULL,
                    hold_hours REAL NOT NULL,
                    exit_reason TEXT NOT NULL,
                    entry_conditions_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_closed_strat ON paper_closed_trades(strategy_name);
                CREATE INDEX IF NOT EXISTS idx_closed_regime ON paper_closed_trades(regime_at_entry);
            """)

    def save_regime_snapshot(self, regime_dict: Dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO regime_snapshots
                   (timestamp_utc, regime, confidence_score, data_quality_status, summary_reason, indicators_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    regime_dict.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    regime_dict["regime"],
                    float(regime_dict["confidence_score"]),
                    regime_dict.get("data_quality_status", "HEALTHY"),
                    regime_dict.get("summary_reason", ""),
                    json.dumps(regime_dict.get("supporting_indicators", {})),
                ),
            )

    def save_signal_evaluation(self, eval_dict: Dict[str, Any], ml_prob: Optional[float] = None) -> None:
        best_sig = eval_dict.get("best_signal") or {}
        best_score = eval_dict.get("best_score") or {}
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO signal_evaluations
                   (timestamp_utc, action, selected_strategy, regime, score, classification,
                    ml_probability, entry_price, stop_loss, take_profit, explanation, rejections_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    eval_dict.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    eval_dict.get("final_action", "NONE"),
                    eval_dict.get("selected_strategy"),
                    eval_dict.get("active_regime", "UNCERTAIN"),
                    best_score.get("total_score"),
                    best_score.get("classification"),
                    ml_prob,
                    best_sig.get("entry_price"),
                    best_sig.get("suggested_stop_loss"),
                    best_sig.get("suggested_take_profit"),
                    eval_dict.get("explanation", ""),
                    json.dumps(eval_dict.get("rejections", [])),
                ),
            )

    def get_recent_signals(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM signal_evaluations ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["rejections"] = json.loads(d.pop("rejections_json") or "[]")
                out.append(d)
            return out

    def save_backtest_run(self, result_dict: Dict[str, Any]) -> None:
        m = result_dict.get("metrics", {})
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO backtest_runs
                   (run_id, created_at_utc, strategy_name, start_date, end_date,
                    total_return_pct, win_rate_pct, profit_factor, max_drawdown_pct,
                    sharpe_ratio, total_trades, result_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result_dict["run_id"],
                    datetime.now(timezone.utc).isoformat(),
                    result_dict["strategy_name"],
                    result_dict.get("start_date", ""),
                    result_dict.get("end_date", ""),
                    m.get("total_return_pct", 0.0),
                    m.get("win_rate_pct", 0.0),
                    m.get("profit_factor", 0.0),
                    m.get("max_drawdown_pct", 0.0),
                    m.get("sharpe_ratio", 0.0),
                    m.get("total_trades", 0),
                    json.dumps(result_dict),
                ),
            )

    def get_backtest_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT result_json FROM backtest_runs ORDER BY created_at_utc DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [json.loads(r["result_json"]) for r in rows]

    def save_ml_model_metadata(self, meta: Dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE ml_models SET is_active = 0")
            conn.execute(
                """INSERT OR REPLACE INTO ml_models
                   (version, trained_at_utc, model_type, model_path, is_active,
                    precision_score, recall_score, brier_score, roc_auc,
                    baseline_accuracy, feature_names_json)
                   VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)""",
                (
                    meta["version"],
                    meta["trained_at_utc"],
                    meta["model_type"],
                    meta["model_path"],
                    meta.get("precision_score"),
                    meta.get("recall_score"),
                    meta.get("brier_score"),
                    meta.get("roc_auc"),
                    meta.get("baseline_accuracy"),
                    json.dumps(meta.get("feature_names", [])),
                ),
            )

    def get_active_ml_model(self) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ml_models WHERE is_active = 1 ORDER BY trained_at_utc DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["feature_names"] = json.loads(d.pop("feature_names_json") or "[]")
            return d

    def get_attribution_summary(self) -> Dict[str, Any]:
        """Generates segmented performance reports by strategy, regime, and score range (Section 15)."""
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM paper_closed_trades ORDER BY id ASC").fetchall()
            trades = [dict(r) for r in rows]

        def _agg(group_key: str) -> Dict[str, Dict[str, Any]]:
            buckets: Dict[str, Dict[str, Any]] = {}
            for t in trades:
                k = str(t[group_key])
                b = buckets.setdefault(k, {"trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0, "total_fees": 0.0})
                b["trades"] += 1
                if t["net_pnl"] > 0:
                    b["wins"] += 1
                else:
                    b["losses"] += 1
                b["net_pnl"] += t["net_pnl"]
                b["total_fees"] += t["fees"]
            for b in buckets.values():
                b["win_rate_pct"] = round(b["wins"] / b["trades"] * 100.0, 1) if b["trades"] > 0 else 0.0
                b["net_pnl"] = round(b["net_pnl"], 2)
                b["total_fees"] = round(b["total_fees"], 4)
            return buckets

        score_buckets: Dict[str, Dict[str, Any]] = {}
        for t in trades:
            sc = t["signal_score"]
            label = "80-100 (Strong)" if sc >= 80 else ("65-79 (Moderate)" if sc >= 65 else "<65")
            b = score_buckets.setdefault(label, {"trades": 0, "wins": 0, "net_pnl": 0.0})
            b["trades"] += 1
            if t["net_pnl"] > 0:
                b["wins"] += 1
            b["net_pnl"] += t["net_pnl"]
        for b in score_buckets.values():
            b["win_rate_pct"] = round(b["wins"] / b["trades"] * 100.0, 1) if b["trades"] > 0 else 0.0
            b["net_pnl"] = round(b["net_pnl"], 2)

        return {
            "total_closed_trades": len(trades),
            "by_strategy": _agg("strategy_name"),
            "by_regime": _agg("regime_at_entry"),
            "by_score_range": score_buckets,
            "recent_trades": trades[-20:][::-1],
        }

"""
Persistent Paper-Trading Execution & Attribution Engine (Sections 11, 15, 19).
Coordinates DataValidator -> IndicatorEngine -> PriceActionDetector ->
StrategySelector -> MLSignalFilter -> RiskManager -> SQLite Persistence.
Restores exact account balance and open positions across application restarts.
Live trading is strictly disabled.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from scripts.engine.data_validator import DataValidator
from scripts.engine.indicators import IndicatorEngine
from scripts.engine.price_action import PriceActionDetector
from scripts.engine.strategy_selector import StrategySelector
from scripts.engine.ml_filter import MLSignalFilter, MLConfig
from scripts.engine.risk_manager import RiskManager, RiskConfig
from scripts.engine.storage import EngineStorage


class PaperTradingEngine:
    """
    Restart-safe paper-trading coordinator with full post-trade attribution.
    """

    LIVE_TRADING_ENABLED = False  # Strictly disabled until separate explicit approval

    def __init__(
        self,
        initial_capital: float = 1000.0,
        storage: Optional[EngineStorage] = None,
        risk_config: Optional[RiskConfig] = None,
        ml_config: Optional[MLConfig] = None,
    ):
        self.storage = storage or EngineStorage()
        self.validator = DataValidator()
        self.ie = IndicatorEngine()
        self.pa = PriceActionDetector()
        self.selector = StrategySelector()
        self.ml_filter = MLSignalFilter(config=ml_config or MLConfig(enabled=False), storage=self.storage)
        self.ml_filter.load_latest()

        self.risk_manager = RiskManager(initial_equity=initial_capital, config=risk_config or RiskConfig())
        self.cash_balance: float = float(initial_capital)
        self.open_positions: List[Dict[str, Any]] = []
        self._restore_or_init_state(initial_capital)

    def _restore_or_init_state(self, initial_capital: float) -> None:
        """Restores account balance and open positions from SQLite on startup."""
        with self.storage._lock, self.storage._connect() as conn:
            acct = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
            if acct:
                self.cash_balance = float(acct["cash_balance"])
                self.risk_manager.initial_equity = float(acct["initial_capital"])
                self.risk_manager.peak_equity = float(acct["peak_equity"])
                self.risk_manager.daily_start_equity = float(acct["daily_start_equity"])
                self.risk_manager.current_utc_date = str(acct["current_utc_date"])
                self.risk_manager.manual_kill_switch = bool(acct["manual_kill_switch"])
            else:
                now_utc = datetime.now(timezone.utc)
                conn.execute(
                    """INSERT INTO paper_account
                       (id, initial_capital, cash_balance, peak_equity, daily_start_equity,
                        current_utc_date, manual_kill_switch, updated_at_utc)
                       VALUES (1, ?, ?, ?, ?, ?, 0, ?)""",
                    (
                        initial_capital,
                        initial_capital,
                        initial_capital,
                        initial_capital,
                        now_utc.date().isoformat(),
                        now_utc.isoformat(),
                    ),
                )

            pos_rows = conn.execute("SELECT * FROM paper_positions").fetchall()
            self.open_positions = [dict(r) for r in pos_rows]

            # Update current equity = cash + open cost basis until marked to market
            open_cost = sum(float(p["entry_cost"]) for p in self.open_positions)
            self.risk_manager.current_equity = self.cash_balance + open_cost

    def _persist_account(self) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.storage._lock, self.storage._connect() as conn:
            conn.execute(
                """UPDATE paper_account
                   SET cash_balance = ?, peak_equity = ?, daily_start_equity = ?,
                       current_utc_date = ?, manual_kill_switch = ?, updated_at_utc = ?
                   WHERE id = 1""",
                (
                    self.cash_balance,
                    self.risk_manager.peak_equity,
                    self.risk_manager.daily_start_equity,
                    self.risk_manager.current_utc_date,
                    int(self.risk_manager.manual_kill_switch),
                    now_iso,
                ),
            )

    def _close_position(
        self,
        pos: Dict[str, Any],
        exit_price: float,
        exit_reason: str,
        now_utc: datetime,
        qty_to_close: Optional[float] = None
    ) -> Dict[str, Any]:
        """Simulates realistic exit with fees/slippage and records closed-trade attribution."""
        slip_rate = self.risk_manager.config.slippage_bps / 10000.0
        fee_rate = self.risk_manager.config.fee_rate
        exec_exit = exit_price * (1.0 - slip_rate)

        full_qty = float(pos["quantity"])
        close_qty = min(full_qty, qty_to_close) if qty_to_close is not None else full_qty
        proportion = close_qty / full_qty if full_qty > 0 else 1.0

        gross_proceeds = close_qty * exec_exit
        exit_fee = gross_proceeds * fee_rate
        net_proceeds = gross_proceeds - exit_fee

        alloc_entry_cost = float(pos["entry_cost"]) * proportion
        alloc_entry_fee = float(pos["entry_fee"]) * proportion
        net_pnl = net_proceeds - alloc_entry_cost
        ret_pct = (net_pnl / alloc_entry_cost * 100.0) if alloc_entry_cost > 0 else 0.0
        slip_cost = (float(pos["entry_price"]) + exec_exit) * slip_rate * close_qty

        entry_dt = datetime.fromisoformat(pos["entry_time_utc"])
        hold_hours = max(0.0, (now_utc - entry_dt).total_seconds() / 3600.0)

        self.cash_balance += net_proceeds

        with self.storage._lock, self.storage._connect() as conn:
            conn.execute(
                """INSERT INTO paper_closed_trades
                   (position_id, strategy_name, regime_at_entry, entry_time_utc, exit_time_utc,
                    entry_price, exit_price, quantity, stop_loss, take_profit, signal_score,
                    ml_probability, fees, slippage_cost, net_pnl, return_pct, hold_hours,
                    exit_reason, entry_conditions_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pos["position_id"],
                    pos["strategy_name"],
                    pos["regime_at_entry"],
                    pos["entry_time_utc"],
                    now_utc.isoformat(),
                    float(pos["entry_price"]),
                    exec_exit,
                    close_qty,
                    float(pos["initial_stop_loss"]),
                    float(pos["take_profit"]),
                    float(pos["signal_score"]),
                    pos.get("ml_probability"),
                    alloc_entry_fee + exit_fee,
                    slip_cost,
                    net_pnl,
                    ret_pct,
                    hold_hours,
                    exit_reason,
                    pos.get("entry_conditions_json", "[]"),
                ),
            )
            if close_qty >= full_qty - 1e-12:
                conn.execute("DELETE FROM paper_positions WHERE position_id = ?", (pos["position_id"],))
                self.open_positions = [p for p in self.open_positions if p["position_id"] != pos["position_id"]]
            else:
                rem_qty = full_qty - close_qty
                rem_cost = float(pos["entry_cost"]) - alloc_entry_cost
                rem_fee = float(pos["entry_fee"]) - alloc_entry_fee
                pos["quantity"] = rem_qty
                pos["entry_cost"] = rem_cost
                pos["entry_fee"] = rem_fee
                pos["partial_taken"] = 1
                conn.execute(
                    """UPDATE paper_positions
                       SET quantity = ?, entry_cost = ?, entry_fee = ?, partial_taken = 1
                       WHERE position_id = ?""",
                    (rem_qty, rem_cost, rem_fee, pos["position_id"]),
                )

        mtm_equity = self.cash_balance + sum(float(p["quantity"]) * exit_price for p in self.open_positions)
        self.risk_manager.update_equity(mtm_equity, realized_pnl=net_pnl, now_utc=now_utc)
        self._persist_account()

        return {
            "position_id": pos["position_id"],
            "exit_reason": exit_reason,
            "net_pnl": round(net_pnl, 2),
            "return_pct": round(ret_pct, 2),
        }

    def step(
        self,
        df_1d: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        now_utc: Optional[datetime] = None,
        check_freshness: bool = True
    ) -> Dict[str, Any]:
        """Executes one complete evaluation cycle on validated closed candles."""
        now_dt = now_utc or datetime.now(timezone.utc)
        self.risk_manager.check_day_rollover(now_dt)

        # 1. Data quality validation
        v_1d = self.validator.validate_dataframe(df_1d, "1d", check_freshness=check_freshness, current_time=now_dt, drop_incomplete_candle=False)
        v_4h = self.validator.validate_dataframe(df_4h, "4h", check_freshness=check_freshness, current_time=now_dt, drop_incomplete_candle=False)
        v_1h = self.validator.validate_dataframe(df_1h, "1h", check_freshness=check_freshness, current_time=now_dt, drop_incomplete_candle=False)

        if not (v_1d.is_valid and v_4h.is_valid and v_1h.is_valid) or (check_freshness and v_1h.is_stale):
            self.risk_manager.set_data_feed_health(False, "Invalid or stale 1D/4H/1H market data")
            return {"status": "DATA_LOCKOUT", "action": "NO_TRADE"}

        self.risk_manager.set_data_feed_health(True)

        # 2. Indicators & Price Action
        c_1d = self.pa.compute_all_price_action(self.ie.compute_all_indicators(v_1d.cleaned_df))
        c_4h = self.pa.compute_all_price_action(self.ie.compute_all_indicators(v_4h.cleaned_df))
        c_1h = self.pa.compute_all_price_action(self.ie.compute_all_indicators(v_1h.cleaned_df))

        latest_bar = c_1h.iloc[-1]
        cur_price = float(latest_bar["close"])
        cur_high = float(latest_bar["high"])
        cur_low = float(latest_bar["low"])
        cur_atr = float(latest_bar.get("atr_14", cur_price * 0.015))

        closed_events = []

        # 3. Manage existing open positions
        for pos in list(self.open_positions):
            if cur_low <= float(pos["stop_loss"]):
                closed_events.append(self._close_position(pos, float(pos["stop_loss"]), "STOP_LOSS", now_dt))
                continue
            if cur_high >= float(pos["take_profit"]):
                closed_events.append(self._close_position(pos, float(pos["take_profit"]), "TAKE_PROFIT", now_dt))
                continue
            if not pos["partial_taken"] and cur_high >= float(pos["partial_tp_price"]):
                partial_qty = float(pos["quantity"]) * self.risk_manager.config.partial_tp_ratio
                closed_events.append(self._close_position(pos, float(pos["partial_tp_price"]), "PARTIAL_TP_1.5R", now_dt, qty_to_close=partial_qty))

        # 4. Evaluate StrategySelector & ML Filter
        has_open = len(self.open_positions) > 0
        sel_res = self.selector.select_and_evaluate(c_1d, c_4h, c_1h, candle_idx=-1, current_position_open=has_open)

        if has_open and sel_res.final_action == "EXIT":
            for pos in list(self.open_positions):
                closed_events.append(self._close_position(pos, cur_price, "MTF_EXIT", now_dt))

        opened_position = None
        ml_prob = None

        if not has_open and sel_res.final_action == "BUY" and sel_res.best_signal is not None:
            sig = sel_res.best_signal
            score_val = sel_res.best_score.total_score if sel_res.best_score else 0.0
            ml_ok, ml_prob, ml_msg = self.ml_filter.passes_filter(c_1h, candle_idx=-1)
            self.storage.save_signal_evaluation(sel_res.to_dict(), ml_prob=ml_prob)

            if ml_ok:
                sl, tp, partial_tp = self.risk_manager.calculate_atr_stops(
                    entry_price=cur_price,
                    atr=cur_atr,
                    swing_low=float(latest_bar.get("last_swing_low", np.nan)),
                    resistance=float(latest_bar.get("resistance_level", np.nan)),
                )
                sizing = self.risk_manager.evaluate_trade_risk(
                    entry_price=cur_price,
                    stop_loss_price=sl,
                    take_profit_price=tp,
                    available_balance=self.cash_balance,
                    open_positions_count=len(self.open_positions),
                    signal_key=f"paper-{latest_bar['date']}-{sig.strategy_name}",
                )
                if sizing.approved and sizing.quantity > 0:
                    slip_rate = self.risk_manager.config.slippage_bps / 10000.0
                    fee_rate = self.risk_manager.config.fee_rate
                    fill_price = cur_price * (1.0 + slip_rate)
                    notional = sizing.quantity * fill_price
                    entry_fee = notional * fee_rate
                    entry_cost = notional + entry_fee
                    if entry_cost <= self.cash_balance:
                        self.cash_balance -= entry_cost
                        pos_id = f"pos-{uuid.uuid4().hex[:10]}"
                        pos_record = {
                            "position_id": pos_id,
                            "strategy_name": sig.strategy_name,
                            "regime_at_entry": sel_res.active_regime.value,
                            "entry_time_utc": now_dt.isoformat(),
                            "entry_price": fill_price,
                            "quantity": sizing.quantity,
                            "entry_fee": entry_fee,
                            "entry_cost": entry_cost,
                            "initial_stop_loss": sl,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "partial_tp_price": partial_tp,
                            "partial_taken": 0,
                            "signal_score": score_val,
                            "ml_probability": ml_prob,
                            "entry_conditions_json": json.dumps(sig.reasons),
                        }
                        with self.storage._lock, self.storage._connect() as conn:
                            conn.execute(
                                """INSERT INTO paper_positions
                                   (position_id, strategy_name, regime_at_entry, entry_time_utc,
                                    entry_price, quantity, entry_fee, entry_cost, initial_stop_loss,
                                    stop_loss, take_profit, partial_tp_price, partial_taken,
                                    signal_score, ml_probability, entry_conditions_json)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
                                (
                                    pos_record["position_id"],
                                    pos_record["strategy_name"],
                                    pos_record["regime_at_entry"],
                                    pos_record["entry_time_utc"],
                                    pos_record["entry_price"],
                                    pos_record["quantity"],
                                    pos_record["entry_fee"],
                                    pos_record["entry_cost"],
                                    pos_record["initial_stop_loss"],
                                    pos_record["stop_loss"],
                                    pos_record["take_profit"],
                                    pos_record["partial_tp_price"],
                                    pos_record["signal_score"],
                                    pos_record["ml_probability"],
                                    pos_record["entry_conditions_json"],
                                ),
                            )
                        self.open_positions.append(pos_record)
                        opened_position = pos_record
                        self._persist_account()
        else:
            self.storage.save_signal_evaluation(sel_res.to_dict(), ml_prob=ml_prob)

        mtm_equity = self.cash_balance + sum(float(p["quantity"]) * cur_price for p in self.open_positions)
        self.risk_manager.update_equity(mtm_equity, realized_pnl=0.0, now_utc=now_dt)
        self._persist_account()

        return {
            "status": "OK",
            "action": sel_res.final_action,
            "equity": round(mtm_equity, 2),
            "cash_balance": round(self.cash_balance, 2),
            "open_positions": len(self.open_positions),
            "opened_position": opened_position,
            "closed_events": closed_events,
        }

"""
Independent Risk Management & Dynamic Position Sizing Engine for BTC Swing Trading.
Enforces portfolio-level risk controls that strategies cannot bypass:
- Spot-only (no leverage, no martingale, no averaging down)
- Dynamic position sizing based on equity risk % (0.5% - 1.0%), stop distance, fees, and slippage
- ATR-based & structural stop-loss and multi-tier take-profit calculation
- Emergency lockouts: Daily loss limit, Max drawdown limit, Data-feed failure, Manual kill switch
- Duplicate signal/order prevention
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Dict, List, Optional, Tuple, Any, Set
import math


@dataclass
class RiskConfig:
    """Configurable risk management parameters (Section 10 & 21)."""
    market: str = "BTC/USDT"
    leverage_enabled: bool = False
    martingale_enabled: bool = False
    averaging_down_enabled: bool = False
    risk_per_trade_pct: float = 0.01          # 1.0% default (conservative 0.5% - 1.0%)
    max_open_positions: int = 2               # 1 - 2 max open positions
    daily_loss_limit_pct: float = 0.03        # 3.0% daily loss lockout
    max_drawdown_limit_pct: float = 0.10      # 10.0% max drawdown lockout
    fee_rate: float = 0.001                   # 0.1% spot fee per side
    slippage_bps: float = 5.0                 # 5 bps (0.05%) estimated slippage per side
    atr_sl_multiplier: float = 2.0            # Stop Distance = ATR * multiplier
    atr_tp_multiplier: float = 3.5            # Take Profit = ATR * multiplier
    min_reward_risk_ratio: float = 1.5        # Minimum R:R required
    min_stop_distance_pct: float = 0.005      # 0.5% minimum stop distance sanity guard
    max_stop_distance_pct: float = 0.08       # 8.0% maximum stop distance sanity guard
    partial_tp_ratio: float = 0.5             # 50% partial exit at 1.5R


@dataclass
class RiskEvent:
    """Audit log entry for risk decisions and lockouts."""
    timestamp: str
    event_type: str
    severity: str       # 'INFO' | 'WARNING' | 'CRITICAL'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class PositionSizingResult:
    """Output of dynamic position sizing & risk validation."""
    approved: bool
    quantity: float
    notional_value: float
    risk_amount: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    partial_tp_price: float
    reward_risk_ratio: float
    estimated_fees: float
    estimated_slippage: float
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "quantity": round(self.quantity, 6),
            "notional_value": round(self.notional_value, 2),
            "risk_amount": round(self.risk_amount, 2),
            "entry_price": round(self.entry_price, 2),
            "stop_loss_price": round(self.stop_loss_price, 2),
            "take_profit_price": round(self.take_profit_price, 2),
            "partial_tp_price": round(self.partial_tp_price, 2),
            "reward_risk_ratio": round(self.reward_risk_ratio, 2),
            "estimated_fees": round(self.estimated_fees, 4),
            "estimated_slippage": round(self.estimated_slippage, 4),
            "rejection_reasons": self.rejection_reasons,
        }


class RiskManager:
    """
    Independent portfolio and trade-level risk manager.
    Strategies must pass through evaluate_trade_risk() before any order can be placed.
    """

    def __init__(self, initial_equity: float = 1000.0, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.initial_equity = float(initial_equity)
        self.current_equity = float(initial_equity)
        self.peak_equity = float(initial_equity)
        self.daily_start_equity = float(initial_equity)
        self.current_utc_date: str = datetime.now(timezone.utc).date().isoformat()
        self.daily_realized_pnl: float = 0.0

        # Lockout flags
        self.daily_loss_lockout: bool = False
        self.max_drawdown_lockout: bool = False
        self.data_feed_lockout: bool = False
        self.manual_kill_switch: bool = False

        # Tracking for duplicate order prevention
        self._processed_signal_keys: Set[str] = set()
        self.risk_events: List[RiskEvent] = []

    def _log_event(self, event_type: str, severity: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.risk_events.append(RiskEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            severity=severity,
            message=message,
            details=details or {}
        ))
        # Keep bounded in-memory log
        if len(self.risk_events) > 500:
            self.risk_events = self.risk_events[-500:]

    def check_day_rollover(self, now_utc: Optional[datetime] = None) -> None:
        """Resets daily PnL tracking at UTC midnight (daily loss lockout resets on new day)."""
        dt = now_utc or datetime.now(timezone.utc)
        today_str = dt.date().isoformat()
        if today_str != self.current_utc_date:
            self.current_utc_date = today_str
            self.daily_start_equity = self.current_equity
            self.daily_realized_pnl = 0.0
            if self.daily_loss_lockout:
                self.daily_loss_lockout = False
                self._log_event("DAILY_RESET", "INFO", "Daily loss lockout reset at UTC midnight")

    def update_equity(self, new_equity: float, realized_pnl: float = 0.0, now_utc: Optional[datetime] = None) -> None:
        """Updates equity, peak equity, daily PnL, and evaluates portfolio lockouts."""
        self.check_day_rollover(now_utc)
        self.current_equity = float(new_equity)
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        self.daily_realized_pnl += float(realized_pnl)

        # Check Daily Loss Limit
        daily_pnl_pct = (self.current_equity - self.daily_start_equity) / self.daily_start_equity if self.daily_start_equity > 0 else 0.0
        if daily_pnl_pct <= -self.config.daily_loss_limit_pct and not self.daily_loss_lockout:
            self.daily_loss_lockout = True
            self._log_event(
                "DAILY_LOSS_LOCKOUT",
                "CRITICAL",
                f"Daily loss limit breached ({daily_pnl_pct*100:.2f}% <= -{self.config.daily_loss_limit_pct*100:.2f}%)",
                {"daily_pnl_pct": round(daily_pnl_pct * 100, 2), "equity": round(self.current_equity, 2)}
            )

        # Check Max Drawdown Limit
        drawdown_pct = self.get_current_drawdown_pct() / 100.0
        if drawdown_pct >= self.config.max_drawdown_limit_pct and not self.max_drawdown_lockout:
            self.max_drawdown_lockout = True
            self._log_event(
                "MAX_DRAWDOWN_LOCKOUT",
                "CRITICAL",
                f"Maximum drawdown limit breached ({drawdown_pct*100:.2f}% >= {self.config.max_drawdown_limit_pct*100:.2f}%)",
                {"drawdown_pct": round(drawdown_pct * 100, 2), "peak_equity": round(self.peak_equity, 2)}
            )

    def get_current_drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        dd = (self.peak_equity - self.current_equity) / self.peak_equity * 100.0
        return max(0.0, dd)

    def set_data_feed_health(self, is_healthy: bool, reason: str = "") -> None:
        if not is_healthy and not self.data_feed_lockout:
            self.data_feed_lockout = True
            self._log_event("DATA_FEED_LOCKOUT", "CRITICAL", f"Data feed unhealthy: {reason}")
        elif is_healthy and self.data_feed_lockout:
            self.data_feed_lockout = False
            self._log_event("DATA_FEED_RECOVERED", "INFO", "Data feed health restored")

    def set_manual_kill_switch(self, active: bool, reason: str = "Manual operator toggle") -> None:
        self.manual_kill_switch = active
        self._log_event(
            "MANUAL_KILL_SWITCH",
            "CRITICAL" if active else "INFO",
            f"Manual kill switch {'ACTIVATED' if active else 'DEACTIVATED'}: {reason}"
        )

    def reset_max_drawdown_lockout(self) -> None:
        """Requires explicit manual approval to clear max drawdown lockout."""
        self.max_drawdown_lockout = False
        self.peak_equity = self.current_equity
        self._log_event("DRAWDOWN_RESET", "WARNING", "Max drawdown lockout manually reset")

    def get_active_lockouts(self) -> List[str]:
        reasons = []
        if self.manual_kill_switch:
            reasons.append("Manual emergency kill switch active")
        if self.daily_loss_lockout:
            reasons.append(f"Daily loss limit ({self.config.daily_loss_limit_pct*100:.1f}%) reached")
        if self.max_drawdown_lockout:
            reasons.append(f"Maximum drawdown limit ({self.config.max_drawdown_limit_pct*100:.1f}%) reached")
        if self.data_feed_lockout:
            reasons.append("Market data feed failure lockout active")
        return reasons

    def calculate_atr_stops(
        self,
        entry_price: float,
        atr: float,
        swing_low: Optional[float] = None,
        resistance: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Calculates volatility-adjusted Stop-Loss, Take-Profit, and Partial Take-Profit.
        Uses ATR distance and respects recent swing structure when valid.
        """
        if entry_price <= 0 or atr <= 0 or math.isnan(atr):
            atr = entry_price * 0.015

        atr_stop = entry_price - (atr * self.config.atr_sl_multiplier)
        stop_loss = atr_stop

        # If a meaningful recent swing low is slightly below ATR stop and within max stop distance, use it
        if swing_low is not None and not math.isnan(swing_low) and 0 < swing_low < entry_price:
            swing_dist_pct = (entry_price - swing_low) / entry_price
            if self.config.min_stop_distance_pct <= swing_dist_pct <= self.config.max_stop_distance_pct:
                # Choose the wider (safer) structural stop only if within bounds
                stop_loss = min(atr_stop, swing_low * 0.997)

        # Clamp stop distance within configured bounds
        stop_dist_pct = (entry_price - stop_loss) / entry_price
        if stop_dist_pct < self.config.min_stop_distance_pct:
            stop_loss = entry_price * (1.0 - self.config.min_stop_distance_pct)
        elif stop_dist_pct > self.config.max_stop_distance_pct:
            stop_loss = entry_price * (1.0 - self.config.max_stop_distance_pct)

        risk_per_unit = entry_price - stop_loss
        # Take profit: at least min_reward_risk_ratio * risk, or ATR target
        atr_tp = entry_price + (atr * self.config.atr_tp_multiplier)
        min_rr_tp = entry_price + (risk_per_unit * self.config.min_reward_risk_ratio)
        take_profit = max(atr_tp, min_rr_tp)

        # If nearby strong resistance is above 1.5R, respect it or keep ATR TP
        if resistance is not None and not math.isnan(resistance) and resistance > min_rr_tp:
            take_profit = min(take_profit, resistance * 0.995) if resistance > entry_price * 1.015 else take_profit

        partial_tp = entry_price + (risk_per_unit * 1.5)
        return stop_loss, take_profit, partial_tp

    def evaluate_trade_risk(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        available_balance: float,
        open_positions_count: int = 0,
        signal_key: Optional[str] = None
    ) -> PositionSizingResult:
        """
        Validates a trade setup against all independent risk rules and computes
        cost-adjusted position sizing.
        """
        rejection_reasons = self.get_active_lockouts()

        # Check max open positions
        if open_positions_count >= self.config.max_open_positions:
            rejection_reasons.append(
                f"Maximum open positions reached ({open_positions_count}/{self.config.max_open_positions})"
            )

        # Duplicate order prevention
        if signal_key and signal_key in self._processed_signal_keys:
            rejection_reasons.append(f"Duplicate signal prevented ({signal_key})")

        # Validate prices
        if entry_price <= 0 or stop_loss_price <= 0 or take_profit_price <= 0:
            rejection_reasons.append("Invalid non-positive entry, stop-loss, or take-profit price")
        elif stop_loss_price >= entry_price:
            rejection_reasons.append(f"Invalid stop-loss ({stop_loss_price:.2f}) >= entry price ({entry_price:.2f})")
        elif take_profit_price <= entry_price:
            rejection_reasons.append(f"Invalid take-profit ({take_profit_price:.2f}) <= entry price ({entry_price:.2f})")

        if rejection_reasons:
            self._log_event("TRADE_REJECTED", "WARNING", "; ".join(rejection_reasons), {"entry_price": entry_price})
            return PositionSizingResult(
                approved=False,
                quantity=0.0,
                notional_value=0.0,
                risk_amount=0.0,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                partial_tp_price=0.0,
                reward_risk_ratio=0.0,
                estimated_fees=0.0,
                estimated_slippage=0.0,
                rejection_reasons=rejection_reasons,
            )

        stop_distance = entry_price - stop_loss_price
        stop_dist_pct = stop_distance / entry_price
        if stop_dist_pct < self.config.min_stop_distance_pct:
            rejection_reasons.append(f"Stop distance ({stop_dist_pct*100:.2f}%) below minimum ({self.config.min_stop_distance_pct*100:.2f}%)")
        if stop_dist_pct > self.config.max_stop_distance_pct:
            rejection_reasons.append(f"Stop distance ({stop_dist_pct*100:.2f}%) exceeds maximum ({self.config.max_stop_distance_pct*100:.2f}%)")

        rr_ratio = (take_profit_price - entry_price) / stop_distance
        if rr_ratio < self.config.min_reward_risk_ratio:
            rejection_reasons.append(
                f"Reward/Risk ratio ({rr_ratio:.2f}) below minimum ({self.config.min_reward_risk_ratio:.2f})"
            )

        # Position Sizing Formula (Section 10) adjusted for fees and slippage
        risk_amount = self.current_equity * self.config.risk_per_trade_pct
        slippage_rate = self.config.slippage_bps / 10000.0
        # Total round-trip execution cost per BTC unit
        cost_per_unit = entry_price * (2.0 * self.config.fee_rate + 2.0 * slippage_rate)
        effective_risk_per_unit = stop_distance + cost_per_unit

        raw_quantity = risk_amount / effective_risk_per_unit if effective_risk_per_unit > 0 else 0.0

        # Spot constraint: cannot exceed available cash balance (no leverage)
        max_spot_notional = available_balance / (1.0 + self.config.fee_rate + slippage_rate)
        max_spot_quantity = max_spot_notional / entry_price if entry_price > 0 else 0.0

        quantity = min(raw_quantity, max_spot_quantity)
        notional_value = quantity * entry_price
        estimated_fees = notional_value * self.config.fee_rate * 2.0
        estimated_slippage = notional_value * slippage_rate * 2.0

        if quantity <= 0 or notional_value < 5.0:
            rejection_reasons.append(f"Insufficient available balance (${available_balance:.2f}) or zero position size")

        if rejection_reasons:
            self._log_event("TRADE_REJECTED", "WARNING", "; ".join(rejection_reasons), {"entry_price": entry_price})
            return PositionSizingResult(
                approved=False,
                quantity=0.0,
                notional_value=0.0,
                risk_amount=risk_amount,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                partial_tp_price=0.0,
                reward_risk_ratio=rr_ratio,
                estimated_fees=0.0,
                estimated_slippage=0.0,
                rejection_reasons=rejection_reasons,
            )

        if signal_key:
            self._processed_signal_keys.add(signal_key)

        partial_tp = entry_price + (stop_distance * 1.5)
        actual_risk = quantity * effective_risk_per_unit

        return PositionSizingResult(
            approved=True,
            quantity=quantity,
            notional_value=notional_value,
            risk_amount=actual_risk,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            partial_tp_price=partial_tp,
            reward_risk_ratio=rr_ratio,
            estimated_fees=estimated_fees,
            estimated_slippage=estimated_slippage,
            rejection_reasons=[],
        )

    def get_status(self, open_exposure: float = 0.0) -> Dict[str, Any]:
        """Returns complete risk monitor telemetry for dashboard and APIs."""
        daily_pnl = self.current_equity - self.daily_start_equity
        daily_pnl_pct = (daily_pnl / self.daily_start_equity * 100.0) if self.daily_start_equity > 0 else 0.0
        return {
            "market": self.config.market,
            "leverage_enabled": self.config.leverage_enabled,
            "risk_per_trade_pct": round(self.config.risk_per_trade_pct * 100.0, 2),
            "max_open_positions": self.config.max_open_positions,
            "current_equity": round(self.current_equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "daily_loss_limit_pct": round(self.config.daily_loss_limit_pct * 100.0, 2),
            "current_drawdown_pct": round(self.get_current_drawdown_pct(), 2),
            "max_drawdown_limit_pct": round(self.config.max_drawdown_limit_pct * 100.0, 2),
            "open_exposure": round(open_exposure, 2),
            "daily_loss_lockout": self.daily_loss_lockout,
            "max_drawdown_lockout": self.max_drawdown_lockout,
            "data_feed_lockout": self.data_feed_lockout,
            "manual_kill_switch": self.manual_kill_switch,
            "active_lockouts": self.get_active_lockouts(),
            "recent_events": [e.to_dict() for e in self.risk_events[-10:]],
        }

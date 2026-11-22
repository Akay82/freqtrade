"""Shared position sizing and loss protections for the BTC strategy fleet."""

from datetime import UTC, timedelta
import json
from pathlib import Path

from freqtrade.exchange import timeframe_to_seconds
from freqtrade.persistence import Trade


class RiskManagementMixin:
    """Limit position size to a fixed fraction of wallet risk per trade."""

    risk_per_trade = 0.005
    round_trip_cost_buffer = 0.003
    daily_loss_limit = 0.02
    max_test_drawdown = 0.10
    volatility_move_trigger_pct = 3.0
    volatility_range_trigger_pct = 4.0
    volatility_recovery_move_pct = 1.5
    volatility_recovery_range_pct = 2.5
    volatility_recovery_candles = 2

    @staticmethod
    def _fast_volatility_pause_active() -> bool:
        state_path = Path(__file__).resolve().parents[1] / "records" / "30_day_test.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        return bool(state.get("fast_volatility_paused") or state.get("fast_volatility_data_paused"))

    def _volatility_guard_active(self, pair, current_time) -> bool:
        """Fail closed on entries until two calm closes follow the latest spike."""
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return True

            now = current_time
            if now.tzinfo is None:
                now = now.replace(tzinfo=UTC)
            candle_duration = timedelta(seconds=timeframe_to_seconds(self.timeframe))
            candles = []
            for row in dataframe.tail(32).itertuples(index=False):
                candle_open = getattr(row, "date")
                if hasattr(candle_open, "to_pydatetime"):
                    candle_open = candle_open.to_pydatetime()
                if candle_open.tzinfo is None:
                    candle_open = candle_open.replace(tzinfo=UTC)
                if candle_open + candle_duration > now:
                    continue
                open_price = float(getattr(row, "open"))
                if open_price <= 0:
                    continue
                close_price = float(getattr(row, "close"))
                high_price = float(getattr(row, "high"))
                low_price = float(getattr(row, "low"))
                candles.append({
                    "move_pct": (close_price / open_price - 1) * 100,
                    "range_pct": (high_price - low_price) / open_price * 100,
                })
        except Exception:
            # Do not open a trade if volatility cannot be checked reliably.
            return True

        if not candles:
            return True

        def is_trigger(candle) -> bool:
            return (
                abs(candle["move_pct"]) >= self.volatility_move_trigger_pct
                or candle["range_pct"] >= self.volatility_range_trigger_pct
            )

        def is_calm(candle) -> bool:
            return (
                abs(candle["move_pct"]) <= self.volatility_recovery_move_pct
                and candle["range_pct"] <= self.volatility_recovery_range_pct
            )

        trigger_indexes = [index for index, candle in enumerate(candles) if is_trigger(candle)]
        if not trigger_indexes:
            return False
        candles_after_trigger = candles[trigger_indexes[-1] + 1:]
        required = self.volatility_recovery_candles
        return not (
            len(candles_after_trigger) >= required
            and all(is_calm(candle) for candle in candles_after_trigger[-required:])
        )

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 2},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 3,
                "stop_duration_candles": 24,
                "required_profit": 0.0,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "calculation_mode": "equity",
                "lookback_period_candles": 24,
                "trade_limit": 1,
                "max_allowed_drawdown": self.daily_loss_limit,
                "stop_duration_candles": 24,
            },
            {
                "method": "MaxDrawdown",
                "calculation_mode": "equity",
                "lookback_period_candles": 720,
                "trade_limit": 1,
                "max_allowed_drawdown": self.max_test_drawdown,
                "stop_duration_candles": 720,
            },
        ]

    def custom_stake_amount(
        self,
        pair,
        current_time,
        current_rate,
        proposed_stake,
        min_stake,
        max_stake,
        leverage,
        entry_tag,
        side,
        **kwargs,
    ) -> float:
        """Size a trade so its configured stop risks at most 0.5% of equity."""
        starting_equity = float(
            self.config.get("dry_run_wallet")
            or self.config.get("available_capital")
            or self.wallets.get_total_stake_amount()
        )
        if starting_equity <= 0:
            return 0.0

        if self._fast_volatility_pause_active():
            return 0.0

        if self._volatility_guard_active(pair, current_time):
            return 0.0

        # Avoid opening a new trade after the rolling daily loss cap is reached.
        day_start = current_time.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            closed_trades = Trade.get_trades_proxy(is_open=False)
        except Exception:
            # Failing closed is safer than falling back to the bot's unlimited proposed stake.
            return 0.0
        daily_profit = sum(
            trade.close_profit_abs or 0.0
            for trade in closed_trades
            if trade.close_date_utc and trade.close_date_utc.replace(tzinfo=UTC) >= day_start
        )
        if daily_profit <= -(starting_equity * self.daily_loss_limit):
            return 0.0

        # Also keep the bot from re-entering after a 10% peak-to-trough test drawdown.
        equity = peak_equity = starting_equity
        for trade in sorted(closed_trades, key=lambda item: item.close_date_utc or current_time):
            equity += trade.close_profit_abs or 0.0
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0 and (peak_equity - equity) / peak_equity >= self.max_test_drawdown:
                return 0.0

        stop_distance = abs(float(self.stoploss)) * max(float(leverage), 1.0)
        estimated_cost = self.round_trip_cost_buffer * max(float(leverage), 1.0)
        current_equity = float(self.wallets.get_total_stake_amount())
        risk_amount = current_equity * self.risk_per_trade
        stake = min(max_stake, risk_amount / (stop_distance + estimated_cost))

        # Never raise the stake to satisfy exchange minimums; skip if minimum size breaches risk.
        if min_stake is not None and stake < min_stake:
            return 0.0
        return max(0.0, stake)

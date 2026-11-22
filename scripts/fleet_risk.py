"""Persistent 30-day test state and portfolio-wide loss guard."""

import base64
import json
import os
import sqlite3
import tempfile
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "user_data" / "configs"
DB_DIR = ROOT / "user_data" / "db"
STATE_PATH = ROOT / "user_data" / "records" / "30_day_test.json"
ALERT_PATH = ROOT / "user_data" / "logs" / "fleet_alerts.log"
BACKUP_DIR = ROOT / "user_data" / "records" / "db_backups"
STARTING_EQUITY = 1000.0
DAILY_LOSS_LIMIT = 0.02
MAX_DRAWDOWN_LIMIT = 0.10
TEST_LENGTH = timedelta(days=30)
VOLATILITY_MOVE_TRIGGER_PCT = 3.0
VOLATILITY_RANGE_TRIGGER_PCT = 4.0
VOLATILITY_RECOVERY_MOVE_PCT = 1.5
VOLATILITY_RECOVERY_RANGE_PCT = 2.5
VOLATILITY_RECOVERY_CANDLES = 2
FAST_VOLATILITY_MOVE_TRIGGER_PCT = 1.5
FAST_VOLATILITY_RANGE_TRIGGER_PCT = 2.0
FAST_VOLATILITY_RECOVERY_MOVE_PCT = 0.4
FAST_VOLATILITY_RECOVERY_RANGE_PCT = 0.7
FAST_VOLATILITY_RECOVERY_CANDLES = 6
FAST_VOLATILITY_FAILURE_LIMIT = 3
_STATE_LOCK = threading.RLock()


def _write_state(state: dict) -> None:
    """Atomically replace the state file so readers never see partial JSON."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _STATE_LOCK:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=STATE_PATH.parent,
                prefix=f"{STATE_PATH.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                json.dump(state, temp_file, indent=2)
                temp_file.write("\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, STATE_PATH)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()


def read_state() -> dict:
    with _STATE_LOCK:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))

    now = datetime.now(UTC)
    state = {
        "started_at_utc": now.isoformat(),
        "ends_at_utc": (now + TEST_LENGTH).isoformat(),
        "starting_equity": STARTING_EQUITY,
        "daily_loss_limit_pct": DAILY_LOSS_LIMIT,
        "max_drawdown_limit_pct": MAX_DRAWDOWN_LIMIT,
        "guard_triggered": False,
        "guard_reason": None,
        "guard_triggered_at_utc": None,
        "daily_pause_until_utc": None,
        "daily_pause_reason": None,
        "market_data_paused": False,
        "volatility_paused": False,
        "volatility_recovery_count": 0,
        "volatility_last_evaluated_candle_utc": None,
        "fast_volatility_paused": False,
        "fast_volatility_recovery_count": 0,
        "fast_volatility_last_evaluated_candle_utc": None,
        "fast_volatility_failure_count": 0,
        "volatility_events": [],
    }
    _write_state(state)
    return state


def _read_closed_trades() -> list[tuple[datetime, float]]:
    trades: list[tuple[datetime, float]] = []
    for db_path in sorted(DB_DIR.glob("btc_*.sqlite")):
        try:
            with sqlite3.connect(db_path, timeout=2) as conn:
                rows = conn.execute(
                    "SELECT close_date, close_profit_abs FROM trades "
                    "WHERE is_open = 0 AND close_date IS NOT NULL"
                ).fetchall()
            for date_text, profit in rows:
                date = datetime.fromisoformat(str(date_text).replace("Z", "+00:00"))
                if date.tzinfo is None:
                    date = date.replace(tzinfo=UTC)
                trades.append((date.astimezone(UTC), float(profit or 0.0)))
        except (sqlite3.Error, ValueError):
            continue
    return sorted(trades, key=lambda item: item[0])


def _get_open_profit(port: int, config: dict) -> float:
    credentials = f"{config['api_server']['username']}:{config['api_server']['password']}"
    token = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
    request = Request(
        f"http://127.0.0.1:{port}/api/v1/status",
        headers={"Authorization": f"Basic {token}"},
    )
    with urlopen(request, timeout=2) as response:
        open_trades = json.load(response)
    return sum(float(trade.get("profit_abs") or 0.0) for trade in open_trades)


def _get_market_data_status(port: int, config: dict) -> dict:
    credentials = f"{config['api_server']['username']}:{config['api_server']['password']}"
    token = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
    query = urlencode({"pair": "BTC/USDT", "timeframe": "1h", "limit": 32})
    request = Request(
        f"http://127.0.0.1:{port}/api/v1/pair_candles?{query}",
        headers={"Authorization": f"Basic {token}"},
    )
    with urlopen(request, timeout=3) as response:
        candle_data = json.load(response)
    columns = candle_data.get("columns", [])
    rows = candle_data.get("data", [])
    if not rows or not all(name in columns for name in ("date", "open", "high", "low", "close")):
        raise ValueError("Bot has not published analyzed BTC/USDT candles yet")
    now = datetime.now(UTC)
    indexes = {name: columns.index(name) for name in ("date", "open", "high", "low", "close")}
    completed = []
    for row in rows:
        candle_open = datetime.fromisoformat(str(row[indexes["date"]]).replace("Z", "+00:00"))
        if candle_open.tzinfo is None:
            candle_open = candle_open.replace(tzinfo=UTC)
        candle_close = candle_open.astimezone(UTC) + timedelta(hours=1)
        if candle_close > now:
            continue
        open_price = float(row[indexes["open"]])
        high_price = float(row[indexes["high"]])
        low_price = float(row[indexes["low"]])
        close_price = float(row[indexes["close"]])
        if open_price <= 0:
            continue
        completed.append({
            "open_utc": candle_open.astimezone(UTC).isoformat(),
            "close": close_price,
            "move_pct": round((close_price / open_price - 1) * 100, 4),
            "range_pct": round((high_price - low_price) / open_price * 100, 4),
        })
    if not completed:
        raise ValueError("No completed BTC/USDT 1h candles are available")
    recent_candles = completed[-30:]
    last_candle = recent_candles[-1]
    candle_close = datetime.fromisoformat(last_candle["open_utc"]) + timedelta(hours=1)
    age_seconds = (now - candle_close).total_seconds()
    return {
        "exchange": config["exchange"]["name"],
        "pair": "BTC/USDT",
        "timeframe": "1h",
        "last_candle_open_utc": last_candle["open_utc"],
        "last_close": last_candle["close"],
        "age_after_close_minutes": round(age_seconds / 60, 2),
        "stale": age_seconds > 4200,
        "recent_completed_candles": recent_candles,
    }


def fetch_fast_market_data() -> dict:
    """Fetch public 5m BTC klines, including the live partial candle."""
    query = urlencode({"symbol": "BTCUSDT", "interval": "5m", "limit": 10})
    request = Request(
        f"https://data-api.binance.vision/api/v3/klines?{query}",
        headers={"User-Agent": "BTCStrategyFleet/1.0"},
    )
    with urlopen(request, timeout=4) as response:
        rows = json.load(response)
    now = datetime.now(UTC)
    now_ms = int(now.timestamp() * 1000)
    completed = []
    current = None
    for row in rows:
        candle = {
            "open_utc": datetime.fromtimestamp(int(row[0]) / 1000, UTC).isoformat(),
            "open_ms": int(row[0]),
            "close_ms": int(row[6]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "price": float(row[4]),
            "move_pct": round((float(row[4]) / float(row[1]) - 1) * 100, 4),
            "range_pct": round((float(row[2]) - float(row[3])) / float(row[1]) * 100, 4),
        }
        if candle["open"] <= 0:
            continue
        if candle["close_ms"] < now_ms:
            completed.append(candle)
        else:
            current = candle
    if current is None:
        raise ValueError("Binance did not return the current BTCUSDT 5m candle")
    age_seconds = max(0, (now_ms - current["open_ms"]) // 1000)
    return {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "interval": "5m",
        "as_of_utc": now.isoformat(),
        "price": current["price"],
        "current_candle": current,
        "recent_completed_candles": completed[-8:],
        "age_seconds": age_seconds,
        "stale": age_seconds > 600,
    }


def _new_volatility_event(state: dict, tier: str, candle: dict, now: datetime) -> dict:
    events = state.setdefault("volatility_events", [])
    event = {
        "event_id": len(events) + 1,
        "tier": tier,
        "triggered_at_utc": now.isoformat(),
        "trigger_candle_open_utc": candle["open_utc"],
        "trigger_price": float(candle.get("price", candle.get("close", 0.0))),
        "move_pct": float(candle["move_pct"]),
        "range_pct": float(candle["range_pct"]),
        "recovered_at_utc": None,
        "recovery_price": None,
        "btc_return_to_recovery_pct": None,
        "btc_return_after_15m_pct": None,
        "btc_return_after_30m_pct": None,
        "btc_return_after_60m_pct": None,
    }
    events.append(event)
    del events[:-500]
    return event


def _latest_open_volatility_event(state: dict, tier: str) -> dict | None:
    return next(
        (event for event in reversed(state.get("volatility_events", []))
         if event.get("tier") == tier and not event.get("recovered_at_utc")),
        None,
    )


def _extend_volatility_event(state: dict, tier: str, candle: dict, now: datetime) -> None:
    event = _latest_open_volatility_event(state, tier)
    if event is None:
        _new_volatility_event(state, tier, candle, now)
        return
    event["last_trigger_at_utc"] = now.isoformat()
    event["last_trigger_candle_open_utc"] = candle["open_utc"]
    event["move_pct"] = float(candle["move_pct"])
    event["range_pct"] = float(candle["range_pct"])
    event["trigger_price"] = float(candle.get("price", candle.get("close", 0.0)))


def _close_volatility_event(state: dict, tier: str, price: float, now: datetime) -> None:
    event = _latest_open_volatility_event(state, tier)
    if event is None:
        return
    event["recovered_at_utc"] = now.isoformat()
    event["recovery_price"] = round(float(price), 8)
    if event.get("trigger_price"):
        event["btc_return_to_recovery_pct"] = round(
            (float(price) / float(event["trigger_price"]) - 1) * 100, 4
        )


def _update_volatility_forward_returns(state: dict, price: float, now: datetime) -> bool:
    changed = False
    for event in state.get("volatility_events", []):
        trigger_price = float(event.get("trigger_price") or 0.0)
        if trigger_price <= 0:
            continue
        triggered_at = datetime.fromisoformat(event["triggered_at_utc"].replace("Z", "+00:00"))
        if triggered_at.tzinfo is None:
            triggered_at = triggered_at.replace(tzinfo=UTC)
        elapsed = (now - triggered_at.astimezone(UTC)).total_seconds()
        for minutes in (15, 30, 60):
            key = f"btc_return_after_{minutes}m_pct"
            if elapsed >= minutes * 60 and event.get(key) is None:
                event[key] = round((price / trigger_price - 1) * 100, 4)
                changed = True
    return changed


def update_fast_volatility_guard(market_data: dict | None) -> dict:
    """Latch a fleet entry pause on a fast BTC move, then require 30m of calm."""
    state = read_state()
    now = datetime.now(UTC)
    if not market_data or market_data.get("stale"):
        state["fast_volatility_failure_count"] = int(state.get("fast_volatility_failure_count", 0)) + 1
        state["fast_volatility_data_recovery_count"] = 0
        if (
            state["fast_volatility_failure_count"] >= FAST_VOLATILITY_FAILURE_LIMIT
            and not state.get("fast_volatility_data_paused")
        ):
            state["fast_volatility_data_paused"] = True
            state["fast_volatility_data_pause_reason"] = "Binance 5m BTC data unavailable or stale"
            write_alert("FAST DATA PAUSE: Binance 5m BTC data unavailable or stale for three checks.")
        _write_state(state)
        return state

    changed = bool(state.get("fast_volatility_failure_count"))
    state["fast_volatility_failure_count"] = 0
    if state.get("fast_volatility_data_paused"):
        state["fast_volatility_data_recovery_count"] = int(
            state.get("fast_volatility_data_recovery_count", 0)
        ) + 1
        changed = True
        if state["fast_volatility_data_recovery_count"] >= 3:
            state["fast_volatility_data_paused"] = False
            state["fast_volatility_data_pause_reason"] = None
            state["fast_volatility_data_recovery_count"] = 0
            write_alert("FAST DATA RECOVERY: Binance 5m BTC feed is healthy again.")

    enabled_raw = state.get("fast_volatility_enabled_at_utc")
    enabled_at = datetime.fromisoformat(enabled_raw.replace("Z", "+00:00")) if enabled_raw else now
    if enabled_at.tzinfo is None:
        enabled_at = enabled_at.replace(tzinfo=UTC)
    enabled_at = enabled_at.astimezone(UTC)
    if state.pop("fast_volatility_baseline_pending", False):
        prior_candles = [
            candle for candle in market_data.get("recent_completed_candles", [])
            if datetime.fromisoformat(candle["open_utc"]).astimezone(UTC) + timedelta(minutes=5)
            <= enabled_at
        ]
        state["fast_volatility_last_evaluated_candle_utc"] = (
            prior_candles[-1]["open_utc"] if prior_candles else None
        )
        changed = True

    current = market_data["current_candle"]
    current_open_time = datetime.fromisoformat(current["open_utc"]).astimezone(UTC)
    trigger_current = (
        abs(float(current["move_pct"])) >= FAST_VOLATILITY_MOVE_TRIGGER_PCT
        or float(current["range_pct"]) >= FAST_VOLATILITY_RANGE_TRIGGER_PCT
    ) and current_open_time >= enabled_at
    current_key = current["open_utc"]
    if trigger_current and current_key != state.get("fast_volatility_last_trigger_candle_utc"):
        already_paused = bool(state.get("fast_volatility_paused"))
        state["fast_volatility_paused"] = True
        state["fast_volatility_pause_reason"] = (
            f"BTC 5m move {current['move_pct']:.2f}% / range {current['range_pct']:.2f}% "
            f"crossed the {FAST_VOLATILITY_MOVE_TRIGGER_PCT:.1f}% / "
            f"{FAST_VOLATILITY_RANGE_TRIGGER_PCT:.1f}% fast-volatility trigger"
        )
        state["fast_volatility_trigger_candle_utc"] = current_key
        state["fast_volatility_last_trigger_candle_utc"] = current_key
        state["fast_volatility_recovery_count"] = 0
        state["fast_volatility_last_evaluated_candle_utc"] = current_key
        if already_paused:
            _extend_volatility_event(state, "5m", current, now)
        else:
            _new_volatility_event(state, "5m", current, now)
        write_alert(
            ("FAST VOLATILITY PAUSE extended: " if already_paused else "FAST VOLATILITY PAUSE: ")
            + state["fast_volatility_pause_reason"]
        )
        changed = True

    last_evaluated = state.get("fast_volatility_last_evaluated_candle_utc") or ""
    for candle in market_data.get("recent_completed_candles", []):
        candle_key = candle["open_utc"]
        if candle_key <= last_evaluated:
            continue
        candle_open_time = datetime.fromisoformat(candle_key).astimezone(UTC)
        if candle_open_time + timedelta(minutes=5) <= enabled_at:
            state["fast_volatility_last_evaluated_candle_utc"] = candle_key
            last_evaluated = candle_key
            changed = True
            continue
        trigger = (
            abs(float(candle["move_pct"])) >= FAST_VOLATILITY_MOVE_TRIGGER_PCT
            or float(candle["range_pct"]) >= FAST_VOLATILITY_RANGE_TRIGGER_PCT
        )
        if not state.get("fast_volatility_paused") and trigger:
            state["fast_volatility_paused"] = True
            state["fast_volatility_pause_reason"] = (
                f"BTC 5m move {candle['move_pct']:.2f}% / range {candle['range_pct']:.2f}% "
                f"crossed the {FAST_VOLATILITY_MOVE_TRIGGER_PCT:.1f}% / "
                f"{FAST_VOLATILITY_RANGE_TRIGGER_PCT:.1f}% fast-volatility trigger"
            )
            state["fast_volatility_trigger_candle_utc"] = candle_key
            state["fast_volatility_last_trigger_candle_utc"] = candle_key
            state["fast_volatility_recovery_count"] = 0
            _new_volatility_event(state, "5m", candle, now)
            write_alert(f"FAST VOLATILITY PAUSE: {state['fast_volatility_pause_reason']}")
            changed = True
        elif state.get("fast_volatility_paused") and candle_key > (
            state.get("fast_volatility_trigger_candle_utc") or ""
        ):
            calm = (
                abs(float(candle["move_pct"])) <= FAST_VOLATILITY_RECOVERY_MOVE_PCT
                and float(candle["range_pct"]) <= FAST_VOLATILITY_RECOVERY_RANGE_PCT
            )
            if trigger:
                state["fast_volatility_trigger_candle_utc"] = candle_key
                state["fast_volatility_last_trigger_candle_utc"] = candle_key
                state["fast_volatility_recovery_count"] = 0
                state["fast_volatility_pause_reason"] = (
                    f"BTC 5m move {candle['move_pct']:.2f}% / range {candle['range_pct']:.2f}% "
                    "crossed the fast-volatility trigger again"
                )
                _extend_volatility_event(state, "5m", candle, now)
                write_alert(f"FAST VOLATILITY PAUSE extended: {state['fast_volatility_pause_reason']}")
                changed = True
            else:
                state["fast_volatility_recovery_count"] = (
                    int(state.get("fast_volatility_recovery_count", 0)) + 1 if calm else 0
                )
                if state["fast_volatility_recovery_count"] >= FAST_VOLATILITY_RECOVERY_CANDLES:
                    state["fast_volatility_paused"] = False
                    state["fast_volatility_pause_reason"] = None
                    state["fast_volatility_recovered_at_utc"] = now.isoformat()
                    state["fast_volatility_recovery_count"] = 0
                    _close_volatility_event(state, "5m", float(candle["price"]), now)
                    write_alert(
                        "FAST VOLATILITY RECOVERY: six consecutive BTC 5m candles met calm limits; "
                        "new entries may resume."
                    )
                changed = True
        if state.get("fast_volatility_last_evaluated_candle_utc") != candle_key:
            state["fast_volatility_last_evaluated_candle_utc"] = candle_key
            changed = True
        last_evaluated = candle_key

    if _update_volatility_forward_returns(state, float(market_data["price"]), now):
        changed = True
    if changed:
        _write_state(state)
    return state


def update_volatility_pause(market_data: dict) -> dict:
    """Pause entries after an outsized closed hourly candle until two calm closes."""
    candles = market_data.get("recent_completed_candles") or []
    if not candles:
        return read_state()

    state = read_state()

    def candle_time(candle_key: str) -> datetime:
        parsed = datetime.fromisoformat(candle_key.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)

    def is_trigger(candle: dict) -> bool:
        return (
            abs(float(candle["move_pct"])) >= VOLATILITY_MOVE_TRIGGER_PCT
            or float(candle["range_pct"]) >= VOLATILITY_RANGE_TRIGGER_PCT
        )

    def is_calm(candle: dict) -> bool:
        return (
            abs(float(candle["move_pct"])) <= VOLATILITY_RECOVERY_MOVE_PCT
            and float(candle["range_pct"]) <= VOLATILITY_RECOVERY_RANGE_PCT
        )

    enabled_at = datetime.fromisoformat(
        state.get("volatility_enabled_at_utc", state["started_at_utc"]).replace("Z", "+00:00")
    )
    if enabled_at.tzinfo is None:
        enabled_at = enabled_at.replace(tzinfo=UTC)
    baseline_pending = state.pop("volatility_baseline_pending", False)
    changed = bool(baseline_pending)
    if baseline_pending:
        prior_candles = [
            candle for candle in candles
            if candle_time(candle["open_utc"]) + timedelta(hours=1) <= enabled_at.astimezone(UTC)
        ]
        state["volatility_last_evaluated_candle_utc"] = (
            prior_candles[-1]["open_utc"] if prior_candles else None
        )

    last_evaluated_key = state.get("volatility_last_evaluated_candle_utc")
    last_evaluated_time = candle_time(last_evaluated_key) if last_evaluated_key else None
    new_candles = [
        candle for candle in candles
        if candle_time(candle["open_utc"]) + timedelta(hours=1) > enabled_at.astimezone(UTC)
        and (last_evaluated_time is None or candle_time(candle["open_utc"]) > last_evaluated_time)
    ]
    for candle in new_candles:
        candle_key = candle["open_utc"]
        if not state.get("volatility_paused") and is_trigger(candle):
            state["volatility_paused"] = True
            state["volatility_pause_reason"] = (
                f"BTC 1h move {candle['move_pct']:.2f}% / range {candle['range_pct']:.2f}% "
                f"crossed the {VOLATILITY_MOVE_TRIGGER_PCT:.1f}% / "
                f"{VOLATILITY_RANGE_TRIGGER_PCT:.1f}% volatility trigger"
            )
            state["volatility_paused_at_utc"] = datetime.now(UTC).isoformat()
            state["volatility_trigger_candle_utc"] = candle_key
            state["volatility_recovery_count"] = 0
            write_alert(f"VOLATILITY PAUSE: {state['volatility_pause_reason']}")
            changed = True
        elif state.get("volatility_paused") and is_trigger(candle):
            state["volatility_trigger_candle_utc"] = candle_key
            state["volatility_recovery_count"] = 0
            state["volatility_pause_reason"] = (
                f"BTC 1h move {candle['move_pct']:.2f}% / range {candle['range_pct']:.2f}% "
                "crossed the volatility trigger again"
            )
            write_alert(f"VOLATILITY PAUSE extended: {state['volatility_pause_reason']}")
            changed = True
        elif state.get("volatility_paused"):
            state["volatility_recovery_count"] = (
                int(state.get("volatility_recovery_count", 0)) + 1 if is_calm(candle) else 0
            )
            changed = True
            if state["volatility_recovery_count"] >= VOLATILITY_RECOVERY_CANDLES:
                state["volatility_paused"] = False
                state["volatility_pause_reason"] = None
                state["volatility_recovered_at_utc"] = datetime.now(UTC).isoformat()
                state["volatility_recovery_count"] = 0
                write_alert(
                    "VOLATILITY RECOVERY: two consecutive BTC 1h candles met the "
                    f"calm limits (move <= {VOLATILITY_RECOVERY_MOVE_PCT:.1f}%, "
                    f"range <= {VOLATILITY_RECOVERY_RANGE_PCT:.1f}%); entries may resume."
                )
        state["volatility_last_evaluated_candle_utc"] = candle_key
        changed = True

    if changed:
        _write_state(state)
    return state


def record_volatility_rule_activation() -> dict:
    """Record the circuit-breaker rule version without resetting the test window."""
    state = read_state()
    rule = {
        "trigger_move_pct": VOLATILITY_MOVE_TRIGGER_PCT,
        "trigger_range_pct": VOLATILITY_RANGE_TRIGGER_PCT,
        "recovery_move_pct": VOLATILITY_RECOVERY_MOVE_PCT,
        "recovery_range_pct": VOLATILITY_RECOVERY_RANGE_PCT,
        "recovery_candles": VOLATILITY_RECOVERY_CANDLES,
    }
    fast_rule = {
        "interval": "5m",
        "trigger_move_pct": FAST_VOLATILITY_MOVE_TRIGGER_PCT,
        "trigger_range_pct": FAST_VOLATILITY_RANGE_TRIGGER_PCT,
        "recovery_move_pct": FAST_VOLATILITY_RECOVERY_MOVE_PCT,
        "recovery_range_pct": FAST_VOLATILITY_RECOVERY_RANGE_PCT,
        "recovery_candles": FAST_VOLATILITY_RECOVERY_CANDLES,
    }
    if state.get("fast_volatility_circuit_breaker") != fast_rule:
        enabled_at = datetime.now(UTC).isoformat()
        state["fast_volatility_circuit_breaker"] = fast_rule
        state["fast_volatility_enabled_at_utc"] = enabled_at
        state["fast_volatility_baseline_pending"] = True
        state["fast_volatility_last_evaluated_candle_utc"] = None
        changes = state.setdefault("rule_changes", [])
        changes.append({
            "at_utc": enabled_at,
            "rule": "5m_fast_volatility_entry_pause",
            **fast_rule,
        })
        _write_state(state)
        write_alert(
            "RULE CHANGE effective now: 5m fast-volatility entry pause enabled; "
            "trigger at move >= 1.5% or range >= 2.0%, recover after six calm candles."
        )
    rule_changed = state.get("volatility_circuit_breaker") != rule
    if rule_changed:
        enabled_at = datetime.now(UTC).isoformat()
        state["volatility_circuit_breaker"] = rule
        changes = state.setdefault("rule_changes", [])
        change = {"at_utc": datetime.now(UTC).isoformat(), "rule": "hourly_volatility_entry_pause", **rule}
        changes.append(change)
        write_alert(
            "RULE CHANGE effective now: hourly volatility entry pause enabled; "
            "trigger at absolute candle move >= 3.0% or range >= 4.0%, "
            "recover after two calm completed candles. Test start/end timestamps preserved."
        )
    elif not state.get("volatility_enabled_at_utc"):
        changes = state.get("rule_changes", [])
        previous_change = next(
            (item for item in reversed(changes) if item.get("rule") == "hourly_volatility_entry_pause"),
            None,
        )
        enabled_at = (previous_change or {}).get("at_utc") or datetime.now(UTC).isoformat()
        write_alert(
            "Volatility guard baseline correction: initial startup alerts were generated "
            "from candles closed before activation; no volatility pause was applied."
        )
    else:
        return state

    state["volatility_enabled_at_utc"] = enabled_at
    state["volatility_baseline_pending"] = True
    state["volatility_last_evaluated_candle_utc"] = None
    _write_state(state)
    return state


def portfolio_snapshot() -> dict:
    """Calculate realized and marked-to-market fleet loss from bot APIs and DBs."""
    now = datetime.now(UTC)
    state = read_state()
    trades = _read_closed_trades()
    test_start = datetime.fromisoformat(state["started_at_utc"])
    if test_start.tzinfo is None:
        test_start = test_start.replace(tzinfo=UTC)
    trades = [(closed_at, profit) for closed_at, profit in trades if closed_at >= test_start]
    realized_profit = sum(profit for _, profit in trades)
    daily_profit = sum(
        profit
        for closed_at, profit in trades
        if closed_at >= now.replace(hour=0, minute=0, second=0, microsecond=0)
    )

    open_profit = 0.0
    telemetry_complete = True
    market_data = {"exchange": "binance", "pair": "BTC/USDT", "timeframe": "1h", "stale": True}
    for config_path in sorted(CONFIG_DIR.glob("config_btc_*.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            port = int(config["api_server"]["listen_port"])
            open_profit += _get_open_profit(port, config)
        except (OSError, KeyError, ValueError, URLError, TimeoutError, json.JSONDecodeError):
            telemetry_complete = False
    try:
        first_config = json.loads((CONFIG_DIR / "config_btc_01.json").read_text(encoding="utf-8"))
        market_data = _get_market_data_status(8081, first_config)
    except (OSError, KeyError, ValueError, URLError, TimeoutError, json.JSONDecodeError):
        telemetry_complete = False

    equity = float(state["starting_equity"]) + realized_profit + open_profit
    previous_snapshot = state.get("last_snapshot", {})
    stored_peak = state.get(
        "peak_equity",
        previous_snapshot.get("peak_equity", state["starting_equity"]),
    )
    peak_equity = max(float(state["starting_equity"]), float(stored_peak))
    running_equity = float(state["starting_equity"])
    for _, profit in trades:
        running_equity += profit
        peak_equity = max(peak_equity, running_equity)
    peak_equity = max(peak_equity, running_equity)
    drawdown = max(0.0, (peak_equity - equity) / peak_equity) if peak_equity else 0.0
    daily_profit += open_profit
    benchmark_return = None
    benchmark_start = state.get("benchmark_start_price")
    current_btc_price = market_data.get("last_close")
    if benchmark_start and current_btc_price:
        benchmark_return = round((current_btc_price / benchmark_start - 1) * 100, 4)

    return {
        "as_of_utc": now.isoformat(),
        "starting_equity": float(state["starting_equity"]),
        "equity": round(equity, 4),
        "realized_profit": round(realized_profit, 4),
        "open_profit": round(open_profit, 4),
        "daily_profit": round(daily_profit, 4),
        "peak_equity": round(peak_equity, 4),
        "drawdown_pct": round(drawdown * 100, 4),
        "telemetry_complete": telemetry_complete,
        "market_data": market_data,
        "benchmark_return_pct": benchmark_return,
        "days_remaining": max(
            0.0,
            (datetime.fromisoformat(state["ends_at_utc"]) - now).total_seconds() / 86400,
        ),
        "guard_triggered": bool(state.get("guard_triggered")),
        "guard_reason": state.get("guard_reason"),
    }


def persist_guard(reason: str) -> dict:
    state = read_state()
    if not state.get("guard_triggered"):
        state["guard_triggered"] = True
        state["guard_reason"] = reason
        state["guard_triggered_at_utc"] = datetime.now(UTC).isoformat()
        _write_state(state)
        write_alert(f"GLOBAL ENTRIES PAUSED: {reason}")
    return state


def persist_snapshot(snapshot: dict) -> None:
    state = read_state()
    if snapshot.get("telemetry_complete"):
        previous_snapshot = state.get("last_snapshot", {})
        stored_peak = state.get(
            "peak_equity",
            previous_snapshot.get("peak_equity", state["starting_equity"]),
        )
        peak_equity = max(
            float(stored_peak),
            float(snapshot.get("peak_equity", state["starting_equity"])),
            float(snapshot["equity"]),
        )
        snapshot["peak_equity"] = round(peak_equity, 4)
        snapshot["drawdown_pct"] = round(
            max(0.0, (peak_equity - float(snapshot["equity"])) / peak_equity * 100),
            4,
        )
        state["peak_equity"] = snapshot["peak_equity"]
    market_data = snapshot.get("market_data", {})
    stale = market_data.get("stale")
    previous_stale = state.get("market_data_stale")
    if stale is not None and previous_stale is not None and stale != previous_stale:
        write_alert("BTC/USDT 1h candles are stale." if stale else "BTC/USDT 1h candles recovered.")
    if stale is not None:
        state["market_data_stale"] = stale
    current_price = market_data.get("last_close")
    if current_price and not state.get("benchmark_start_price"):
        state["benchmark_start_price"] = current_price
        state["benchmark_started_at_utc"] = snapshot.get("as_of_utc")
    benchmark_start = state.get("benchmark_start_price")
    if current_price and benchmark_start:
        snapshot["benchmark_return_pct"] = round((current_price / benchmark_start - 1) * 100, 4)
    state["last_snapshot"] = snapshot
    _write_state(state)


def persist_daily_pause(reason: str) -> dict:
    state = read_state()
    now = datetime.now(UTC)
    next_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    new_until = next_midnight.isoformat()
    if state.get("daily_pause_until_utc") != new_until:
        write_alert(f"DAILY FLEET PAUSE until {new_until}: {reason}")
    state["daily_pause_until_utc"] = new_until
    state["daily_pause_reason"] = reason
    _write_state(state)
    return state


def clear_daily_pause() -> dict:
    state = read_state()
    state["daily_pause_until_utc"] = None
    state["daily_pause_reason"] = None
    _write_state(state)
    write_alert("Daily fleet pause ended.")
    return state


def set_market_data_pause(reason: str) -> dict:
    state = read_state()
    if not state.get("market_data_paused"):
        state["market_data_paused"] = True
        state["market_data_pause_reason"] = reason
        state["market_data_pause_at_utc"] = datetime.now(UTC).isoformat()
        _write_state(state)
        write_alert(f"DATA PAUSE: {reason}")
    return state


def clear_market_data_pause() -> dict:
    state = read_state()
    if state.get("market_data_paused"):
        state["market_data_paused"] = False
        state["market_data_pause_reason"] = None
        _write_state(state)
        write_alert("BTC candle feed recovered; market-data pause cleared.")
    return state


def write_alert(message: str) -> None:
    ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ALERT_PATH.open("a", encoding="utf-8") as alert_file:
        alert_file.write(f"{datetime.now(UTC).isoformat()} {message}\n")


def recent_alerts(limit: int = 10) -> list[str]:
    if not ALERT_PATH.exists():
        return []
    return ALERT_PATH.read_text(encoding="utf-8").splitlines()[-limit:]


def backup_databases(retention_days: int = 35) -> list[Path]:
    """Create consistent SQLite backups, including databases using WAL mode."""
    today = datetime.now(UTC).date()
    day_dir = BACKUP_DIR / today.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for source_path in sorted(DB_DIR.glob("btc_*.sqlite")):
        backup_path = day_dir / source_path.name
        if backup_path.exists():
            continue
        try:
            with sqlite3.connect(source_path, timeout=10) as source:
                with sqlite3.connect(backup_path, timeout=10) as destination:
                    source.backup(destination)
            created.append(backup_path)
        except sqlite3.Error as exc:
            write_alert(f"DATABASE BACKUP FAILED for {source_path.name}: {exc}")

    cutoff = today - timedelta(days=retention_days)
    for folder in BACKUP_DIR.iterdir():
        if not folder.is_dir():
            continue
        try:
            if datetime.strptime(folder.name, "%Y-%m-%d").date() < cutoff:
                for backup_file in folder.iterdir():
                    if backup_file.is_file():
                        backup_file.unlink()
                folder.rmdir()
        except ValueError:
            continue
    return created

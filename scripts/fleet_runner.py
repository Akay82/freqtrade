import os
import sys

# Ensure project root and scripts directory are on python path
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPTS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import time
import subprocess
import signal
import threading
import json
import base64
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import uvicorn
from comparison_dashboard import app, STRATEGIES, record_snapshot, get_strategy_metrics
from fleet_risk import (
    backup_databases,
    clear_market_data_pause,
    clear_daily_pause,
    fetch_fast_market_data,
    persist_daily_pause,
    portfolio_snapshot,
    persist_guard,
    persist_snapshot,
    read_state,
    record_volatility_rule_activation,
    set_market_data_pause,
    update_fast_volatility_guard,
    update_volatility_pause,
    write_alert,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREQTRADE_EXE = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "freqtrade.exe")
LOG_DIR = os.path.join(PROJECT_ROOT, "user_data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

bot_processes = {}
running = True
RISK_CHECK_INTERVAL = 10


def pause_fleet_entries(reason):
    print(f"[FleetManager] Risk guard active: {reason}. Pausing new entries across fleet.", flush=True)
    for strat in STRATEGIES:
        cfg_file = os.path.join(PROJECT_ROOT, "user_data", "configs", f"config_btc_{strat['id']}.json")
        try:
            with open(cfg_file, encoding="utf-8") as cfg_handle:
                config = json.load(cfg_handle)
            api = config["api_server"]
            raw_auth = f"{api['username']}:{api['password']}".encode("utf-8")
            token = base64.b64encode(raw_auth).decode("ascii")
            req = Request(
                f"http://127.0.0.1:{strat['port']}/api/v1/stopentry",
                data=b"",
                headers={"Authorization": f"Basic {token}"},
                method="POST",
            )
            with urlopen(req, timeout=3):
                pass
        except Exception as exc:
            print(f"[FleetManager] Could not pause bot {strat['id']}: {exc}", flush=True)


def resume_fleet_entries():
    print("[FleetManager] Daily pause ended; resuming eligible bots.", flush=True)
    for strat in STRATEGIES:
        cfg_file = os.path.join(PROJECT_ROOT, "user_data", "configs", f"config_btc_{strat['id']}.json")
        try:
            with open(cfg_file, encoding="utf-8") as cfg_handle:
                config = json.load(cfg_handle)
            api = config["api_server"]
            raw_auth = f"{api['username']}:{api['password']}".encode("utf-8")
            token = base64.b64encode(raw_auth).decode("ascii")
            req = Request(
                f"http://127.0.0.1:{strat['port']}/api/v1/start",
                data=b"",
                headers={"Authorization": f"Basic {token}"},
                method="POST",
            )
            with urlopen(req, timeout=3):
                pass
        except Exception as exc:
            print(f"[FleetManager] Could not resume bot {strat['id']}: {exc}", flush=True)

def start_bot(strat):
    bot_id = strat["id"]
    name = strat["name"]
    port = strat["port"]
    cfg_file = os.path.join(PROJECT_ROOT, "user_data", "configs", f"config_btc_{bot_id}.json")
    log_path = os.path.join(LOG_DIR, f"bot_{bot_id}.log")
    
    cmd = [
        FREQTRADE_EXE,
        "trade",
        "--config", cfg_file,
        "--strategy", name,
        "--logfile", log_path,
    ]
    print(f"[FleetManager] Starting Bot {bot_id}: {name} (Port {port})...", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )
    bot_processes[bot_id] = {
        "proc": proc,
        "strat": strat,
    }

def monitor_and_maintain_fleet():
    global running
    record_volatility_rule_activation()
    backup_databases()
    last_backup_date = datetime.now(timezone.utc).date()
    # Initial start staggered by 700ms to avoid exchange rate-limits on startup
    for s in STRATEGIES:
        if not running:
            break
        start_bot(s)
        time.sleep(0.7)

    print("[FleetManager] All 10 BTC bots initiated. Entering supervision loop...", flush=True)

    last_risk_check = 0.0
    last_metrics_snapshot = time.monotonic()
    telemetry_failures = 0
    while running:
        time.sleep(10)
        for s in STRATEGIES:
            bot_id = s["id"]
            entry = bot_processes.get(bot_id)
            if entry:
                proc = entry["proc"]
                ret = proc.poll()
                if ret is not None and running:
                    print(f"[FleetManager] WARNING: Bot {bot_id} exited with code {ret}. Restarting in 5s...", flush=True)
                    write_alert(f"Bot {bot_id} exited with code {ret}; restarting.")
                    time.sleep(5)
                    start_bot(s)

        today_utc = datetime.now(timezone.utc).date()
        if today_utc != last_backup_date:
            backup_databases()
            last_backup_date = today_utc

        if time.monotonic() - last_metrics_snapshot >= 300:
            last_metrics_snapshot = time.monotonic()
            try:
                record_snapshot([get_strategy_metrics(strategy) for strategy in STRATEGIES])
            except Exception as exc:
                print(f"[FleetManager] Could not write 5-minute risk samples: {exc}", flush=True)

        if time.monotonic() - last_risk_check >= RISK_CHECK_INTERVAL:
            last_risk_check = time.monotonic()
            state = read_state()
            previous_entry_pause = any(
                state.get(key)
                for key in (
                    "daily_pause_until_utc",
                    "market_data_paused",
                    "volatility_paused",
                    "fast_volatility_paused",
                    "fast_volatility_data_paused",
                )
            )
            snapshot = portfolio_snapshot()
            try:
                fast_market_data = fetch_fast_market_data()
            except Exception as exc:
                print(f"[FleetManager] Fast BTC 5m feed check failed: {exc}", flush=True)
                fast_market_data = None
            state = update_fast_volatility_guard(fast_market_data)
            if fast_market_data is None:
                snapshot["market_data"]["fast_volatility"] = {
                    "available": False,
                    "paused": bool(state.get("fast_volatility_paused")),
                    "data_paused": bool(state.get("fast_volatility_data_paused")),
                    "reason": state.get("fast_volatility_pause_reason") or state.get("fast_volatility_data_pause_reason"),
                }
            else:
                snapshot["market_data"]["fast_volatility"] = {
                    "available": True,
                    "paused": bool(state.get("fast_volatility_paused")),
                    "data_paused": bool(state.get("fast_volatility_data_paused")),
                    "reason": state.get("fast_volatility_pause_reason") or state.get("fast_volatility_data_pause_reason"),
                    "interval": "5m",
                    "price": fast_market_data.get("price"),
                    "move_pct": fast_market_data["current_candle"].get("move_pct"),
                    "range_pct": fast_market_data["current_candle"].get("range_pct"),
                    "age_seconds": fast_market_data.get("age_seconds"),
                    "stale": fast_market_data.get("stale"),
                    "trigger_move_pct": 1.5,
                    "trigger_range_pct": 2.0,
                    "recovery_candles_seen": int(state.get("fast_volatility_recovery_count", 0)),
                    "recovery_candles_required": 6,
                }
            if snapshot.get("market_data", {}).get("recent_completed_candles"):
                state = update_volatility_pause(snapshot["market_data"])
            snapshot["market_data"]["volatility_guard"] = {
                "paused": bool(state.get("volatility_paused")),
                "reason": state.get("volatility_pause_reason"),
                "recovery_candles_seen": int(state.get("volatility_recovery_count", 0)),
                "trigger_move_pct": 3.0,
                "trigger_range_pct": 4.0,
                "recovery_move_pct": 1.5,
                "recovery_range_pct": 2.5,
                "recovery_candles_required": 2,
            }
            persist_snapshot(snapshot)
            state = read_state()
            if state.get("guard_triggered"):
                pause_fleet_entries(state.get("guard_reason") or "previous risk guard")
            elif not snapshot["telemetry_complete"]:
                telemetry_failures += 1
                print("[FleetManager] Risk telemetry incomplete; global guard check will retry.", flush=True)
                active_reasons = [
                    state.get("volatility_pause_reason")
                    if state.get("volatility_paused") else None,
                    state.get("market_data_pause_reason")
                    if state.get("market_data_paused") else None,
                    state.get("fast_volatility_pause_reason")
                    if state.get("fast_volatility_paused") else None,
                    state.get("fast_volatility_data_pause_reason")
                    if state.get("fast_volatility_data_paused") else None,
                    state.get("daily_pause_reason")
                    if state.get("daily_pause_until_utc") else None,
                ]
                active_reasons = [reason for reason in active_reasons if reason]
                if active_reasons:
                    pause_fleet_entries("; ".join(active_reasons))
                if telemetry_failures >= 4:
                    state = persist_guard("global risk telemetry unavailable; entries paused for safety")
                    pause_fleet_entries(state["guard_reason"])
            else:
                telemetry_failures = 0
                reason = None
                daily_pause_cleared = False
                if snapshot["days_remaining"] <= 0:
                    reason = "30-day test window complete"
                elif snapshot["drawdown_pct"] >= 10:
                    reason = "fleet peak-to-trough drawdown reached 10%"
                if reason:
                    state = persist_guard(reason)
                    pause_fleet_entries(state["guard_reason"])
                elif snapshot["daily_profit"] <= -snapshot["starting_equity"] * 0.02:
                    state = persist_daily_pause("fleet daily loss reached 2% of starting equity")
                elif state.get("daily_pause_until_utc"):
                    pause_until = datetime.fromisoformat(state["daily_pause_until_utc"])
                    if datetime.now(timezone.utc) >= pause_until:
                        clear_daily_pause()
                        state = read_state()
                        daily_pause_cleared = True

                if (
                    state
                    and not state.get("guard_triggered")
                    and not state.get("daily_pause_until_utc")
                ):
                    market_data = snapshot.get("market_data", {})
                    if market_data.get("stale"):
                        state = set_market_data_pause("Binance BTC/USDT 1h candle feed is stale")
                    elif state.get("market_data_paused"):
                        clear_market_data_pause()

                state = read_state()
                if not state.get("guard_triggered"):
                    active_pauses = []
                    if state.get("daily_pause_until_utc"):
                        active_pauses.append(state.get("daily_pause_reason") or "daily loss pause")
                    if state.get("market_data_paused"):
                        active_pauses.append(state.get("market_data_pause_reason") or "stale BTC feed")
                    if state.get("volatility_paused"):
                        active_pauses.append(state.get("volatility_pause_reason") or "BTC volatility pause")
                    if state.get("fast_volatility_paused"):
                        active_pauses.append(state.get("fast_volatility_pause_reason") or "BTC 5m volatility pause")
                    if state.get("fast_volatility_data_paused"):
                        active_pauses.append(state.get("fast_volatility_data_pause_reason") or "BTC 5m feed unavailable")
                    if active_pauses:
                        pause_fleet_entries("; ".join(active_pauses))
                    elif previous_entry_pause or daily_pause_cleared:
                        resume_fleet_entries()

def stop_all_bots():
    global running
    running = False
    print("[FleetManager] Shutting down all 10 bot subprocesses...", flush=True)
    for bot_id, entry in list(bot_processes.items()):
        try:
            proc = entry["proc"]
            proc.terminate()
        except Exception:
            pass
    print("[FleetManager] Fleet stopped cleanly.", flush=True)

def signal_handler(sig, frame):
    stop_all_bots()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    fleet_thread = threading.Thread(target=monitor_and_maintain_fleet, daemon=True)
    fleet_thread.start()

    print("[FleetManager] Launching Unified Comparison Dashboard on http://127.0.0.1:8000...", flush=True)
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
    finally:
        stop_all_bots()

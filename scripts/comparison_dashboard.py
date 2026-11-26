import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPTS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
import glob
import csv
import math
import json
import sqlite3
import datetime
import threading
import time
import base64
from typing import Dict, List, Any
import urllib.request
import urllib.error

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
try:
    from scripts.fleet_risk import read_state, recent_alerts
    from scripts.engine.api_router import engine_router
except ModuleNotFoundError:
    from fleet_risk import read_state, recent_alerts
    from engine.api_router import engine_router

app = FastAPI(title="BTC 10-Strategy Swing Fleet Dashboard")
app.include_router(engine_router)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD_DIR = os.path.join(PROJECT_ROOT, "user_data", "records")
os.makedirs(RECORD_DIR, exist_ok=True)
CSV_PATH = os.path.join(RECORD_DIR, "daily_snapshots.csv")
DETAIL_CSV_PATH = os.path.join(RECORD_DIR, "detailed_strategy_metrics.csv")
SNAPSHOT_LOCK = threading.Lock()

STRATEGIES = [
    {"id": "01", "name": "BTCSwing_EMATrend", "port": 8081, "type": "Multi-Timeframe Trend", "desc": "4h Trend Filter (50/200) + 1h Pullback"},
    {"id": "02", "name": "BTCSwing_SuperTrend", "port": 8082, "type": "Volatility Breakout", "desc": "SuperTrend ATR 3.0 + ADX > 22 Filter"},
    {"id": "03", "name": "BTCSwing_MACDDivergence", "port": 8083, "type": "Momentum Acceleration", "desc": "MACD Signal Cross + Expanding Histogram"},
    {"id": "04", "name": "BTCSwing_SqueezeBreakout", "port": 8084, "type": "TTM Squeeze", "desc": "Bollinger inside Keltner Volatility Expansion"},
    {"id": "05", "name": "BTCSwing_DonchianTurtle", "port": 8085, "type": "Turtle Breakout", "desc": "20-Period High Breakout + 10-Period Low Exit"},
    {"id": "06", "name": "BTCSwing_IchimokuCloud", "port": 8086, "type": "Ichimoku Cloud", "desc": "Kumo Cloud Breakout + Tenkan/Kijun Alignment"},
    {"id": "07", "name": "BTCSwing_RSI_VWAP", "port": 8087, "type": "Mean Reversion", "desc": "Anchored Rolling VWAP Discount + RSI Reclaim"},
    {"id": "08", "name": "BTCSwing_EMARibbon", "port": 8088, "type": "Guppy Ribbon", "desc": "Fast EMA (8/13/21) over Slow EMA (55/89/144)"},
    {"id": "09", "name": "BTCSwing_StochRSI_CMF", "port": 8089, "type": "Money Flow", "desc": "Chaikin Money Flow (>0.03) + StochRSI Cross"},
    {"id": "10", "name": "BTCSwing_FibonacciBands", "port": 8090, "type": "Fibonacci Volatility", "desc": "1.618 ATR Fibonacci Band Discount Buying"}
]

def check_bot_online(port: int) -> bool:
    url = f"http://127.0.0.1:{port}/api/v1/ping"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FleetDashboard"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            data = json.loads(resp.read().decode())
            return data.get("status") == "pong"
    except Exception:
        return False

def get_bot_open_status(bot_id: str, port: int) -> List[Dict[str, Any]]:
    config_path = os.path.join(PROJECT_ROOT, "user_data", "configs", f"config_btc_{bot_id}.json")
    try:
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
        api = config["api_server"]
        raw_auth = f"{api['username']}:{api['password']}".encode("utf-8")
        token = base64.b64encode(raw_auth).decode("ascii")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/status",
            headers={"Authorization": f"Basic {token}"},
        )
        with urllib.request.urlopen(req, timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

def parse_utc_datetime(value: Any) -> datetime.datetime:
    parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)


def sampled_max_drawdown(bot_id: str, test_start: datetime.datetime, current_balance: float) -> float:
    """Estimate equity drawdown from periodic marked-to-market balance samples."""
    points = []
    if os.path.exists(CSV_PATH):
        with SNAPSHOT_LOCK:
            try:
                with open(CSV_PATH, newline="", encoding="utf-8") as snapshot_file:
                    for row in csv.DictReader(snapshot_file):
                        if row.get("bot_id") != bot_id:
                            continue
                        timestamp = parse_utc_datetime(row["timestamp"])
                        if timestamp >= test_start:
                            points.append((timestamp, float(row["balance"])))
            except (OSError, KeyError, ValueError, csv.Error):
                pass
    points.sort(key=lambda point: point[0])
    points.append((datetime.datetime.now(datetime.timezone.utc), current_balance))
    peak = 100.0
    maximum = 0.0
    for _, balance in points:
        peak = max(peak, balance)
        if peak > 0:
            maximum = max(maximum, (peak - balance) / peak * 100)
    return maximum

def get_strategy_metrics(strat: Dict[str, Any]) -> Dict[str, Any]:
    bot_id = strat["id"]
    name = strat["name"]
    port = strat["port"]
    strat_type = strat["type"]
    desc = strat["desc"]
    
    online = check_bot_online(port)
    
    # Locate database
    db_candidates = [
        os.path.join(PROJECT_ROOT, "user_data", "db", f"btc_{bot_id}.sqlite"),
        os.path.join(PROJECT_ROOT, "user_data", "db", f"btc_{bot_id}.dryrun.sqlite"),
        os.path.join(PROJECT_ROOT, "user_data", f"tradesv3_btc_{bot_id}.dryrun.sqlite")
    ]
    db_path = None
    for cand in db_candidates:
        if os.path.exists(cand):
            db_path = cand
            break
            
    starting_balance = 100.00
    realized_profit = 0.0
    total_trades = 0
    wins = 0
    losses = 0
    open_trade = None
    unrealized_profit = 0.0
    gross_wins = 0.0
    gross_losses = 0.0
    total_fees = 0.0
    trade_durations = []
    slippage_bps = []
    closed_trade_drawdown_pct = 0.0
    total_trade_hours = 0.0
    test_state = read_state()
    test_start = parse_utc_datetime(test_state["started_at_utc"])
    
    if db_path:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Query closed trades
            cursor.execute("""
                SELECT id, pair, open_date, close_date, open_rate, open_rate_requested,
                       close_rate, close_rate_requested, close_profit, close_profit_abs,
                       exit_reason, fee_open_cost, fee_close_cost
                FROM trades 
                WHERE is_open = 0
                ORDER BY close_date ASC
            """)
            closed_rows = cursor.fetchall()
            closed_rows = [row for row in closed_rows if parse_utc_datetime(row["close_date"]) >= test_start]
            total_trades = len(closed_rows)
            for row in closed_rows:
                p_abs = row["close_profit_abs"] or 0.0
                realized_profit += p_abs
                if p_abs > 0:
                    wins += 1
                    gross_wins += p_abs
                elif p_abs < 0:
                    losses += 1
                    gross_losses += abs(p_abs)
                total_fees += abs(row["fee_open_cost"] or 0.0) + abs(row["fee_close_cost"] or 0.0)

                open_date = parse_utc_datetime(row["open_date"])
                close_date = parse_utc_datetime(row["close_date"])
                duration_hours = max(0.0, (close_date - open_date).total_seconds() / 3600)
                trade_durations.append(duration_hours)
                total_trade_hours += duration_hours
                for requested, actual in (
                    (row["open_rate_requested"], row["open_rate"]),
                    (row["close_rate_requested"], row["close_rate"]),
                ):
                    if requested and actual:
                        slippage_bps.append(abs((actual / requested - 1) * 10000))

            equity = peak = starting_balance
            for row in closed_rows:
                equity += row["close_profit_abs"] or 0.0
                peak = max(peak, equity)
                if peak > 0:
                    closed_trade_drawdown_pct = max(
                        closed_trade_drawdown_pct, (peak - equity) / peak * 100
                    )

            # Query open trades
            cursor.execute("""
                SELECT id, pair, open_date, open_rate, amount, stake_amount, stop_loss_pct
                FROM trades 
                WHERE is_open = 1
                ORDER BY open_date DESC
                LIMIT 1
            """)
            open_row = cursor.fetchone()
            if open_row:
                open_trade = {
                    "id": open_row["id"],
                    "pair": open_row["pair"],
                    "open_date": str(open_row["open_date"]),
                    "open_rate": open_row["open_rate"],
                    "stake_amount": open_row["stake_amount"],
                    "stop_loss_pct": open_row["stop_loss_pct"]
                }
            conn.close()
        except Exception as e:
            pass

    api_open_trades = get_bot_open_status(bot_id, port) if online else []
    if api_open_trades:
        unrealized_profit = sum(float(t.get("profit_abs") or 0.0) for t in api_open_trades)
        if open_trade:
            open_date = parse_utc_datetime(open_trade["open_date"])
            total_trade_hours += max(
                0.0,
                (datetime.datetime.now(datetime.timezone.utc) - open_date).total_seconds() / 3600,
            )
        if open_trade:
            open_trade["current_profit_abs"] = round(unrealized_profit, 4)
            open_trade["current_profit_pct"] = round(
                sum(float(t.get("profit_pct") or 0.0) for t in api_open_trades), 4
            )
            open_trade["current_rate"] = api_open_trades[0].get("current_rate")

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else None
    expectancy = realized_profit / total_trades if total_trades else 0.0
    elapsed_hours = max(
        1.0,
        (datetime.datetime.now(datetime.timezone.utc) - test_start).total_seconds() / 3600,
    )
    exposure_pct = min(100.0, total_trade_hours / elapsed_hours * 100)
    current_balance = starting_balance + realized_profit + unrealized_profit
    total_roi_pct = ((current_balance - starting_balance) / starting_balance) * 100
    max_drawdown_pct = max(
        closed_trade_drawdown_pct,
        sampled_max_drawdown(bot_id, test_start, current_balance),
    )
    sorted_slippage_bps = sorted(slippage_bps)
    p95_slippage_bps = (
        sorted_slippage_bps[max(0, math.ceil(len(sorted_slippage_bps) * 0.95) - 1)]
        if sorted_slippage_bps else None
    )

    return {
        "id": bot_id,
        "name": name,
        "port": port,
        "type": strat_type,
        "desc": desc,
        "online": online,
        "starting_balance": round(starting_balance, 2),
        "current_balance": round(current_balance, 2),
        "realized_profit": round(realized_profit, 2),
        "unrealized_profit": round(unrealized_profit, 2),
        "net_profit": round(realized_profit + unrealized_profit, 2),
        "roi_pct": round(total_roi_pct, 2),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "expectancy": round(expectancy, 4),
        "average_win": round(gross_wins / wins, 4) if wins else 0.0,
        "average_loss": round(-gross_losses / losses, 4) if losses else 0.0,
        "max_drawdown_pct": round(max_drawdown_pct, 3),
        "max_drawdown_basis": "closed-trade path plus periodic marked-to-market samples",
        "total_fees": round(total_fees, 4),
        "average_hold_hours": round(sum(trade_durations) / len(trade_durations), 2) if trade_durations else 0.0,
        "exposure_pct": round(exposure_pct, 2),
        "average_slippage_bps": round(sum(slippage_bps) / len(slippage_bps), 2) if slippage_bps else None,
        "p95_slippage_bps": round(p95_slippage_bps, 2) if p95_slippage_bps is not None else None,
        "has_open_trade": open_trade is not None,
        "open_trade": open_trade
    }

def record_snapshot(fleet_data: List[Dict[str, Any]]) -> None:
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    header_needed = not os.path.exists(CSV_PATH)
    
    with SNAPSHOT_LOCK:
        with open(CSV_PATH, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("timestamp,bot_id,strategy,type,balance,net_profit,roi_pct,trades,wins,losses,win_rate,online\n")
            for b in fleet_data:
                f.write(f"{now_iso},{b['id']},{b['name']},{b['type']},{b['current_balance']},{b['net_profit']},{b['roi_pct']},{b['total_trades']},{b['wins']},{b['losses']},{b['win_rate']},{b['online']}\n")

        detail_header = "timestamp,bot_id,strategy,trades,net_profit,win_rate,profit_factor,expectancy,max_drawdown_pct,average_win,average_loss,total_fees,average_hold_hours,exposure_pct,average_slippage_bps,p95_slippage_bps\n"
        with open(DETAIL_CSV_PATH, "a", encoding="utf-8") as detail_file:
            if detail_file.tell() == 0:
                detail_file.write(detail_header)
            for b in fleet_data:
                detail_file.write(
                    f"{now_iso},{b['id']},{b['name']},{b['total_trades']},{b['net_profit']},"
                    f"{b['win_rate']},{b['profit_factor']},{b['expectancy']},{b['max_drawdown_pct']},"
                    f"{b['average_win']},{b['average_loss']},{b['total_fees']},{b['average_hold_hours']},{b['exposure_pct']},"
                    f"{b['average_slippage_bps']},{b['p95_slippage_bps']}\n"
                )

def auto_snapshot_worker():
    while True:
        try:
            fleet = [get_strategy_metrics(s) for s in STRATEGIES]
            record_snapshot(fleet)
        except Exception:
            pass
        # sleep 6 hours (21600 seconds)
        time.sleep(21600)

snapshot_thread = threading.Thread(target=auto_snapshot_worker, daemon=True)
snapshot_thread.start()

@app.get("/api/fleet")
def api_fleet():
    fleet = [get_strategy_metrics(s) for s in STRATEGIES]
    # Sort leaderboard by net_profit descending
    fleet_sorted = sorted(fleet, key=lambda x: x["net_profit"], reverse=True)
    
    total_starting = sum(x["starting_balance"] for x in fleet)
    total_current = sum(x["current_balance"] for x in fleet)
    total_net = sum(x["net_profit"] for x in fleet)
    total_trades = sum(x["total_trades"] for x in fleet)
    total_wins = sum(x["wins"] for x in fleet)
    total_losses = sum(x["losses"] for x in fleet)
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    fleet_roi = ((total_current - total_starting) / total_starting) * 100
    test_state = read_state()
    risk_snapshot = test_state.get("last_snapshot", {})
    test_info = {
        "started_at_utc": test_state.get("started_at_utc"),
        "ends_at_utc": test_state.get("ends_at_utc"),
        "days_remaining": risk_snapshot.get("days_remaining"),
        "daily_profit": risk_snapshot.get("daily_profit"),
        "drawdown_pct": risk_snapshot.get("drawdown_pct"),
        "benchmark_return_pct": risk_snapshot.get("benchmark_return_pct"),
        "market_data": risk_snapshot.get("market_data"),
        "guard_triggered": test_state.get("guard_triggered", False),
        "guard_reason": test_state.get("guard_reason"),
        "daily_pause_until_utc": test_state.get("daily_pause_until_utc"),
        "daily_pause_reason": test_state.get("daily_pause_reason"),
        "market_data_paused": test_state.get("market_data_paused", False),
        "market_data_pause_reason": test_state.get("market_data_pause_reason"),
        "volatility_paused": test_state.get("volatility_paused", False),
        "volatility_pause_reason": test_state.get("volatility_pause_reason"),
        "volatility_recovery_count": test_state.get("volatility_recovery_count", 0),
        "volatility_circuit_breaker": test_state.get("volatility_circuit_breaker"),
        "fast_volatility_paused": test_state.get("fast_volatility_paused", False),
        "fast_volatility_data_paused": test_state.get("fast_volatility_data_paused", False),
        "fast_volatility_pause_reason": test_state.get("fast_volatility_pause_reason"),
        "fast_volatility_data_pause_reason": test_state.get("fast_volatility_data_pause_reason"),
        "fast_volatility_recovery_count": test_state.get("fast_volatility_recovery_count", 0),
        "fast_volatility_circuit_breaker": test_state.get("fast_volatility_circuit_breaker"),
        "fast_volatility_data_circuit_breaker": {
            "failure_checks_before_pause": 3,
            "healthy_checks_before_recovery": 3,
        },
        "volatility_events": test_state.get("volatility_events", [])[-8:],
        "volatility_event_count": len(test_state.get("volatility_events", [])),
        "recent_alerts": recent_alerts(5),
    }

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_starting": round(total_starting, 2),
            "total_current": round(total_current, 2),
            "total_net": round(total_net, 2),
            "fleet_roi": round(fleet_roi, 2),
            "total_trades": total_trades,
            "overall_win_rate": round(overall_win_rate, 1),
            "active_positions": sum(1 for x in fleet if x["has_open_trade"]),
            "online_bots": sum(1 for x in fleet if x["online"]),
            "test": test_info,
        },
        "leaderboard": fleet_sorted
    }

@app.post("/api/snapshot")
def api_manual_snapshot():
    fleet = [get_strategy_metrics(s) for s in STRATEGIES]
    record_snapshot(fleet)
    return {"status": "success", "message": "Performance snapshot successfully logged to CSV"}

@app.get("/api/download_csv")
def api_download_csv():
    if not os.path.exists(CSV_PATH):
        fleet = [get_strategy_metrics(s) for s in STRATEGIES]
        record_snapshot(fleet)
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content=content, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=btc_strategies_snapshots.csv"})

@app.get("/", response_class=HTMLResponse)
def index_page():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC 10-Strategy Swing Fleet Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background-color: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
        .glass-card { background: rgba(22, 27, 34, 0.85); backdrop-filter: blur(10px); border: 1px solid #30363d; }
        .glass-card:hover { border-color: #58a6ff; transition: border-color 0.2s ease-in-out; }
    </style>
</head>
<body class="min-h-screen p-4 md:p-8">
    <div class="max-w-7xl mx-auto space-y-6">
        
        <!-- Header -->
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 pb-4 border-b border-gray-800">
            <div>
                <div class="flex items-center gap-3">
                    <div class="p-2.5 bg-yellow-500/10 text-yellow-500 rounded-xl border border-yellow-500/20">
                        <i class="fa-brands fa-bitcoin text-2xl"></i>
                    </div>
                    <div>
                        <h1 class="text-2xl font-bold text-white flex items-center gap-2">
                            BTC 10-Strategy Swing Fleet
                            <span class="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">1-Month Challenge</span>
                        </h1>
                <p class="text-sm text-gray-400">Binance Spot: <span class="text-yellow-400 font-semibold">BTC/USDT</span> • Candles: <span class="text-white font-medium">1h, completed candles</span> • Paper allocation: <span class="text-white font-medium">$100 Each</span></p>
                    </div>
                </div>
            </div>
            <div class="flex items-center gap-3">
                <span id="lastUpdated" class="text-xs text-gray-400">Syncing...</span>
                <button onclick="fetchFleetData()" class="px-3.5 py-2 bg-gray-800 hover:bg-gray-700 text-sm font-medium text-white rounded-lg border border-gray-700 flex items-center gap-2 transition">
                    <i class="fa-solid fa-arrows-rotate" id="refreshIcon"></i> Refresh
                </button>
                <button onclick="saveSnapshot()" class="px-3.5 py-2 bg-blue-600 hover:bg-blue-500 text-sm font-medium text-white rounded-lg border border-blue-500 flex items-center gap-2 transition">
                    <i class="fa-solid fa-camera"></i> Snapshot
                </button>
                <a href="/api/download_csv" class="px-3.5 py-2 bg-emerald-600 hover:bg-emerald-500 text-sm font-medium text-white rounded-lg border border-emerald-500 flex items-center gap-2 transition">
                    <i class="fa-solid fa-file-csv"></i> Export CSV
                </a>
            </div>
        </header>

        <!-- Navigation Tabs (Section 18) -->
        <nav class="flex flex-wrap gap-2 border-b border-gray-800 pb-3">
            <button onclick="switchTab('fleet')" id="btn-tab-fleet" class="tab-btn px-4 py-2 rounded-lg text-sm font-medium bg-blue-600 text-white">10-Bot Fleet</button>
            <button onclick="switchTab('market')" id="btn-tab-market" class="tab-btn px-4 py-2 rounded-lg text-sm font-medium bg-gray-800 text-gray-300 hover:bg-gray-700">Market &amp; Signals</button>
            <button onclick="switchTab('risk')" id="btn-tab-risk" class="tab-btn px-4 py-2 rounded-lg text-sm font-medium bg-gray-800 text-gray-300 hover:bg-gray-700">Risk Monitor</button>
            <button onclick="switchTab('backtest')" id="btn-tab-backtest" class="tab-btn px-4 py-2 rounded-lg text-sm font-medium bg-gray-800 text-gray-300 hover:bg-gray-700">Backtesting</button>
            <button onclick="switchTab('ml')" id="btn-tab-ml" class="tab-btn px-4 py-2 rounded-lg text-sm font-medium bg-gray-800 text-gray-300 hover:bg-gray-700">ML &amp; Attribution</button>
        </nav>

        <div id="tab-fleet" class="space-y-6">
        <div id="riskStatus" class="text-sm text-gray-300 px-1">Risk guard: loading test status…</div>
        <div class="text-xs text-amber-300 px-1">All 10 strategies trade BTC/USDT. This is a rules comparison, not diversification; their returns can be highly correlated.</div>
        <div id="alertBanner" class="hidden rounded-lg border px-4 py-3 text-sm" role="alert"></div>
        <section class="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
            <div class="flex justify-between text-sm font-semibold text-gray-200"><span>Fast volatility pause study (5m BTC candles)</span><span id="volatilityEventCount" class="text-gray-400">0 events</span></div>
            <div id="volatilityEvents" class="mt-2 space-y-2 text-xs text-gray-400">Waiting for volatility events</div>
        </section>

        <!-- KPI Grid -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="glass-card p-5 rounded-2xl">
                <div class="flex justify-between items-center text-gray-400 text-sm mb-1">
                    <span>Total Fleet Capital</span>
                    <i class="fa-solid fa-wallet text-blue-400"></i>
                </div>
                <div class="text-2xl font-bold text-white" id="kpiBalance">$1,000.00</div>
                <div class="text-xs text-gray-400 mt-1">Seeded: $1,000.00</div>
            </div>

            <div class="glass-card p-5 rounded-2xl">
                <div class="flex justify-between items-center text-gray-400 text-sm mb-1">
                    <span>Fleet Net Return</span>
                    <i class="fa-solid fa-chart-line text-emerald-400"></i>
                </div>
                <div class="text-2xl font-bold text-emerald-400" id="kpiReturn">+0.00% ($0.00)</div>
                <div class="text-xs text-gray-400 mt-1">Across 10 Strategies</div>
            </div>

            <div class="glass-card p-5 rounded-2xl">
                <div class="flex justify-between items-center text-gray-400 text-sm mb-1">
                    <span>Win Rate / Total Trades</span>
                    <i class="fa-solid fa-bullseye text-purple-400"></i>
                </div>
                <div class="text-2xl font-bold text-white" id="kpiWinRate">0.0% (0 Trades)</div>
                <div class="text-xs text-gray-400 mt-1">Completed swing cycles</div>
            </div>

            <div class="glass-card p-5 rounded-2xl">
                <div class="flex justify-between items-center text-gray-400 text-sm mb-1">
                    <span>Fleet Online Status</span>
                    <i class="fa-solid fa-server text-emerald-400"></i>
                </div>
                <div class="text-2xl font-bold text-emerald-400" id="kpiOnline">10 / 10 Active</div>
                <div class="text-xs text-gray-400 mt-1" id="kpiActivePositions">0 in active trades</div>
            </div>
        </div>

        <!-- Strategy Leaderboard Table -->
        <div class="glass-card rounded-2xl overflow-hidden border border-gray-800">
            <div class="p-5 border-b border-gray-800 flex justify-between items-center bg-gray-900/40">
                <div>
                    <h2 class="text-lg font-bold text-white flex items-center gap-2">
                        <i class="fa-solid fa-trophy text-yellow-400"></i> Performance Leaderboard
                    </h2>
                    <p class="text-xs text-gray-400">Ranked by Net Profit / ROI. All strategies operate independently on BTC/USDT.</p>
                </div>
                <div class="text-xs text-gray-400 flex items-center gap-2">
                    <span class="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Auto-refreshing every 10s
                </div>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm">
                    <thead class="bg-gray-900/60 text-xs uppercase text-gray-400 border-b border-gray-800">
                        <tr>
                            <th class="py-3.5 px-4 font-semibold">Rank</th>
                            <th class="py-3.5 px-4 font-semibold">Strategy</th>
                            <th class="py-3.5 px-4 font-semibold">Type</th>
                            <th class="py-3.5 px-4 font-semibold">Balance</th>
                            <th class="py-3.5 px-4 font-semibold">Net PnL ($)</th>
                            <th class="py-3.5 px-4 font-semibold">ROI (%)</th>
                            <th class="py-3.5 px-4 font-semibold">Trades</th>
                            <th class="py-3.5 px-4 font-semibold">Win Rate</th>
                            <th class="py-3.5 px-4 font-semibold">Trade Quality</th>
                            <th class="py-3.5 px-4 font-semibold">Current State</th>
                            <th class="py-3.5 px-4 font-semibold text-right">FreqUI</th>
                        </tr>
                    </thead>
                    <tbody id="leaderboardBody" class="divide-y divide-gray-800">
                        <!-- Populated by JS -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Strategy Distribution Visualizer -->
        <div class="glass-card rounded-2xl p-6 border border-gray-800 space-y-4">
            <h3 class="text-base font-bold text-white flex items-center gap-2">
                <i class="fa-solid fa-chart-simple text-blue-400"></i> Strategy Performance Visualizer
            </h3>
            <div id="chartBars" class="space-y-3">
                <!-- Populated by JS -->
            </div>
        </div>
        </div><!-- end tab-fleet -->

        <!-- Tab 2: Market Overview & Signal Analysis (Sections 18A, 18B, 18C) -->
        <div id="tab-market" class="hidden space-y-6">
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4" id="marketKpis">
                <div class="glass-card p-5 rounded-2xl">
                    <div class="text-gray-400 text-xs mb-1">BTC/USDT Spot Price</div>
                    <div id="engPrice" class="text-2xl font-bold text-white">--</div>
                    <div id="engChange24h" class="text-xs text-gray-400 mt-1">--</div>
                </div>
                <div class="glass-card p-5 rounded-2xl">
                    <div class="text-gray-400 text-xs mb-1">Market Regime (4H)</div>
                    <div id="engRegime" class="text-xl font-bold text-yellow-400">--</div>
                    <div id="engRegimeConf" class="text-xs text-gray-400 mt-1">Confidence: --</div>
                </div>
                <div class="glass-card p-5 rounded-2xl">
                    <div class="text-gray-400 text-xs mb-1">MTF Alignment (1D / 4H / 1H)</div>
                    <div id="engMtfBias" class="text-lg font-bold text-white">--</div>
                    <div id="engMtfScore" class="text-xs text-gray-400 mt-1">Score: --</div>
                </div>
                <div class="glass-card p-5 rounded-2xl">
                    <div class="text-gray-400 text-xs mb-1">1H Key Indicators</div>
                    <div id="engIndSummary" class="text-xs text-gray-300 mt-1">RSI: -- · ATR%: -- · ADX: --</div>
                </div>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="glass-card rounded-2xl p-5 border border-gray-800 space-y-3">
                    <h3 class="text-base font-bold text-white">Strategy Monitor (Regime Eligibility)</h3>
                    <div id="engStrategies" class="space-y-2 text-xs text-gray-300">Loading...</div>
                </div>
                <div class="glass-card rounded-2xl p-5 border border-gray-800 space-y-3">
                    <h3 class="text-base font-bold text-white">Latest Signal Analysis &amp; Score</h3>
                    <pre id="engSignalExplanation" class="text-xs text-gray-300 whitespace-pre-wrap bg-gray-900/70 p-3 rounded-lg border border-gray-800">Loading...</pre>
                </div>
            </div>
        </div>

        <!-- Tab 3: Risk Monitor & Kill Switch (Section 18G) -->
        <div id="tab-risk" class="hidden space-y-6">
            <div class="glass-card rounded-2xl p-6 border border-gray-800 space-y-4">
                <div class="flex flex-wrap justify-between items-center gap-4">
                    <div>
                        <h3 class="text-lg font-bold text-white">Independent Risk Management Engine</h3>
                        <p class="text-xs text-gray-400">Spot-only · Leverage Disabled · Cost-adjusted ATR position sizing</p>
                    </div>
                    <div class="flex gap-2">
                        <button onclick="toggleKillSwitch(true)" class="px-4 py-2 bg-rose-600 hover:bg-rose-500 text-xs font-bold text-white rounded-lg">Activate Kill Switch</button>
                        <button onclick="toggleKillSwitch(false)" class="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-xs font-bold text-white rounded-lg">Clear Kill Switch</button>
                    </div>
                </div>
                <div id="engRiskGrid" class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">Loading...</div>
                <div id="engRiskLockouts" class="text-xs text-amber-300"></div>
            </div>
        </div>

        <!-- Tab 4: Backtesting Dashboard (Section 18F) -->
        <div id="tab-backtest" class="hidden space-y-6">
            <div class="glass-card rounded-2xl p-6 border border-gray-800 space-y-4">
                <div class="flex flex-wrap justify-between items-center gap-4">
                    <div>
                        <h3 class="text-lg font-bold text-white">Historical Backtesting &amp; Benchmark Comparison</h3>
                        <p class="text-xs text-gray-400">Zero-lookahead 1D/4H/1H simulation with realistic fees &amp; slippage</p>
                    </div>
                    <button onclick="runEngineBacktest()" class="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-sm font-medium text-white rounded-lg">Run BTC Backtest</button>
                </div>
                <div id="engBacktestStatus" class="text-xs text-gray-400"></div>
                <div id="engBacktestResults" class="space-y-4 text-xs text-gray-300">No backtests loaded yet.</div>
            </div>
        </div>

        <!-- Tab 5: ML Monitor & Attribution (Sections 15 & 18H) -->
        <div id="tab-ml" class="hidden space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="glass-card rounded-2xl p-5 border border-gray-800 space-y-3">
                    <h3 class="text-base font-bold text-white">ML Signal Quality Filter Status</h3>
                    <div id="engMlStatus" class="text-xs text-gray-300">Checking ML model registry...</div>
                </div>
                <div class="glass-card rounded-2xl p-5 border border-gray-800 space-y-3">
                    <h3 class="text-base font-bold text-white">Performance Attribution (By Strategy &amp; Regime)</h3>
                    <div id="engAttribution" class="text-xs text-gray-300">Loading attribution...</div>
                </div>
            </div>
        </div>

    </div>

    <script>
        async function fetchFleetData() {
            const icon = document.getElementById('refreshIcon');
            icon.classList.add('fa-spin');
            try {
                const res = await fetch('/api/fleet');
                const data = await res.json();
                renderDashboard(data);
            } catch (err) {
                console.error("Error fetching fleet metrics:", err);
            } finally {
                setTimeout(() => icon.classList.remove('fa-spin'), 500);
            }
        }

        function renderDashboard(data) {
            document.getElementById('lastUpdated').innerText = 'Last updated: ' + data.timestamp;
            const s = data.summary;
            document.getElementById('kpiBalance').innerText = '$' + s.total_current.toFixed(2);
            
            const returnElem = document.getElementById('kpiReturn');
            const sign = s.total_net >= 0 ? '+' : '';
            returnElem.innerText = `${sign}${s.fleet_roi.toFixed(2)}% (${sign}$${s.total_net.toFixed(2)})`;
            returnElem.className = s.total_net >= 0 ? "text-2xl font-bold text-emerald-400" : "text-2xl font-bold text-rose-400";

            document.getElementById('kpiWinRate').innerText = `${s.overall_win_rate.toFixed(1)}% (${s.total_trades} Trades)`;
            document.getElementById('kpiOnline').innerText = `${s.online_bots} / 10 Active`;
            document.getElementById('kpiActivePositions').innerText = `${s.active_positions} in active trades`;
            const test = s.test;
            const market = test.market_data || {};
            const feedStatus = market.stale === undefined ? 'checking' : (market.stale ? 'STALE' : 'fresh');
            const fast = market.fast_volatility || {};
            const riskText = test.guard_triggered
                ? `ENTRIES PAUSED: ${test.guard_reason}`
                : test.daily_pause_until_utc
                  ? `DAILY LOSS PAUSE until ${test.daily_pause_until_utc}: ${test.daily_pause_reason}`
                  : test.market_data_paused
                    ? `ENTRIES PAUSED: ${test.market_data_pause_reason}`
                    : test.fast_volatility_paused || test.fast_volatility_data_paused
                    ? `FAST BTC GUARD: ${test.fast_volatility_pause_reason || test.fast_volatility_data_pause_reason} · calm candles ${test.fast_volatility_recovery_count}/6`
                    : test.volatility_paused
                      ? `VOLATILITY PAUSE: ${test.volatility_pause_reason} · calm candles ${test.volatility_recovery_count}/2`
                : `30-day paper test · ${market.exchange || 'Binance'} ${market.pair || 'BTC/USDT'} ${market.timeframe || '1h'} candles ${feedStatus} · ` +
                  `daily fleet loss cap $20 · max drawdown 10% · ` +
                  `5m pause at ±1.5% move or 2% range (recover after 6 calm candles); 1h pause at ±3% move or 4% range · ` +
                  `${Number(test.days_remaining || 0).toFixed(1)} days remaining · ` +
                  `daily P&L $${Number(test.daily_profit || 0).toFixed(2)} · ` +
                  `drawdown ${Number(test.drawdown_pct || 0).toFixed(2)}% · ` +
                  `BTC buy & hold ${test.benchmark_return_pct == null ? 'setting baseline' : Number(test.benchmark_return_pct).toFixed(2) + '%'}`;
            document.getElementById('riskStatus').innerText = riskText;
            const riskElement = document.getElementById('riskStatus');
            const alertBanner = document.getElementById('alertBanner');
            if (test.guard_triggered || test.daily_pause_until_utc || test.market_data_paused || test.fast_volatility_paused || test.fast_volatility_data_paused || test.volatility_paused || market.stale) {
                riskElement.className = 'text-sm text-rose-300 px-1 font-semibold';
                alertBanner.className = 'rounded-lg border border-rose-500/50 bg-rose-950/60 px-4 py-3 text-sm text-rose-200';
                alertBanner.innerText = test.recent_alerts?.slice(-1)[0] || riskText;
            } else {
                riskElement.className = 'text-sm text-gray-300 px-1';
                alertBanner.className = 'hidden rounded-lg border px-4 py-3 text-sm';
                alertBanner.innerText = '';
            }

            const events = test.volatility_events || [];
            document.getElementById('volatilityEventCount').textContent = `${test.volatility_event_count || events.length} events`;
            const eventList = document.getElementById('volatilityEvents');
            eventList.replaceChildren();
            if (!events.length) {
                eventList.textContent = 'No qualifying volatility pauses recorded yet.';
            } else {
                [...events].reverse().forEach(event => {
                    const line = document.createElement('div');
                    const outcome = ['15m', '30m', '60m'].map(window => {
                        const value = event[`btc_return_after_${window}`];
                        return `${window} ${value == null ? 'pending' : `${Number(value).toFixed(2)}%`}`;
                    }).join(' · ');
                    line.textContent = `${event.tier || 'BTC'} · ${event.triggered_at_utc || ''} · move ${Number(event.move_pct || 0).toFixed(2)}% / range ${Number(event.range_pct || 0).toFixed(2)}% · ${event.recovered_at_utc ? 'recovered' : 'pause active'} · BTC after trigger: ${outcome}`;
                    eventList.appendChild(line);
                });
            }

            const tbody = document.getElementById('leaderboardBody');
            tbody.innerHTML = '';

            const chartContainer = document.getElementById('chartBars');
            chartContainer.innerHTML = '';

            data.leaderboard.forEach((b, idx) => {
                const tr = document.createElement('tr');
                tr.className = "hover:bg-gray-800/40 transition";
                
                const pnlSign = b.net_profit >= 0 ? '+' : '';
                const pnlColor = b.net_profit >= 0 ? "text-emerald-400" : "text-rose-400";
                
                let stateBadge = `<span class="px-2 py-0.5 rounded text-xs bg-gray-800 text-gray-400 border border-gray-700">Scanning</span>`;
                if (b.has_open_trade) {
                    stateBadge = `<span class="px-2 py-0.5 rounded text-xs bg-yellow-500/20 text-yellow-300 border border-yellow-500/40 animate-pulse font-semibold">HOLDING BTC</span>`;
                }

                const onlineDot = b.online ? '<span class="w-2 h-2 rounded-full bg-emerald-400 inline-block mr-1"></span>' : '<span class="w-2 h-2 rounded-full bg-red-500 inline-block mr-1"></span>';
                const quality = b.total_trades
                    ? `Sampled DD -${Number(b.max_drawdown_pct).toFixed(2)}% · PF ${b.profit_factor == null ? '--' : Number(b.profit_factor).toFixed(2)} · Exp $${Number(b.expectancy).toFixed(2)} · Fees $${Number(b.total_fees).toFixed(2)} · Hold ${Number(b.average_hold_hours).toFixed(1)}h · Exposure ${Number(b.exposure_pct).toFixed(1)}% · Slip avg/p95 ${b.average_slippage_bps == null ? '--' : Number(b.average_slippage_bps).toFixed(1) + '/' + (b.p95_slippage_bps == null ? '--' : Number(b.p95_slippage_bps).toFixed(1)) + ' bps'}`
                    : 'Waiting for closed trades';

                tr.innerHTML = `
                    <td class="py-4 px-4 font-bold ${idx === 0 ? 'text-yellow-400' : 'text-gray-400'}">#${idx + 1}</td>
                    <td class="py-4 px-4">
                        <div class="font-semibold text-white">${b.name}</div>
                        <div class="text-xs text-gray-500">${b.desc}</div>
                    </td>
                    <td class="py-4 px-4 text-xs text-gray-300">${b.type}</td>
                    <td class="py-4 px-4 font-medium text-white">$${b.current_balance.toFixed(2)}</td>
                    <td class="py-4 px-4 font-bold ${pnlColor}">${pnlSign}$${b.net_profit.toFixed(2)}</td>
                    <td class="py-4 px-4 font-bold ${pnlColor}">${pnlSign}${b.roi_pct.toFixed(2)}%</td>
                    <td class="py-4 px-4 text-gray-300">${b.total_trades}</td>
                    <td class="py-4 px-4 text-gray-300">${b.win_rate.toFixed(1)}% <span class="text-xs text-gray-500">(${b.wins}W / ${b.losses}L)</span></td>
                    <td class="py-4 px-4 text-xs text-gray-300 whitespace-nowrap">${quality}</td>
                    <td class="py-4 px-4">${stateBadge}</td>
                    <td class="py-4 px-4 text-right">
                        <a href="http://127.0.0.1:${b.port}" target="_blank" class="inline-flex items-center gap-1 text-xs px-2.5 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-blue-400 hover:text-blue-300 font-mono transition">
                            :${b.port} <i class="fa-solid fa-arrow-up-right-from-square text-[10px]"></i>
                        </a>
                    </td>
                `;
                tbody.appendChild(tr);

                // Chart row
                const barWidth = Math.min(Math.max((b.current_balance / 100) * 50, 10), 100);
                const barColor = b.net_profit >= 0 ? "bg-emerald-500" : "bg-rose-500";
                const chartRow = document.createElement('div');
                chartRow.innerHTML = `
                    <div class="flex justify-between text-xs mb-1">
                        <span class="font-medium text-gray-300">${b.name} (${b.type})</span>
                        <span class="${pnlColor} font-bold">${pnlSign}${b.roi_pct.toFixed(2)}% ($${b.current_balance.toFixed(2)})</span>
                    </div>
                    <div class="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
                        <div class="${barColor} h-2 rounded-full" style="width: ${barWidth}%"></div>
                    </div>
                `;
                chartContainer.appendChild(chartRow);
            });
        }

        async function saveSnapshot() {
            try {
                const res = await fetch('/api/snapshot', { method: 'POST' });
                const json = await res.json();
                alert(json.message);
            } catch (e) {
                alert("Failed to save snapshot.");
            }
        }

        function switchTab(tabName) {
            ['fleet', 'market', 'risk', 'backtest', 'ml'].forEach(t => {
                document.getElementById('tab-' + t).classList.toggle('hidden', t !== tabName);
                const btn = document.getElementById('btn-tab-' + t);
                btn.className = t === tabName
                    ? 'tab-btn px-4 py-2 rounded-lg text-sm font-medium bg-blue-600 text-white'
                    : 'tab-btn px-4 py-2 rounded-lg text-sm font-medium bg-gray-800 text-gray-300 hover:bg-gray-700';
            });
            if (tabName === 'market') loadMarketAndSignals();
            if (tabName === 'risk') loadRiskStatus();
            if (tabName === 'backtest') loadBacktestHistory();
            if (tabName === 'ml') loadMlAndAttribution();
        }

        async function loadMarketAndSignals() {
            try {
                const [mRes, sRes] = await Promise.all([
                    fetch('/api/engine/market-overview').then(r => r.json()),
                    fetch('/api/engine/signals').then(r => r.json())
                ]);
                document.getElementById('engPrice').innerText = `$${Number(mRes.price).toLocaleString()}`;
                document.getElementById('engChange24h').innerText = `24h: ${mRes.change_24h_pct >= 0 ? '+' : ''}${mRes.change_24h_pct}%`;
                document.getElementById('engRegime').innerText = mRes.regime.regime;
                document.getElementById('engRegimeConf').innerText = `Confidence: ${mRes.regime.confidence_score}%`;
                document.getElementById('engMtfBias').innerText = `${mRes.mtf.daily_bias} / ${mRes.mtf.h4_bias} / ${mRes.mtf.h1_bias}`;
                document.getElementById('engMtfScore').innerText = `Alignment: ${mRes.mtf.alignment_score}/100 (${mRes.mtf.decision})`;
                const ind = mRes.indicators_1h;
                document.getElementById('engIndSummary').innerText = `RSI: ${ind.rsi_14} · ATR%: ${ind.atr_pct}% · ADX: ${ind.adx_14} · Vol: ${ind.relative_volume}x`;
                document.getElementById('engStrategies').innerHTML = sRes.strategies.map(s =>
                    `<div class="flex justify-between p-2 rounded bg-gray-900/60 border border-gray-800">
                        <span><strong>${s.name}</strong> (${s.type})</span>
                        <span class="${s.eligible ? 'text-emerald-400' : 'text-gray-500'}">${s.enabled ? (s.eligible ? 'ELIGIBLE' : 'STANDBY') : 'DISABLED'}</span>
                    </div>`
                ).join('');
                document.getElementById('engSignalExplanation').innerText = sRes.current.explanation || 'No explanation';
            } catch (e) { console.error(e); }
        }

        async function loadRiskStatus() {
            try {
                const r = await fetch('/api/engine/risk').then(res => res.json());
                document.getElementById('engRiskGrid').innerHTML = `
                    <div class="p-3 rounded bg-gray-900/70 border border-gray-800"><div class="text-xs text-gray-400">Risk / Trade</div><div class="font-bold text-white">${r.risk_per_trade_pct}%</div></div>
                    <div class="p-3 rounded bg-gray-900/70 border border-gray-800"><div class="text-xs text-gray-400">Daily P&L</div><div class="font-bold text-white">$${r.daily_pnl} (${r.daily_pnl_pct}%)</div></div>
                    <div class="p-3 rounded bg-gray-900/70 border border-gray-800"><div class="text-xs text-gray-400">Drawdown / Limit</div><div class="font-bold text-white">${r.current_drawdown_pct}% / ${r.max_drawdown_limit_pct}%</div></div>
                    <div class="p-3 rounded bg-gray-900/70 border border-gray-800"><div class="text-xs text-gray-400">Kill Switch</div><div class="font-bold ${r.manual_kill_switch ? 'text-rose-400' : 'text-emerald-400'}">${r.manual_kill_switch ? 'ACTIVE' : 'OFF'}</div></div>
                `;
                document.getElementById('engRiskLockouts').innerText = r.active_lockouts.length ? `Active Lockouts: ${r.active_lockouts.join(' | ')}` : 'All risk checks healthy. No active lockouts.';
            } catch (e) { console.error(e); }
        }

        async function toggleKillSwitch(active) {
            await fetch('/api/engine/risk/kill-switch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ active, reason: 'Dashboard operator action' })
            });
            loadRiskStatus();
        }

        async function runEngineBacktest() {
            document.getElementById('engBacktestStatus').innerText = 'Starting background backtest...';
            const res = await fetch('/api/engine/backtest/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            }).then(r => r.json());
            document.getElementById('engBacktestStatus').innerText = `Job ${res.run_id} queued. Refreshing shortly...`;
            setTimeout(loadBacktestHistory, 4000);
        }

        async function loadBacktestHistory() {
            try {
                const data = await fetch('/api/engine/backtest/history').then(r => r.json());
                const runs = data.runs || [];
                if (!runs.length) {
                    document.getElementById('engBacktestResults').innerText = 'No backtest runs stored yet. Click "Run BTC Backtest" to start.';
                    return;
                }
                document.getElementById('engBacktestResults').innerHTML = runs.map(run => {
                    const m = run.metrics;
                    return `<div class="p-4 rounded-xl bg-gray-900/70 border border-gray-800 space-y-2">
                        <div class="flex justify-between font-bold text-white"><span>${run.run_id} (${run.strategy_name})</span><span>Return: ${m.total_return_pct}% | Buy&amp;Hold: ${run.benchmarks.btc_buy_hold_pct}%</span></div>
                        <div class="text-gray-400">Trades: ${m.total_trades} · Win Rate: ${m.win_rate_pct}% · PF: ${m.profit_factor} · Max DD: ${m.max_drawdown_pct}% · Sharpe: ${m.sharpe_ratio} · Sortino: ${m.sortino_ratio} · Expectancy: $${m.expectancy}</div>
                    </div>`;
                }).join('');
            } catch (e) { console.error(e); }
        }

        async function loadMlAndAttribution() {
            try {
                const [ml, attr] = await Promise.all([
                    fetch('/api/engine/ml-status').then(r => r.json()),
                    fetch('/api/engine/attribution').then(r => r.json())
                ]);
                document.getElementById('engMlStatus').innerText = ml.implemented
                    ? `Active Model: ${ml.active_model.version} (${ml.active_model.model_type}) · Trained: ${ml.active_model.trained_at_utc} · Precision: ${ml.active_model.precision_score} · Brier: ${ml.active_model.brier_score}`
                    : 'ML Signal Filter not yet trained.';
                document.getElementById('engAttribution').innerText = `Closed Trades: ${attr.total_closed_trades}\nBy Strategy: ${JSON.stringify(attr.by_strategy)}\nBy Regime: ${JSON.stringify(attr.by_regime)}`;
            } catch (e) { console.error(e); }
        }

        fetchFleetData();
        setInterval(fetchFleetData, 10000);
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

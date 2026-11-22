import argparse
import json
import os
import secrets

strategies = [
    ('01', 'BTCSwing_EMATrend', 8081),
    ('02', 'BTCSwing_SuperTrend', 8082),
    ('03', 'BTCSwing_MACDDivergence', 8083),
    ('04', 'BTCSwing_SqueezeBreakout', 8084),
    ('05', 'BTCSwing_DonchianTurtle', 8085),
    ('06', 'BTCSwing_IchimokuCloud', 8086),
    ('07', 'BTCSwing_RSI_VWAP', 8087),
    ('08', 'BTCSwing_EMARibbon', 8088),
    ('09', 'BTCSwing_StochRSI_CMF', 8089),
    ('10', 'BTCSwing_FibonacciBands', 8090)
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "user_data", "configs")
DB_DIR = os.path.join(PROJECT_ROOT, "user_data", "db")

parser = argparse.ArgumentParser(description="Generate the 10 BTC paper-trading bot configs.")
parser.add_argument(
    "--force",
    action="store_true",
    help="Replace existing bot configs. Do not use while a fleet is running.",
)
args = parser.parse_args()

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

existing = [
    os.path.join(CONFIG_DIR, f"config_btc_{idx}.json")
    for idx, _, _ in strategies
    if os.path.exists(os.path.join(CONFIG_DIR, f"config_btc_{idx}.json"))
]
if existing and not args.force:
    raise SystemExit(
        "Refusing to overwrite existing bot configs. Stop the fleet, review the impact, "
        "then rerun with --force if replacement is intended."
    )

for idx, strat_name, port in strategies:
    cfg = {
        "$schema": "https://schema.freqtrade.io/schema.json",
        "bot_name": f"btc_swing_{idx}_{strat_name}",
        "dry_run": True,
        "dry_run_wallet": 100,
        "stake_currency": "USDT",
        "stake_amount": "unlimited",
        "max_open_trades": 1,
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "USD",
        "timeframe": "1h",
        "cancel_open_orders_on_exit": False,
        "unfilledtimeout": {
            "entry": 60,
            "exit": 60,
            "exit_timeout_count": 0,
            "unit": "minutes"
        },
        "entry_pricing": {
            "price_side": "same",
            "use_order_book": True,
            "order_book_top": 1,
            "price_last_balance": 0.0,
            "check_depth_of_market": {
                "enabled": False,
                "bids_to_ask_delta": 1
            }
        },
        "exit_pricing": {
            "price_side": "same",
            "use_order_book": True,
            "order_book_top": 1
        },
        "exchange": {
            "name": "binance",
            "key": "",
            "secret": "",
            "ccxt_config": {},
            "ccxt_async_config": {"enable_ws": True},
            "pair_whitelist": ["BTC/USDT"],
            "pair_blacklist": []
        },
        "pairlists": [
            {"method": "StaticPairList"}
        ],
        "telegram": {
            "enabled": False,
            "token": "",
            "chat_id": ""
        },
        "api_server": {
            "enabled": True,
            "listen_ip_address": "127.0.0.1",
            "listen_port": port,
            "verbosity": "error",
            "jwt_secret_key": secrets.token_urlsafe(32),
            "CORS_origins": [],
            "username": "admin",
            "password": secrets.token_urlsafe(24)
        },
        "db_url": f"sqlite:///{os.path.join(DB_DIR, f'btc_{idx}.sqlite').replace(os.sep, '/')}",
        "initial_state": "running",
        "force_entry_enable": False,
        "internals": {
            "process_throttle_secs": 5
        }
    }
    filepath = os.path.join(CONFIG_DIR, f"config_btc_{idx}.json")
    with open(filepath, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"Generated {filepath} on port {port}")

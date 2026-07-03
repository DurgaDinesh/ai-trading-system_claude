"""
Quick setup and validation script.
Run: python setup.py
"""

import subprocess
import sys
from pathlib import Path


def check_python():
    if sys.version_info < (3, 11):
        print(f"❌ Python 3.11+ required. Current: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]}")


def check_env():
    env_file = Path("config/.env")
    if not env_file.exists():
        print("❌ config/.env not found. Copy config/.env.example → config/.env and fill values.")
        sys.exit(1)
    from dotenv import load_dotenv
    load_dotenv(env_file)
    import os
    required = ["KITE_API_KEY", "KITE_API_SECRET", "TELEGRAM_BOT_TOKEN",
                "TELEGRAM_CHAT_ID", "VAULT_ENCRYPTION_KEY", "TRADE_PIN_HASH",
                "DASHBOARD_SECRET_KEY", "ANTHROPIC_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"❌ Missing env vars: {', '.join(missing)}")
        sys.exit(1)
    print(f"✅ All {len(required)} required env vars present")


def check_dirs():
    dirs = ["data", "logs", "reports", "reports/backtests", "data/models"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Required directories created")


def init_db():
    from database.trade_journal import init_db as _init
    _init()
    print("✅ Database initialized (data/trading.db)")


def verify_telegram():
    import os
    from dotenv import load_dotenv
    load_dotenv("config/.env")
    import requests
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("⚠️  Telegram token not set — skipping")
        return
    resp = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ Telegram Bot: @{data['result']['username']}")
    else:
        print(f"❌ Telegram token invalid: {resp.text[:100]}")


if __name__ == "__main__":
    print("\n🎯 NiftySniper Setup Verification\n" + "─" * 40)
    check_python()
    check_dirs()

    print("\nChecking environment...")
    try:
        check_env()
    except SystemExit:
        print("   Run: python main.py --setup to generate credentials")
        sys.exit(1)

    print("\nInitializing database...")
    init_db()

    print("\nVerifying Telegram...")
    verify_telegram()

    print("\n" + "─" * 40)
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("  1. Run backtesting:  python main.py --backtest")
    print("  2. Start paper trading: python main.py")
    print("  3. Open dashboard: http://localhost:8000")
    print("  4. Login with your 6-digit PIN\n")

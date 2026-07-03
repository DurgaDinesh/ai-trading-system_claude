"""
NiftySniper — Main Entry Point
Initializes all subsystems and starts the bot.

Usage:
  python main.py                     → Start the full trading bot + dashboard
  python main.py --backtest          → Run backtesting only
  python main.py --setup             → First-time setup (generate PIN hash)
  python main.py --dashboard-only    → Start dashboard only (no trading)
"""

import sys
import io
import signal
import argparse
import structlog
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 output on Windows to handle special characters
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Load environment variables first
load_dotenv(Path("config/.env"))

logger = structlog.get_logger(__name__)


def setup_logging():
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
    )


def first_time_setup():
    """Interactive setup wizard for new installations."""
    print("\n" + "=" * 60)
    print("  NiftySniper - First Time Setup")
    print("=" * 60)

    print("\n1. Generating vault encryption key...")
    from cryptography.fernet import Fernet
    key = Fernet.generate_key().decode()
    print(f"   Add to config/.env:\n   VAULT_ENCRYPTION_KEY={key}")

    print("\n2. Generating PIN hash...")
    pin = input("   Enter your desired 6-digit trading PIN: ").strip()
    if not pin.isdigit() or len(pin) != 6:
        print("   ERROR: PIN must be exactly 6 digits")
        sys.exit(1)
    from core.security.pin_auth import generate_pin_hash
    pin_hash = generate_pin_hash(pin)
    print(f"   Add to config/.env:\n   TRADE_PIN_HASH={pin_hash}")

    print("\n3. Generating dashboard secret key...")
    import secrets
    dash_key = secrets.token_hex(32)
    print(f"   Add to config/.env:\n   DASHBOARD_SECRET_KEY={dash_key}")

    print("\nSetup complete. Fill in all remaining values in config/.env")
    print("   See config/.env.example for the full list of required variables.\n")


def run_backtest_cli():
    """Run backtesting from command line."""
    print("\n🔄 NiftySniper Backtesting Suite\n")
    from backtesting.report import run_full_pre_live_validation
    approved = run_full_pre_live_validation("NIFTY")
    sys.exit(0 if approved else 1)


def run_dashboard_only():
    """Start only the web dashboard."""
    import uvicorn
    from database.trade_journal import init_db
    init_db()
    cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))["dashboard"]
    logger.info("starting_dashboard_only", host=cfg["host"], port=cfg["port"])
    uvicorn.run("dashboard.app:app", host=cfg["host"], port=cfg["port"], reload=False)


_SINGLETON_LOCK_PORT = 47653  # arbitrary unused port, used purely as a cross-process mutex


def _acquire_singleton_lock():
    """Bind a localhost-only socket as a mutex so a second instance can't start.

    A second `python main.py` launched while one is already running used to
    silently double-poll Telegram (409 Conflict) and crash on the port-8000
    bind failure only *after* re-running the whole startup sequence. Binding
    this first, before any other init, makes a duplicate launch fail
    instantly with a clear message. The OS releases the port automatically
    on process exit — even a crash or force-kill — so there's no stale lock
    file to clean up.
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", _SINGLETON_LOCK_PORT))
    except OSError:
        print(
            "\nERROR: NiftySniper appears to already be running "
            f"(singleton lock port {_SINGLETON_LOCK_PORT} is in use).\n"
            "Refusing to start a second instance — stop the existing one first.\n"
        )
        sys.exit(1)
    return sock


def run_full_bot():
    """Start the complete trading bot system."""
    _singleton_lock = _acquire_singleton_lock()  # noqa: F841 — kept alive for process lifetime
    cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))

    print("\n" + "=" * 60)
    print(f"  NiftySniper v{cfg['system']['version']}")
    print(f"  Mode: {cfg['system']['mode'].upper()}")
    print(f"  Capital: Rs.{cfg['capital']['total']:,.0f}")
    print("=" * 60 + "\n")

    # ── Initialize Database ────────────────────────────────────────────────
    logger.info("initializing_database")
    from database.trade_journal import init_db
    init_db()

    # ── Initialize Kite Connect ────────────────────────────────────────────
    from core.market_data.kite_client import kite_client
    if cfg["system"]["mode"] == "live":
        logger.info("connecting_kite")
        kite_client.connect()
    else:
        # Paper mode never blocks on the interactive login flow. If a cached
        # token or KITE_REQUEST_TOKEN happens to be available, use real Kite
        # chain/LTP data; otherwise fall back to yfinance spot + Black-Scholes
        # estimated premiums (see options_analytics.py / position_manager.py).
        if kite_client.connect_if_possible():
            logger.info("kite_connected_for_paper_mode")
        else:
            logger.info("kite_unavailable_paper_mode_using_black_scholes_fallback")

    # ── Start Telegram Bot ─────────────────────────────────────────────────
    logger.info("starting_telegram_bot")
    from notifications.telegram_bot import telegram
    from core.execution.risk_manager import risk_manager

    def emergency_stop():
        risk_manager.halt("Emergency stop via Telegram /STOP")
        from core.execution.position_manager import position_manager
        mode = cfg["system"]["mode"]
        prices = position_manager.get_current_prices_for_mode(mode)
        if mode == "paper":
            from core.execution.paper_trader import paper_trader
            paper_trader.force_squareoff_all(prices)
        else:
            from core.execution.order_manager import order_manager
            position_manager.squareoff_all(prices, mode="live")
            order_manager.cancel_all_pending()
        from database.trade_journal import journal
        journal.log_event("EMERGENCY_STOP", "Triggered via Telegram /STOP", severity="CRITICAL")

    from notifications.telegram_bot import set_emergency_stop_callback
    import time
    set_emergency_stop_callback(emergency_stop)
    telegram.start()
    time.sleep(3)   # allow bot thread to finish initializing

    # ── Start WebSocket Feed (live mode only) ──────────────────────────────
    if cfg["system"]["mode"] == "live":
        logger.info("starting_websocket_feed")
        from core.market_data.websocket_feed import live_feed
        live_feed.start()

    # ── Start APScheduler ──────────────────────────────────────────────────
    logger.info("starting_scheduler")
    from scheduler.job_scheduler import create_scheduler
    scheduler = create_scheduler()
    scheduler.start()

    # ── Graceful shutdown handler ──────────────────────────────────────────
    def shutdown(signum, frame):
        logger.info("shutdown_signal_received")
        scheduler.shutdown(wait=False)
        if cfg["system"]["mode"] == "live":
            from core.market_data.websocket_feed import live_feed
            live_feed.stop()
        telegram.send_alert("🛑 NiftySniper shutting down.", severity="WARNING")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Start Dashboard ────────────────────────────────────────────────────
    logger.info("starting_dashboard", port=cfg["dashboard"]["port"])
    telegram.send_alert(
        f"🚀 NiftySniper started | Mode: {cfg['system']['mode'].upper()} | "
        f"Dashboard: http://localhost:{cfg['dashboard']['port']}",
        severity="INFO"
    )

    import uvicorn
    uvicorn.run(
        "dashboard.app:app",
        host=cfg["dashboard"]["host"],
        port=cfg["dashboard"]["port"],
        reload=False,
        log_level="warning",
    )


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="NiftySniper Algo Trading Bot")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting suite")
    parser.add_argument("--setup", action="store_true", help="First-time setup wizard")
    parser.add_argument("--dashboard-only", action="store_true", help="Start dashboard only")
    args = parser.parse_args()

    if args.setup:
        first_time_setup()
    elif args.backtest:
        run_backtest_cli()
    elif args.dashboard_only:
        run_dashboard_only()
    else:
        run_full_bot()


if __name__ == "__main__":
    main()

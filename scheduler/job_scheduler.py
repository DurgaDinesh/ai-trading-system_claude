"""
APScheduler job definitions.
All timed tasks: pre-market scan, signal loop, square-off, reports.
"""

import pytz
from datetime import datetime, date
from typing import Optional
import structlog
import yaml

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))
IST = pytz.timezone("Asia/Kolkata")

# State shared between jobs
_global_context: dict = {}
_news_sentiment: dict = {}
_options_context: dict = {}
_nifty_prev_close: Optional[float] = None
_system_mode: str = _cfg["system"]["mode"]


def _get_mode() -> str:
    """Dynamically read current mode (can be changed at runtime via dashboard)."""
    try:
        from dashboard.app import _system_mode as dash_mode
        return dash_mode
    except ImportError:
        return _system_mode


def _is_paused() -> bool:
    try:
        from dashboard.app import _bot_paused
        return _bot_paused
    except ImportError:
        return False


# ── Job Definitions ────────────────────────────────────────────────────────────

def job_pre_market_analysis():
    """9:00 AM — Fetch all global market data before market opens."""
    global _global_context, _news_sentiment, _nifty_prev_close
    logger.info("job_pre_market_analysis_starting")

    try:
        from core.analysis.global_market import get_full_global_context
        from core.analysis.news_sentiment import get_market_sentiment
        from core.market_data.historical import fetch_historical_yfinance
        from notifications.telegram_bot import telegram

        # Get yesterday's Nifty close for gap calculation
        df = fetch_historical_yfinance("NIFTY", period="5d", interval="1d")
        if not df.empty:
            _nifty_prev_close = float(df["close"].iloc[-1])

        _global_context = get_full_global_context(_nifty_prev_close)
        _news_sentiment = get_market_sentiment()

        # Send pre-market brief to Telegram
        vix = _global_context.get("india_vix")
        fii = _global_context.get("fii_net_cash_cr")
        gap = _global_context.get("gift_nifty_gap_pct")
        sentiment = _news_sentiment.get("score", 0)

        brief = (
            f"🌅 <b>Pre-Market Brief</b> | {datetime.now(IST).strftime('%d %b %Y')}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🌐 Gift Nifty Gap: {f'{gap:+.2f}%' if gap else 'N/A'}\n"
            f"📊 India VIX: {f'{vix:.1f}' if vix else 'N/A'}\n"
            f"💹 FII Net: {f'₹{fii:+,.0f}Cr' if fii else 'N/A'}\n"
            f"📰 Sentiment: {f'{sentiment:+.2f}' if sentiment else '0.00'}\n"
            f"🌍 Global Score: {_global_context.get('global_score', 0)}"
        )
        telegram.send_alert(brief, severity="INFO")
        logger.info("pre_market_analysis_complete", global_score=_global_context.get("global_score"))

    except Exception as e:
        logger.error("pre_market_analysis_failed", error=str(e))


def job_signal_scan():
    """
    9:30 AM – 2:30 PM every 5 minutes — Main signal generation loop.
    Fetches live data, runs analysis, generates and executes signals.
    """
    if _is_paused():
        return

    from core.execution.risk_manager import risk_manager
    if risk_manager.is_halted:
        return

    try:
        from core.market_data.historical import fetch_historical_yfinance
        from core.analysis.technical import compute_all
        from core.analysis.options_analytics import get_options_analytics_summary
        from core.signals.regime_detector import detect_regime
        from core.signals.signal_engine import generate_signal
        from core.signals.strategy_selector import resolve_tradeable_instrument, get_next_thursday_expiry
        from core.execution.paper_trader import paper_trader
        from core.execution.order_manager import order_manager
        from core.execution.position_manager import position_manager
        from core.learning.performance_tracker import performance_tracker
        from notifications.telegram_bot import telegram

        mode = _get_mode()
        capital_cfg = _cfg["capital"]
        total_capital = capital_cfg["total"]

        # Check max positions
        from database.trade_journal import journal
        from database.models import TradingMode
        open_trades = journal.get_open_trades(TradingMode(mode.upper()))
        if len(open_trades) >= capital_cfg["max_open_positions"]:
            logger.debug("max_positions_reached", count=len(open_trades))
            return

        # Get current Nifty data (5-minute bars, last 200 bars)
        df = fetch_historical_yfinance("NIFTY", period="10d", interval="5m")
        if df.empty or len(df) < 50:
            logger.warning("insufficient_data_for_signal")
            return

        df = compute_all(df)
        spot_price = float(df["close"].iloc[-1])

        # Fetch options context
        expiry = get_next_thursday_expiry()
        options_ctx = get_options_analytics_summary("NIFTY", expiry, spot_price)

        # Regime detection
        regime = detect_regime(df, _global_context, _news_sentiment, options_ctx)

        # Signal generation
        signal = generate_signal(
            df_5m=df,
            regime=regime,
            options_context=options_ctx,
            global_context=_global_context,
            news_sentiment=_news_sentiment,
            spot_price=spot_price,
            available_capital=total_capital,
        )

        if not signal.is_valid:
            logger.debug("no_valid_signal", reason=signal.invalidation_reason)
            return

        # Resolve instrument
        instrument = resolve_tradeable_instrument(
            signal=signal, regime=regime, underlying="NIFTY",
            spot_price=spot_price, expiry=expiry,
        )
        if not instrument:
            return

        # Approval workflow
        if _cfg["approval"]["telegram_approval_enabled"]:
            telegram.send_signal_alert(signal, instrument)
            approved = telegram.send_approval_request(
                signal, instrument,
                timeout_seconds=_cfg["approval"]["approval_timeout_seconds"]
            )
            if not approved:
                logger.info("signal_skipped_no_approval")
                return

        # Execute
        if mode == "paper":
            trade_id = paper_trader.place_order(signal, instrument, _global_context, _news_sentiment)
        else:
            trade_id = order_manager.place_order(
                signal, instrument, _global_context, _news_sentiment,
                approved_by="HUMAN"
            )

        if trade_id:
            telegram.send_trade_executed(
                trade_id, instrument["tradingsymbol"],
                instrument.get("option_ltp", 0), instrument.get("quantity", 0), mode
            )

    except Exception as e:
        logger.error("signal_scan_failed", error=str(e))


def job_monitor_positions():
    """
    Every 1 minute — Check open positions for SL/TP hits.
    Deliberately NOT gated on _is_paused(): pausing stops new trade entries
    (see job_signal_scan) but must never stop protecting already-open
    positions — otherwise a paused bot leaves live risk unmonitored.
    """
    try:
        from core.execution.position_manager import position_manager
        from database.trade_journal import journal
        from database.models import TradingMode

        mode = _get_mode()
        open_trades = journal.get_open_trades(TradingMode(mode.upper()))
        if not open_trades:
            return

        # SL/TP are premium-based, so positions must be checked against the
        # option's own real LTP — never a spot-index proxy.
        current_prices = position_manager.get_current_prices_for_mode(mode)
        if not current_prices:
            return
        position_manager.on_candle(current_prices, mode)

    except Exception as e:
        logger.error("position_monitor_failed", error=str(e))


def job_daily_squareoff():
    """3:10 PM — Force close all open positions."""
    try:
        from core.execution.position_manager import position_manager
        from database.trade_journal import journal
        from database.models import TradingMode
        from notifications.telegram_bot import telegram

        mode = _get_mode()
        open_trades = journal.get_open_trades(TradingMode(mode.upper()))
        position_manager.squareoff_all({}, mode)
        telegram.send_alert(f"📢 Auto square-off complete. {len(open_trades)} positions closed.", "INFO")
        logger.info("daily_squareoff_complete", count=len(open_trades))

    except Exception as e:
        logger.error("daily_squareoff_failed", error=str(e))


def job_daily_report():
    """3:30 PM — Generate daily P&L report and send to Telegram."""
    try:
        from notifications.report_generator import generate_daily_csv, generate_daily_pdf
        from notifications.telegram_bot import telegram
        from core.execution.risk_manager import risk_manager
        from database.trade_journal import journal

        today = date.today()
        mode = _get_mode()
        journal.compute_and_save_daily_pnl(today, mode)

        stats = risk_manager.get_daily_stats()
        pnl_data = {
            "daily_realized": stats["daily_realized_pnl"],
            "total_unrealized": 0,
            "open_positions": [],
        }
        telegram.send_daily_summary(pnl_data)

        generate_daily_csv(today, mode)
        generate_daily_pdf(today, mode)
        logger.info("daily_report_generated", date=str(today))

    except Exception as e:
        logger.error("daily_report_failed", error=str(e))


def job_weekly_report():
    """Friday 4 PM — Weekly performance summary."""
    try:
        from notifications.report_generator import generate_weekly_summary
        from notifications.telegram_bot import telegram
        from core.learning.performance_tracker import performance_tracker

        mode = _get_mode()
        summary = generate_weekly_summary(date.today(), mode)
        stats = summary["stats"]
        readiness = performance_tracker.check_live_escalation_readiness()

        msg = (
            f"📅 <b>Weekly Performance Summary</b>\n"
            f"Period: {summary['period']}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Trades: {stats.get('total_trades', 0)}\n"
            f"Win Rate: {stats.get('win_rate', 0)*100:.1f}%\n"
            f"Profit Factor: {stats.get('profit_factor', 0):.2f}\n"
            f"Net P&L: ₹{stats.get('total_net_pnl', 0):+,.2f}\n"
            f"Sharpe: {stats.get('sharpe_approx', 0):.3f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Live Readiness: {'✅ READY' if readiness['ready_for_live'] else '❌ NOT YET'}"
        )
        telegram.send_alert(msg, severity="INFO")
        logger.info("weekly_report_sent")

    except Exception as e:
        logger.error("weekly_report_failed", error=str(e))


def job_ml_retrain_check():
    """Daily 8 PM — Check if ML model needs retraining."""
    try:
        from core.learning.ml_scorer import ml_scorer
        if ml_scorer.should_train():
            logger.info("scheduled_ml_retrain_triggered")
            ml_scorer.train()
    except Exception as e:
        logger.error("ml_retrain_check_failed", error=str(e))


# ── Scheduler Setup ────────────────────────────────────────────────────────────

def create_scheduler() -> BackgroundScheduler:
    """Create and configure the APScheduler instance."""
    scheduler = BackgroundScheduler(timezone=IST)

    session = _cfg["session"]

    # Run pre-market analysis immediately on startup (in case bot started after 9 AM)
    scheduler.add_job(
        job_pre_market_analysis, "date",
        id="pre_market_startup", replace_existing=True
    )

    # Pre-market: 9:00 AM Mon–Thu
    scheduler.add_job(
        job_pre_market_analysis, CronTrigger(
            hour=9, minute=0, day_of_week="mon-thu", timezone=IST
        ), id="pre_market", replace_existing=True
    )

    # Signal scan: 9:30 AM – 2:25 PM every 5 min, Mon–Thu
    # hour 9 fires only at :30,:35,...,:55  (avoids 9:00–9:25 observation window)
    # hours 10–13 fire every 5 min
    # hour 14 fires only at :00,:05,...,:25 (market closes 14:30)
    scheduler.add_job(
        job_signal_scan, CronTrigger(
            hour="9", minute="30,35,40,45,50,55",
            day_of_week="mon-thu", timezone=IST
        ), id="signal_scan_9", replace_existing=True
    )
    scheduler.add_job(
        job_signal_scan, CronTrigger(
            hour="10,11,12,13", minute="0,5,10,15,20,25,30,35,40,45,50,55",
            day_of_week="mon-thu", timezone=IST
        ), id="signal_scan_10_13", replace_existing=True
    )
    scheduler.add_job(
        job_signal_scan, CronTrigger(
            hour="14", minute="0,5,10,15,20,25",
            day_of_week="mon-thu", timezone=IST
        ), id="signal_scan_14", replace_existing=True
    )

    # Position monitor: 9:00 AM – 3:09 PM every minute, Mon–Thu.
    # Must never overlap the 15:10 squareoff job below — both jobs run on
    # separate threads, and if they both fired at 15:10 they could each try
    # to close the same open positions concurrently (duplicate exit orders).
    scheduler.add_job(
        job_monitor_positions, CronTrigger(
            hour="9-14", minute="*", day_of_week="mon-thu", timezone=IST
        ), id="position_monitor_9_14", replace_existing=True
    )
    scheduler.add_job(
        job_monitor_positions, CronTrigger(
            hour="15", minute="0,1,2,3,4,5,6,7,8,9",
            day_of_week="mon-thu", timezone=IST
        ), id="position_monitor_15", replace_existing=True
    )

    # Auto square-off: 3:10 PM Mon–Thu (sole owner of the mandatory EOD close)
    scheduler.add_job(
        job_daily_squareoff, CronTrigger(
            hour=15, minute=10, day_of_week="mon-thu", timezone=IST
        ), id="squareoff", replace_existing=True
    )

    # Daily report: 3:30 PM Mon–Thu
    scheduler.add_job(
        job_daily_report, CronTrigger(
            hour=15, minute=30, day_of_week="mon-thu", timezone=IST
        ), id="daily_report", replace_existing=True
    )

    # Weekly report: Friday 4:00 PM
    scheduler.add_job(
        job_weekly_report, CronTrigger(
            hour=16, minute=0, day_of_week="fri", timezone=IST
        ), id="weekly_report", replace_existing=True
    )

    # ML retrain check: 8 PM daily
    scheduler.add_job(
        job_ml_retrain_check, CronTrigger(
            hour=20, minute=0, day_of_week="mon-fri", timezone=IST
        ), id="ml_retrain", replace_existing=True
    )

    logger.info("scheduler_configured", jobs=len(scheduler.get_jobs()))
    return scheduler

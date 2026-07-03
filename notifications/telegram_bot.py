"""
Telegram bot integration.
Handles: signal alerts, trade confirmations, human-approval workflow,
emergency stop, daily P&L summaries, and /STOP command.
"""

import asyncio
import threading
import uuid
from datetime import datetime
from typing import Optional, Callable
import structlog
import yaml
import pytz

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
)
from core.security.vault import vault
from core.signals.signal_engine import TradeSignal

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))
IST = pytz.timezone("Asia/Kolkata")

_approval_callbacks: dict[str, dict] = {}  # message_id → { event, approved }
_emergency_stop_callback: Optional[Callable] = None


def set_emergency_stop_callback(fn: Callable):
    """Register the function to call when /STOP is received."""
    global _emergency_stop_callback
    _emergency_stop_callback = fn


# ── Message Formatters ─────────────────────────────────────────────────────────

def _format_signal_message(signal: TradeSignal, instrument: dict) -> str:
    now_ist = datetime.now(IST).strftime("%d-%b-%Y %H:%M IST")
    direction_emoji = "🟢 CALL (CE)" if signal.direction == "CE" else "🔴 PUT (PE)"
    return (
        f"🎯 <b>TRADE SIGNAL GENERATED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {now_ist}\n"
        f"📊 Instrument: <code>{instrument.get('tradingsymbol', 'N/A')}</code>\n"
        f"Direction: {direction_emoji}\n"
        f"Strike: {instrument.get('strike', 'N/A')}\n"
        f"Expiry: {instrument.get('expiry', 'N/A')}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 <b>Entry:</b> ₹{instrument.get('option_ltp', signal.entry_price):.2f}\n"
        f"🛑 <b>Stop Loss:</b> ₹{instrument.get('stop_loss', 0):.2f}\n"
        f"🎯 <b>TP1:</b> ₹{instrument.get('tp1', 0):.2f} (40%)\n"
        f"🎯 <b>TP2:</b> ₹{instrument.get('tp2', 0):.2f} (35%)\n"
        f"🎯 <b>TP3:</b> ₹{instrument.get('tp3', 0):.2f} (25% trail)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Composite Score:</b> {signal.composite_score}/100\n"
        f"🔗 <b>Confluence:</b> {signal.confluence_count} indicators\n"
        f"✅ <b>Triggered:</b> {', '.join(signal.indicators_triggered)}\n"
        f"⚖️ <b>R:R Ratio:</b> 1:{signal.rr_ratio:.1f}\n"
        f"🌐 <b>Regime:</b> {signal.regime}\n"
        f"📦 <b>Qty:</b> {instrument.get('quantity', 0)} ({instrument.get('lots', 0)} lots)\n"
        f"💼 <b>Order Value:</b> ₹{instrument.get('order_value', 0):,.0f}"
    )


def _format_pnl_message(pnl_data: dict) -> str:
    now_ist = datetime.now(IST).strftime("%d-%b-%Y %H:%M IST")
    realized = pnl_data.get("daily_realized", 0)
    unrealized = pnl_data.get("total_unrealized", 0)
    total = realized + unrealized
    emoji = "🟢" if total >= 0 else "🔴"
    positions = pnl_data.get("open_positions", [])

    msg = (
        f"{emoji} <b>DAILY P&L SUMMARY</b>\n"
        f"📅 {now_ist}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ Realized P&L:    ₹{realized:+,.2f}\n"
        f"📊 Unrealized P&L:  ₹{unrealized:+,.2f}\n"
        f"💰 Total Day P&L:   ₹{total:+,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📂 Open Positions: {len(positions)}\n"
    )
    for pos in positions:
        pnl_emoji = "🟢" if pos["unrealized_pnl"] >= 0 else "🔴"
        msg += f"{pnl_emoji} {pos['instrument']}: ₹{pos['unrealized_pnl']:+,.2f}\n"

    if pnl_data.get("is_halted"):
        msg += "\n⚠️ <b>SYSTEM HALTED</b> — Daily loss limit reached"
    return msg


# ── Telegram Application ───────────────────────────────────────────────────────

class TelegramNotifier:

    def __init__(self):
        self._app: Optional[Application] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._chat_id = None
        self._admin_chat_id = None

    def start(self):
        """Start the Telegram bot in a background thread."""
        self._chat_id = vault.telegram_chat_id
        self._admin_chat_id = vault.telegram_admin_chat_id

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._start_async())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        logger.info("telegram_bot_started")

    async def _start_async(self):
        self._app = (
            Application.builder()
            .token(vault.telegram_bot_token)
            .build()
        )
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("STOP", self._cmd_emergency_stop))
        self._app.add_handler(CommandHandler("stop", self._cmd_emergency_stop))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("pnl", self._cmd_pnl))
        self._app.add_handler(CallbackQueryHandler(self._on_approval_callback))
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        logger.info("telegram_polling_started")
        # Keep running
        import asyncio as aio
        await aio.Event().wait()

    def _is_authorized(self, update: Update) -> bool:
        """Only the configured owner/admin chat may issue commands or approvals."""
        chat_id = update.effective_chat.id if update.effective_chat else None
        allowed = {str(self._chat_id), str(self._admin_chat_id)} - {"None"}
        return str(chat_id) in allowed

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            logger.warning("unauthorized_telegram_command", chat_id=update.effective_chat.id, cmd="start")
            return
        await update.message.reply_text(
            "🤖 NiftySniper is running.\n"
            "Commands:\n/status — System status\n/pnl — Daily P&L\n/STOP — Emergency stop"
        )

    async def _cmd_emergency_stop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            logger.warning("unauthorized_telegram_command", chat_id=update.effective_chat.id, cmd="STOP")
            return
        await update.message.reply_text("🛑 <b>EMERGENCY STOP TRIGGERED</b>", parse_mode="HTML")
        logger.critical("emergency_stop_via_telegram", user=update.effective_user.username)
        if _emergency_stop_callback:
            _emergency_stop_callback()

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            logger.warning("unauthorized_telegram_command", chat_id=update.effective_chat.id, cmd="status")
            return
        from core.execution.risk_manager import risk_manager
        stats = risk_manager.get_daily_stats()
        msg = (
            f"📊 <b>System Status</b>\n"
            f"Mode: {_cfg['system']['mode'].upper()}\n"
            f"Halted: {'YES ⚠️' if stats['is_halted'] else 'NO ✅'}\n"
            f"Open positions: {stats['open_positions']}\n"
            f"Daily P&L: ₹{stats['total_daily_pnl']:+,.2f}"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_pnl(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            logger.warning("unauthorized_telegram_command", chat_id=update.effective_chat.id, cmd="pnl")
            return
        from core.execution.risk_manager import risk_manager
        stats = risk_manager.get_daily_stats()
        await update.message.reply_text(
            _format_pnl_message({
                "daily_realized": stats["daily_realized_pnl"],
                "total_unrealized": stats["daily_unrealized_pnl"],
                "open_positions": [],
            }),
            parse_mode="HTML",
        )

    async def _on_approval_callback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        if not self._is_authorized(update):
            logger.warning("unauthorized_telegram_approval", chat_id=update.effective_chat.id)
            return
        data = query.data  # "approve:MSG_ID" or "skip:MSG_ID"
        action, msg_id = data.split(":", 1)
        if msg_id in _approval_callbacks:
            _approval_callbacks[msg_id]["approved"] = (action == "approve")
            _approval_callbacks[msg_id]["event"].set()
        await query.edit_message_text(
            query.message.text + f"\n\n{'✅ APPROVED' if action == 'approve' else '❌ SKIPPED'}",
            parse_mode="HTML",
        )

    def _send_sync(self, coroutine):
        """Thread-safe bridge: schedule a coroutine from non-async context."""
        if self._app is None or self._loop is None or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def send_signal_alert(self, signal: TradeSignal, instrument: dict):
        """Send signal notification (no approval needed — just FYI)."""
        msg = _format_signal_message(signal, instrument)
        self._send_sync(
            self._app.bot.send_message(
                chat_id=self._chat_id, text=msg, parse_mode="HTML"
            )
        )

    def send_approval_request(
        self, signal: TradeSignal, instrument: dict, timeout_seconds: int = 60
    ) -> bool:
        """
        Send signal to Telegram and wait for human approval.
        Returns True if approved, False if rejected/timed out.
        """
        import threading

        approval_cfg = _cfg["approval"]
        if not approval_cfg["telegram_approval_enabled"]:
            return True

        # A globally unique ID — two signals generated in the same second
        # must never collide and clobber each other's pending approval Event.
        msg_id = f"signal_{uuid.uuid4().hex}"
        event = threading.Event()
        _approval_callbacks[msg_id] = {"event": event, "approved": False}

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ APPROVE", callback_data=f"approve:{msg_id}"),
                InlineKeyboardButton("❌ SKIP", callback_data=f"skip:{msg_id}"),
            ]
        ])

        msg_text = _format_signal_message(signal, instrument)

        async def _send_and_wait():
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text=msg_text + f"\n\n⏱ <i>Auto-skips in {timeout_seconds}s</i>",
                parse_mode="HTML",
                reply_markup=keyboard,
            )

        self._send_sync(_send_and_wait())
        approved = event.wait(timeout=timeout_seconds)

        result = _approval_callbacks.pop(msg_id, {})
        if not approved:
            logger.info("approval_timed_out", msg_id=msg_id)
            auto_execute = approval_cfg["auto_execute_on_timeout"]
            return auto_execute
        return result.get("approved", False)

    def send_trade_executed(self, trade_id: str, instrument: str, price: float, qty: int, mode: str):
        msg = (
            f"✅ <b>TRADE EXECUTED ({mode.upper()})</b>\n"
            f"ID: <code>{trade_id}</code>\n"
            f"Instrument: <code>{instrument}</code>\n"
            f"Price: ₹{price:.2f} | Qty: {qty}"
        )
        self._send_sync(
            self._app.bot.send_message(chat_id=self._chat_id, text=msg, parse_mode="HTML")
        )

    def send_trade_closed(self, trade_id: str, instrument: str, reason: str, pnl: float):
        emoji = "🟢" if pnl >= 0 else "🔴"
        msg = (
            f"{emoji} <b>TRADE CLOSED</b>\n"
            f"ID: <code>{trade_id}</code>\n"
            f"Instrument: <code>{instrument}</code>\n"
            f"Reason: {reason}\n"
            f"P&L: ₹{pnl:+,.2f}"
        )
        self._send_sync(
            self._app.bot.send_message(chat_id=self._chat_id, text=msg, parse_mode="HTML")
        )

    def send_daily_summary(self, pnl_data: dict):
        self._send_sync(
            self._app.bot.send_message(
                chat_id=self._chat_id,
                text=_format_pnl_message(pnl_data),
                parse_mode="HTML",
            )
        )

    def send_alert(self, message: str, severity: str = "INFO"):
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(severity, "ℹ️")
        self._send_sync(
            self._app.bot.send_message(
                chat_id=self._admin_chat_id or self._chat_id,
                text=f"{emoji} <b>[{severity}]</b> {message}",
                parse_mode="HTML",
            )
        )


telegram = TelegramNotifier()

"""
Risk manager: enforces daily loss limits, position caps, and VIX-adjusted sizing.
Acts as the last safety gate before any order reaches the broker.
"""

import threading
from datetime import datetime, date
from typing import Optional
import structlog
import yaml

from database.trade_journal import journal
from database.models import TradingMode

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))


class RiskManager:
    """
    Thread-safe risk manager.
    Maintains intraday state: daily P&L, open position count.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._daily_realized_pnl: float = 0.0
        self._daily_unrealized_pnl: float = 0.0
        self._open_positions: int = 0
        self._halted: bool = False
        self._halt_reason: str = ""
        self._last_reset_date: Optional[date] = None

    def _reset_if_new_day(self):
        today = date.today()
        if self._last_reset_date != today:
            self._daily_realized_pnl = 0.0
            self._daily_unrealized_pnl = 0.0
            self._open_positions = 0
            self._halted = False
            self._halt_reason = ""
            self._last_reset_date = today
            logger.info("risk_manager_daily_reset", date=str(today))

    @property
    def is_halted(self) -> bool:
        return self._halted

    def halt(self, reason: str):
        with self._lock:
            self._halted = True
            self._halt_reason = reason
        journal.log_event("HALT", reason, severity="CRITICAL")
        logger.critical("trading_halted", reason=reason)

    def resume(self):
        with self._lock:
            self._halted = False
            self._halt_reason = ""
        journal.log_event("RESUME", "Trading resumed by human")
        logger.info("trading_resumed")

    def can_open_trade(
        self,
        order_value: float,
        mode: TradingMode = TradingMode.PAPER,
    ) -> tuple[bool, str]:
        """
        Gate: returns (allowed, reason).
        Call this before every trade attempt.
        """
        with self._lock:
            self._reset_if_new_day()

        if self._halted:
            return False, f"System halted: {self._halt_reason}"

        cap = _cfg["capital"]
        total_capital = cap["total"]
        max_daily_loss = total_capital * cap["max_daily_loss_pct"]
        max_positions = cap["max_open_positions"]

        # Daily loss limit
        total_daily_loss = self._daily_realized_pnl + self._daily_unrealized_pnl
        if total_daily_loss <= -max_daily_loss:
            reason = f"Daily loss limit hit: ₹{abs(total_daily_loss):,.0f} ≥ ₹{max_daily_loss:,.0f}"
            self.halt(reason)
            return False, reason

        # Max simultaneous positions
        if self._open_positions >= max_positions:
            return False, f"Max open positions ({max_positions}) reached"

        # Per-trade capital cap
        max_per_trade = total_capital * cap["max_per_trade_pct"]
        if order_value > max_per_trade * 1.05:   # 5% tolerance
            return False, f"Order value ₹{order_value:,.0f} exceeds max ₹{max_per_trade:,.0f}"

        return True, ""

    def on_trade_opened(self, order_value: float):
        with self._lock:
            self._open_positions += 1
            logger.info("position_opened", open_count=self._open_positions, value=order_value)

    def on_trade_closed(self, net_pnl: float):
        with self._lock:
            self._open_positions = max(0, self._open_positions - 1)
            self._daily_realized_pnl += net_pnl
        logger.info(
            "position_closed",
            pnl=net_pnl,
            daily_pnl=self._daily_realized_pnl,
            open_count=self._open_positions,
        )

    def record_realized_pnl(self, net_pnl: float):
        """
        Record realized P&L from a partial exit (e.g. TP1 booking) without
        freeing the position slot — the trade is still open until the
        remaining quantity is closed via on_trade_closed.
        """
        with self._lock:
            self._daily_realized_pnl += net_pnl
        logger.info("partial_pnl_recorded", pnl=net_pnl, daily_pnl=self._daily_realized_pnl)

    def update_unrealized_pnl(self, total_unrealized: float):
        with self._lock:
            self._daily_unrealized_pnl = total_unrealized

    def compute_position_size(
        self,
        signal_score: float,
        vix: Optional[float],
        available_capital: float,
    ) -> float:
        """
        Score-adjusted position sizing.
        Higher score → larger fraction (within caps).
        VIX > 20 → 50% size reduction.
        """
        cap = _cfg["capital"]
        base_pct = cap["max_per_trade_pct"]

        # Score scaling: 65–100 score maps to 50%–100% of max
        min_score = _cfg["signals"]["min_composite_score"]
        score_factor = min(1.0, (signal_score - min_score) / (100 - min_score) * 0.5 + 0.5)

        # VIX adjustment
        vix_factor = 1.0
        if vix and vix > cap["vix_size_reduction_threshold"]:
            vix_factor = cap["vix_size_reduction_factor"]

        order_value = available_capital * base_pct * score_factor * vix_factor
        max_order = available_capital * base_pct
        return min(order_value, max_order)

    def get_daily_stats(self) -> dict:
        with self._lock:
            return {
                "daily_realized_pnl": self._daily_realized_pnl,
                "daily_unrealized_pnl": self._daily_unrealized_pnl,
                "total_daily_pnl": self._daily_realized_pnl + self._daily_unrealized_pnl,
                "open_positions": self._open_positions,
                "is_halted": self._halted,
                "halt_reason": self._halt_reason,
            }


risk_manager = RiskManager()

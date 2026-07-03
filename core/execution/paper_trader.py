"""
Paper trading engine.
Simulates order execution with realistic fills, brokerage, and P&L tracking.
Mirrors the same interface as order_manager.py so switching to live is a flag flip.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict
import structlog
import yaml

from core.signals.signal_engine import TradeSignal
from database.trade_journal import journal
from database.models import Trade, TradeStatus, TradingMode
from core.execution.risk_manager import risk_manager
from core.learning.performance_tracker import performance_tracker

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))


BROKERAGE_PER_LOT = 20.0        # Zerodha flat ₹20/order
STT_PCT = 0.0005                 # 0.05% on premium for options buy
EXCHANGE_CHARGE_PCT = 0.00053    # NSE exchange charge
GST_PCT = 0.18                   # 18% GST on brokerage
SLIPPAGE_PCT = 0.001             # 0.1% slippage on fill


class PaperTrader:
    """
    Paper trading simulation.
    Tracks simulated positions in memory + persists to DB.
    """

    def __init__(self):
        self._positions: Dict[str, dict] = {}   # trade_id → position dict

    def _apply_slippage(self, price: float, direction: str) -> float:
        """Simulate adverse slippage on fill."""
        slippage = price * SLIPPAGE_PCT
        return round(price + slippage if direction == "CE" else price - slippage, 2)

    def _compute_brokerage(self, order_value: float, quantity: int) -> float:
        lots = max(1, quantity // 50)
        brokerage = BROKERAGE_PER_LOT * 2  # Entry + Exit orders
        stt = order_value * STT_PCT
        exchange = order_value * EXCHANGE_CHARGE_PCT
        gst = (brokerage + exchange) * GST_PCT
        return round(brokerage + stt + exchange + gst, 2)

    def place_order(
        self,
        signal: TradeSignal,
        instrument: dict,
        global_context: dict,
        news_sentiment: dict,
    ) -> Optional[str]:
        """
        Simulate placing an order.
        Returns trade_id if successful, None if blocked by risk manager.
        """
        order_value = instrument.get("order_value", 0)
        allowed, reason = risk_manager.can_open_trade(order_value, TradingMode.PAPER)
        if not allowed:
            logger.warning("paper_order_blocked", reason=reason)
            return None

        fill_price = self._apply_slippage(
            instrument.get("option_ltp", signal.entry_price), signal.direction
        )
        quantity = instrument.get("quantity", signal.quantity)
        actual_order_value = fill_price * quantity
        brokerage = self._compute_brokerage(actual_order_value, quantity)

        trade = journal.create_trade(
            instrument=instrument.get("tradingsymbol", "PAPER_INSTRUMENT"),
            underlying=instrument.get("underlying", "NIFTY"),
            exchange=instrument.get("exchange", "NFO"),
            strike=instrument.get("strike"),
            option_type=signal.direction,
            expiry=instrument.get("expiry"),
            mode=TradingMode.PAPER,
            direction="LONG",
            composite_score=signal.composite_score,
            confluence_count=signal.confluence_count,
            indicators_triggered=signal.indicators_triggered,
            regime=signal.regime,
            strategy=signal.strategy,
            signal_time=signal.timestamp,
            entry_time=datetime.utcnow(),
            entry_price=fill_price,
            quantity=quantity,
            remaining_quantity=quantity,
            order_value=actual_order_value,
            stop_loss=instrument["stop_loss"],
            tp1=instrument["tp1"],
            tp2=instrument["tp2"],
            tp3=instrument["tp3"],
            atr_at_entry=signal.atr,
            status=TradeStatus.OPEN,
            brokerage=brokerage,
            global_context=global_context,
            news_sentiment=news_sentiment.get("score"),
            vix_at_signal=global_context.get("india_vix"),
            pcr_at_signal=None,
            approved_by="PAPER",
        )

        self._positions[trade.trade_id] = {
            "trade_id": trade.trade_id,
            "entry_price": fill_price,
            "quantity": quantity,
            "direction": signal.direction,
            "stop_loss": instrument["stop_loss"],
            "tp1": instrument["tp1"],
            "tp2": instrument["tp2"],
            "tp3": instrument["tp3"],
            "tp1_booked": False,
            "tp2_booked": False,
            "remaining_qty": quantity,
            "brokerage": brokerage,
            "total_pnl": 0.0,
            "indicators_triggered": signal.indicators_triggered,
            "order_value": actual_order_value,
        }

        risk_manager.on_trade_opened(actual_order_value)
        logger.info(
            "paper_order_placed",
            trade_id=trade.trade_id,
            instrument=instrument.get("tradingsymbol"),
            fill_price=fill_price,
            quantity=quantity,
        )
        return trade.trade_id

    def on_tick(self, trade_id: str, current_price: float) -> Optional[str]:
        """
        Process a price tick against an open position.
        Handles tiered TP booking and SL exits.
        Returns exit_reason if position closed, else None.
        """
        pos = self._positions.get(trade_id)
        if not pos:
            return None

        direction = pos["direction"]
        entry = pos["entry_price"]
        remaining = pos["remaining_qty"]

        alloc = _cfg["signals"]["tp_allocation"]

        def _pnl_at_price(qty: int, exit_price: float) -> float:
            gross = (exit_price - entry) * qty if direction == "CE" else (entry - exit_price) * qty
            return round(gross, 2)

        exit_reason = None

        # SL check
        sl_hit = (current_price <= pos["stop_loss"]) if direction == "CE" else (current_price >= pos["stop_loss"])
        if sl_hit:
            pnl = _pnl_at_price(remaining, pos["stop_loss"])
            pos["total_pnl"] += pnl
            exit_reason = "SL"
            self._close_position(trade_id, pos["stop_loss"], pos["total_pnl"], "SL")
            return exit_reason

        # TP1
        tp1_hit = (current_price >= pos["tp1"]) if direction == "CE" else (current_price <= pos["tp1"])
        if tp1_hit and not pos["tp1_booked"]:
            qty_to_book = int(pos["quantity"] * alloc["tp1"])
            pnl = _pnl_at_price(qty_to_book, pos["tp1"])
            pos["total_pnl"] += pnl
            pos["remaining_qty"] -= qty_to_book
            pos["tp1_booked"] = True
            journal.update_trade(
                trade_id, status=TradeStatus.PARTIAL,
                remaining_quantity=pos["remaining_qty"], gross_pnl=pos["total_pnl"],
            )
            logger.info("tp1_booked", trade_id=trade_id, pnl=pnl, qty=qty_to_book)

        # TP2
        tp2_hit = (current_price >= pos["tp2"]) if direction == "CE" else (current_price <= pos["tp2"])
        if tp2_hit and pos["tp1_booked"] and not pos["tp2_booked"]:
            qty_to_book = int(pos["quantity"] * alloc["tp2"])
            qty_to_book = min(qty_to_book, pos["remaining_qty"])
            pnl = _pnl_at_price(qty_to_book, pos["tp2"])
            pos["total_pnl"] += pnl
            pos["remaining_qty"] -= qty_to_book
            pos["tp2_booked"] = True
            journal.update_trade(
                trade_id, remaining_quantity=pos["remaining_qty"], gross_pnl=pos["total_pnl"],
            )
            logger.info("tp2_booked", trade_id=trade_id, pnl=pnl, qty=qty_to_book)

        # TP3 — trail remaining 25%
        tp3_hit = (current_price >= pos["tp3"]) if direction == "CE" else (current_price <= pos["tp3"])
        if tp3_hit and pos["tp2_booked"] and pos["remaining_qty"] > 0:
            pnl = _pnl_at_price(pos["remaining_qty"], pos["tp3"])
            pos["total_pnl"] += pnl
            exit_reason = "TP3"
            self._close_position(trade_id, pos["tp3"], pos["total_pnl"], "TP3")
            return exit_reason

        # Update unrealized PnL
        unrealized = _pnl_at_price(pos["remaining_qty"], current_price)
        all_unrealized = sum(
            _pnl_at_price(p["remaining_qty"], current_price)
            for p in self._positions.values()
            if p.get("remaining_qty", 0) > 0
        )
        risk_manager.update_unrealized_pnl(all_unrealized)
        return None

    def force_squareoff(self, trade_id: str, current_price: float):
        """Auto square-off at session end."""
        pos = self._positions.get(trade_id)
        if not pos or pos["remaining_qty"] == 0:
            return
        entry = pos["entry_price"]
        qty = pos["remaining_qty"]
        direction = pos["direction"]
        gross = (current_price - entry) * qty if direction == "CE" else (entry - current_price) * qty
        pos["total_pnl"] += gross
        self._close_position(trade_id, current_price, pos["total_pnl"], "SQUAREOFF")

    def _close_position(self, trade_id: str, exit_price: float, total_pnl: float, reason: str):
        pos = self._positions.pop(trade_id, {})
        if not pos:
            return
        net_pnl = total_pnl - pos.get("brokerage", 0)
        journal.update_trade(
            trade_id,
            exit_time=datetime.utcnow(),
            exit_price=exit_price,
            exit_reason=reason,
            status=TradeStatus.CLOSED,
            gross_pnl=total_pnl,
            net_pnl=net_pnl,
            realized_rr=abs(total_pnl / (pos.get("entry_price", 1) * pos.get("quantity", 1))),
        )
        risk_manager.on_trade_closed(net_pnl)
        try:
            performance_tracker.on_trade_closed(
                trade_id, net_pnl,
                pos.get("indicators_triggered", []),
                pos.get("entry_price", 0.0),
                pos.get("order_value", 0.0),
            )
        except Exception as e:
            logger.error("performance_tracker_update_failed", trade_id=trade_id, error=str(e))
        logger.info("paper_position_closed", trade_id=trade_id, reason=reason, net_pnl=net_pnl)

    def get_open_positions(self) -> dict:
        return dict(self._positions)

    def force_squareoff_all(self, prices: dict[str, float]):
        """Close all open positions — emergency stop or end-of-day."""
        for trade_id in list(self._positions.keys()):
            price = prices.get(trade_id, self._positions[trade_id]["entry_price"])
            self.force_squareoff(trade_id, price)


paper_trader = PaperTrader()

"""
Live order manager — wraps Kite Connect order placement.
Enforces anomaly detection, PIN auth, and audit logging on every order.
"""

from datetime import datetime
from typing import Optional
import structlog
import yaml

from core.market_data.kite_client import kite_client
from core.security.pin_auth import verify_pin
from core.security.anomaly_detector import anomaly_detector
from core.signals.signal_engine import TradeSignal
from database.trade_journal import journal
from database.models import TradeStatus, TradingMode
from core.execution.risk_manager import risk_manager

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))


class OrderManager:
    """
    Live order manager. Only used when mode='live'.
    Every order goes through: anomaly_check → risk_check → PIN_auth → kite_place.
    """

    def place_order(
        self,
        signal: TradeSignal,
        instrument: dict,
        global_context: dict,
        news_sentiment: dict,
        pin: Optional[str] = None,
        approved_by: str = "HUMAN",
    ) -> Optional[str]:
        order_value = instrument.get("order_value", 0)

        # ── Anomaly check ──────────────────────────────────────────────────
        is_anomaly, anomaly_reason = anomaly_detector.check_order(order_value)
        if is_anomaly:
            journal.log_event("ANOMALY", anomaly_reason, severity="WARNING")
            logger.error("order_blocked_anomaly", reason=anomaly_reason)
            return None

        # ── Risk check ──────────────────────────────────────────────────────
        allowed, risk_reason = risk_manager.can_open_trade(order_value, TradingMode.LIVE)
        if not allowed:
            logger.warning("order_blocked_risk", reason=risk_reason)
            return None

        # ── PIN verification ────────────────────────────────────────────────
        if _cfg["security"]["pin_required_for_live"]:
            if not pin or not verify_pin(pin):
                logger.error("order_blocked_invalid_pin")
                journal.log_event("SECURITY", "Live order blocked — invalid PIN", severity="WARNING")
                return None

        # ── Place order on Kite ────────────────────────────────────────────
        try:
            kite = kite_client.kite
            transaction = kite.TRANSACTION_TYPE_BUY   # Always buying options
            kite_order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=instrument["exchange"],
                tradingsymbol=instrument["tradingsymbol"],
                transaction_type=transaction,
                quantity=instrument["quantity"],
                order_type=kite.ORDER_TYPE_MARKET,
                product=kite.PRODUCT_MIS,
                validity=kite.VALIDITY_DAY,
                tag="NiftySniper",
            )
        except Exception as e:
            logger.error("kite_order_failed", error=str(e))
            journal.log_event("ERROR", f"Kite order failed: {e}", severity="CRITICAL")
            return None

        # ── Record in journal ──────────────────────────────────────────────
        trade = journal.create_trade(
            instrument=instrument["tradingsymbol"],
            underlying=instrument["underlying"],
            exchange=instrument["exchange"],
            strike=instrument["strike"],
            option_type=signal.direction,
            expiry=instrument["expiry"],
            mode=TradingMode.LIVE,
            direction="LONG",
            composite_score=signal.composite_score,
            confluence_count=signal.confluence_count,
            indicators_triggered=signal.indicators_triggered,
            regime=signal.regime,
            strategy=signal.strategy,
            signal_time=signal.timestamp,
            entry_time=datetime.utcnow(),
            entry_price=instrument.get("option_ltp", signal.entry_price),
            quantity=instrument["quantity"],
            remaining_quantity=instrument["quantity"],
            order_value=order_value,
            kite_order_id=str(kite_order_id),
            stop_loss=instrument["stop_loss"],
            tp1=instrument["tp1"],
            tp2=instrument["tp2"],
            tp3=instrument["tp3"],
            atr_at_entry=signal.atr,
            status=TradeStatus.OPEN,
            global_context=global_context,
            news_sentiment=news_sentiment.get("score"),
            vix_at_signal=global_context.get("india_vix"),
            approved_by=approved_by,
        )

        risk_manager.on_trade_opened(order_value)
        anomaly_detector.update_average(order_value)

        logger.info(
            "live_order_placed",
            trade_id=trade.trade_id,
            kite_order_id=kite_order_id,
            symbol=instrument["tradingsymbol"],
        )
        return trade.trade_id

    def close_position(
        self,
        trade_id: str,
        tradingsymbol: str,
        exchange: str,
        quantity: int,
        exit_reason: str,
        exit_price: float = 0.0,
    ) -> bool:
        """
        Close `quantity` of an open live position. If `quantity` is less than
        the trade's remaining quantity (e.g. a TP1 partial), the trade stays
        OPEN/PARTIAL with the leftover still tracked for further SL/TP checks
        and the mandatory EOD square-off. Only the final leg marks it CLOSED.
        """
        trade = journal.get_trade(trade_id)
        if not trade:
            logger.error("close_position_trade_not_found", trade_id=trade_id)
            return False

        try:
            kite = kite_client.kite
            kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=kite.TRANSACTION_TYPE_SELL,
                quantity=quantity,
                order_type=kite.ORDER_TYPE_MARKET,
                product=kite.PRODUCT_MIS,
                validity=kite.VALIDITY_DAY,
                tag="NiftySniper_EXIT",
            )
        except Exception as e:
            logger.error("kite_exit_failed", trade_id=trade_id, error=str(e))
            return False

        if trade.option_type == "CE":
            leg_pnl = (exit_price - trade.entry_price) * quantity
        else:
            leg_pnl = (trade.entry_price - exit_price) * quantity

        prior_remaining = trade.remaining_quantity if trade.remaining_quantity is not None else trade.quantity
        new_remaining = max(0, prior_remaining - quantity)
        cum_gross_pnl = (trade.gross_pnl or 0.0) + leg_pnl
        cum_net_pnl = (trade.net_pnl or 0.0) + leg_pnl   # No brokerage model for live fills yet

        if new_remaining <= 0:
            journal.update_trade(
                trade_id,
                exit_time=datetime.utcnow(),
                exit_price=exit_price,
                exit_reason=exit_reason,
                status=TradeStatus.CLOSED,
                remaining_quantity=0,
                gross_pnl=cum_gross_pnl,
                net_pnl=cum_net_pnl,
            )
            risk_manager.on_trade_closed(cum_net_pnl)
            logger.info("live_position_closed", trade_id=trade_id, reason=exit_reason, net_pnl=cum_net_pnl)
        else:
            journal.update_trade(
                trade_id,
                status=TradeStatus.PARTIAL,
                remaining_quantity=new_remaining,
                gross_pnl=cum_gross_pnl,
                net_pnl=cum_net_pnl,
            )
            risk_manager.record_realized_pnl(leg_pnl)
            logger.info(
                "live_position_partial_close", trade_id=trade_id, reason=exit_reason,
                leg_pnl=leg_pnl, remaining_quantity=new_remaining,
            )
        return True

    def cancel_all_pending(self):
        """Cancel all pending/open orders — emergency stop."""
        try:
            orders = kite_client.kite.orders()
            for order in orders:
                if order["status"] in ("OPEN", "TRIGGER PENDING"):
                    kite_client.kite.cancel_order(
                        variety=kite_client.kite.VARIETY_REGULAR,
                        order_id=order["order_id"],
                    )
                    logger.info("order_cancelled", order_id=order["order_id"])
        except Exception as e:
            logger.error("cancel_all_failed", error=str(e))


order_manager = OrderManager()

"""
Position manager: monitors open positions, triggers SL/TP, manages trailing stops.
Works for both paper and live mode by delegating to the appropriate executor.
"""

from typing import Optional
import structlog
import yaml

from core.execution.risk_manager import risk_manager
from core.execution.paper_trader import paper_trader
from core.execution.order_manager import order_manager
from core.market_data.kite_client import kite_client
from database.trade_journal import journal
from database.models import TradingMode, TradeStatus

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))


class PositionManager:
    """
    Called on every 5-minute candle close to check open positions.
    Handles SL, tiered TP, trailing stops, and mandatory square-off.
    """

    def __init__(self):
        self._trailing_stops: dict[str, float] = {}  # trade_id → trailing SL level

    def on_candle(self, current_prices: dict[str, float], mode: str = "paper"):
        """
        Process all open positions against current prices (SL/TP only).
        current_prices: { trade_id: current_option_price }
        The mandatory 3:10 PM square-off is owned solely by the scheduler's
        dedicated squareoff job — duplicating that trigger here as well as
        watching for it before this job's own cron window ends would let
        both jobs try to close the same positions concurrently.
        """
        open_trades = journal.get_open_trades(TradingMode(mode.upper()))

        for trade in open_trades:
            price = current_prices.get(trade.trade_id)
            if price is None:
                continue

            # Update trailing stop for TP3 portion
            self._update_trailing_stop(trade, price)
            effective_sl = self._trailing_stops.get(trade.trade_id, trade.stop_loss)

            if mode == "paper":
                # Inject effective trailing SL into paper trader's position
                if trade.trade_id in paper_trader._positions:
                    paper_trader._positions[trade.trade_id]["stop_loss"] = effective_sl
                exit_reason = paper_trader.on_tick(trade.trade_id, price)
            else:
                exit_reason = self._check_live_exit(trade, price, effective_sl)

            if exit_reason:
                self._trailing_stops.pop(trade.trade_id, None)
                logger.info("position_exit_triggered", trade_id=trade.trade_id, reason=exit_reason, price=price)

    def _update_trailing_stop(self, trade, current_price: float):
        """
        After TP1 is hit, trail stop to breakeven.
        After TP2 is hit, trail stop to TP1 level.
        """
        tid = trade.trade_id
        if trade.option_type == "CE":
            # Move SL up as price rises
            if current_price > trade.tp2 and trade.tp2:
                new_sl = max(self._trailing_stops.get(tid, trade.stop_loss), trade.tp1 or trade.stop_loss)
                self._trailing_stops[tid] = new_sl
            elif current_price > trade.tp1 and trade.tp1:
                new_sl = max(self._trailing_stops.get(tid, trade.stop_loss), trade.entry_price)
                self._trailing_stops[tid] = new_sl
        else:  # PE
            if current_price < trade.tp2 and trade.tp2:
                new_sl = min(self._trailing_stops.get(tid, trade.stop_loss), trade.tp1 or trade.stop_loss)
                self._trailing_stops[tid] = new_sl
            elif current_price < trade.tp1 and trade.tp1:
                new_sl = min(self._trailing_stops.get(tid, trade.stop_loss), trade.entry_price)
                self._trailing_stops[tid] = new_sl

    def _check_live_exit(self, trade, current_price: float, effective_sl: float) -> Optional[str]:
        """Check live position for exit conditions and place exit order if triggered."""
        direction = trade.option_type
        remaining = trade.remaining_quantity if trade.remaining_quantity is not None else trade.quantity
        sl_hit = (current_price <= effective_sl) if direction == "CE" else (current_price >= effective_sl)

        if sl_hit:
            order_manager.close_position(
                trade.trade_id, trade.instrument, trade.exchange,
                remaining, "SL", current_price,
            )
            return "SL"

        # TP checks (simplified — full tiered booking similar to paper_trader).
        # Gated on status == OPEN so TP1 only ever books once per trade —
        # once booked, close_position() flips status to PARTIAL and this
        # won't re-fire on every subsequent candle while price stays above TP1.
        tp1_hit = (current_price >= trade.tp1) if direction == "CE" else (current_price <= trade.tp1)
        if tp1_hit and trade.status == TradeStatus.OPEN:
            alloc = _cfg["signals"]["tp_allocation"]
            qty = min(int(trade.quantity * alloc["tp1"]), remaining)
            order_manager.close_position(
                trade.trade_id, trade.instrument, trade.exchange, qty, "TP1", current_price
            )
            return "TP1_PARTIAL"

        return None

    def get_current_prices_for_mode(self, mode: str) -> dict[str, float]:
        """
        Fetch live LTP for every open trade in `mode`, keyed by trade_id.
        Used for emergency square-off so exits book real P&L instead of
        falling back to entry price (which always books zero P&L).
        """
        open_trades = journal.get_open_trades(TradingMode(mode.upper()))
        if not open_trades:
            return {}

        if not kite_client.is_connected:
            # Paper mode without a broker login — estimate current premiums
            # with Black-Scholes instead of a real LTP feed.
            return self._estimate_prices_black_scholes(open_trades)

        instruments = [f"{t.exchange}:{t.instrument}" for t in open_trades]
        try:
            ltp_map = kite_client.get_ltp(instruments)
        except Exception as e:
            logger.error("ltp_fetch_failed", mode=mode, error=str(e))
            return {}
        return {
            t.trade_id: ltp_map.get(f"{t.exchange}:{t.instrument}", t.entry_price)
            for t in open_trades
        }

    def _estimate_prices_black_scholes(self, open_trades: list) -> dict[str, float]:
        from datetime import datetime
        from core.market_data.historical import fetch_historical_yfinance
        from core.analysis.options_analytics import black_scholes_price

        opt_cfg = _cfg["options"]
        r = opt_cfg.get("risk_free_rate", 0.07)
        iv = opt_cfg.get("synthetic_iv", 0.15)

        spot_cache: dict[str, float] = {}
        prices: dict[str, float] = {}
        for t in open_trades:
            if t.underlying not in spot_cache:
                df = fetch_historical_yfinance(t.underlying, period="1d", interval="1m")
                if df.empty:
                    continue
                spot_cache[t.underlying] = float(df["close"].iloc[-1])
            spot = spot_cache[t.underlying]
            expiry_date = t.expiry.date() if hasattr(t.expiry, "date") else t.expiry
            T = max((expiry_date - datetime.utcnow().date()).days, 0) / 365.0
            prices[t.trade_id] = round(black_scholes_price(spot, t.strike, T, r, iv, t.option_type), 2)
        return prices

    def squareoff_all(self, current_prices: dict[str, float], mode: str = "paper"):
        """Force-close all open positions at market."""
        open_trades = journal.get_open_trades(TradingMode(mode.upper()))
        for trade in open_trades:
            price = current_prices.get(trade.trade_id, trade.entry_price)
            if mode == "paper":
                paper_trader.force_squareoff(trade.trade_id, price)
            else:
                remaining = trade.remaining_quantity if trade.remaining_quantity is not None else trade.quantity
                order_manager.close_position(
                    trade.trade_id, trade.instrument, trade.exchange,
                    remaining, "SQUAREOFF", price
                )
        logger.info("all_positions_squared_off", count=len(open_trades))

    def get_live_pnl_summary(self, current_prices: dict[str, float]) -> dict:
        """Compute real-time P&L summary for dashboard."""
        open_trades = journal.get_open_trades()
        total_unrealized = 0.0
        positions = []

        for trade in open_trades:
            price = current_prices.get(trade.trade_id, trade.entry_price)
            remaining = trade.remaining_quantity if trade.remaining_quantity is not None else trade.quantity
            if trade.option_type == "CE":
                unrealized = (price - trade.entry_price) * remaining
            else:
                unrealized = (trade.entry_price - price) * remaining
            total_unrealized += unrealized
            positions.append({
                "trade_id": trade.trade_id,
                "instrument": trade.instrument,
                "direction": trade.option_type,
                "entry_price": trade.entry_price,
                "current_price": price,
                "unrealized_pnl": round(unrealized, 2),
                "sl": self._trailing_stops.get(trade.trade_id, trade.stop_loss),
                "tp1": trade.tp1,
                "tp2": trade.tp2,
            })

        daily_stats = risk_manager.get_daily_stats()
        return {
            "open_positions": positions,
            "total_unrealized": round(total_unrealized, 2),
            "daily_realized": daily_stats["daily_realized_pnl"],
            "total_daily_pnl": daily_stats["total_daily_pnl"],
            "is_halted": daily_stats["is_halted"],
        }


position_manager = PositionManager()

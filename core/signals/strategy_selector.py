"""
Maps regime + signal to concrete option strike selection and trade parameters.
"""

from datetime import date, timedelta
from typing import Optional
import structlog
import yaml

from core.analysis.options_analytics import select_option_strike, get_options_analytics_summary
from core.analysis.technical import compute_premium_levels
from core.market_data.kite_client import kite_client
from core.signals.signal_engine import TradeSignal
from core.signals.regime_detector import RegimeResult

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))


def get_next_thursday_expiry() -> date:
    """Return the nearest Thursday (weekly Nifty expiry)."""
    today = date.today()
    days_ahead = (3 - today.weekday()) % 7   # Thursday = weekday 3
    if days_ahead == 0:
        days_ahead = 7  # Already Thursday → next week
    return today + timedelta(days=days_ahead)


def get_next_wednesday_expiry() -> date:
    """Return the nearest Wednesday (BankNifty expiry)."""
    today = date.today()
    days_ahead = (2 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def resolve_tradeable_instrument(
    signal: TradeSignal,
    regime: RegimeResult,
    underlying: str = "NIFTY",
    spot_price: float = 0.0,
    expiry: Optional[date] = None,
) -> dict:
    """
    Given a valid signal, select the exact option to trade:
    - Underlying (NIFTY / BANKNIFTY)
    - Expiry (next weekly Thursday / Wednesday)
    - Strike (ATM | ATM+1 | ATM+2 based on config)
    - Direction (CE | PE)

    Returns full trade parameters dict.
    """
    if not signal.is_valid:
        return {}

    expiry = expiry or (
        get_next_thursday_expiry() if underlying == "NIFTY" else get_next_wednesday_expiry()
    )

    # Fetch OI chain for strike selection
    try:
        options_ctx = get_options_analytics_summary(underlying, expiry, spot_price)
        chain_df = options_ctx.get("chain_df")
    except Exception as e:
        logger.warning("chain_fetch_failed", error=str(e))
        chain_df = None

    # Select strike
    preference = _cfg["options"]["strike_selection"]  # ATM | ATM_PLUS_1 | ATM_PLUS_2
    selected = None
    if chain_df is not None and not chain_df.empty:
        selected = select_option_strike(chain_df, spot_price, signal.direction, preference, underlying)

    instrument_cfg = (
        _cfg["instruments"]["primary"] if underlying == _cfg["instruments"]["primary"]["name"]
        else _cfg["instruments"]["secondary"]
    )
    strike_interval = instrument_cfg["strike_interval"]

    if selected is None:
        # Fallback: compute ATM strike manually
        atm_strike = round(spot_price / strike_interval) * strike_interval
        selected = {"strike": atm_strike, "CE_ltp": spot_price * 0.01, "PE_ltp": spot_price * 0.01}

    ltp_key = f"{signal.direction}_ltp"
    option_ltp = selected.get(ltp_key, spot_price * 0.01)
    strike = selected.get("strike")

    # Build instrument symbol (Kite format: NIFTY24JAN23500CE)
    expiry_str = expiry.strftime("%y%b%d").upper()
    tradingsymbol = f"{underlying}{expiry_str}{int(strike)}{signal.direction}"

    # Recompute quantity based on option LTP (not spot price)
    cap_cfg = _cfg["capital"]
    base_order = signal.order_value if signal.order_value > 0 else (
        _cfg["capital"]["total"] * cap_cfg["max_per_trade_pct"]
    )
    lot_size = instrument_cfg["lot_size"]
    lots = max(1, int(base_order / (option_ltp * lot_size)))
    quantity = lots * lot_size

    # SL/TP are computed here (not in signal_engine) because they depend on
    # the actual option premium, which is only known once a strike is picked.
    levels = compute_premium_levels(option_ltp, _cfg)

    return {
        "underlying": underlying,
        "exchange": "NFO",
        "tradingsymbol": tradingsymbol,
        "expiry": expiry,
        "strike": strike,
        "option_type": signal.direction,
        "quantity": quantity,
        "lots": lots,
        "option_ltp": option_ltp,
        "order_value": round(quantity * option_ltp, 2),
        "stop_loss": levels["stop_loss"],
        "tp1": levels["tp1"],
        "tp2": levels["tp2"],
        "tp3": levels["tp3"],
    }

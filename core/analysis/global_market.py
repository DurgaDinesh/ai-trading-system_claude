"""
Global market intelligence: Gift Nifty gap, FII/DII flows, VIX, DXY, Crude, Gold/Silver.
Runs pre-market at 9:00 AM to set the regime context for the day.
"""

import re
import requests
import structlog
import yaml
from datetime import datetime, date
from typing import Optional

from core.market_data.historical import fetch_global_snapshot
from core.security.vault import vault

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))

NSE_FII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
NSE_VIX_URL = "https://www.nseindia.com/api/allIndices"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com",
}


def fetch_india_vix() -> Optional[float]:
    """Fetch India VIX from NSE API."""
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=5)
        resp = session.get(NSE_VIX_URL, headers=NSE_HEADERS, timeout=10)
        data = resp.json()
        for item in data.get("data", []):
            if item.get("index") == "INDIA VIX":
                return float(item["last"])
    except Exception as e:
        logger.warning("vix_fetch_failed", error=str(e))
    return None


def fetch_fii_dii_flows() -> dict:
    """
    Fetch FII and DII cash market net flows from NSE.
    Returns dict with fii_net_cash and dii_net_cash in crores.
    """
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=5)
        resp = session.get(NSE_FII_URL, headers=NSE_HEADERS, timeout=10)
        data = resp.json()
        result = {"fii_net_cash": None, "dii_net_cash": None, "date": None}
        for row in data:
            category = row.get("category", "").upper()
            if "FII" in category or "FOREIGN" in category:
                result["fii_net_cash"] = _parse_crore(row.get("netVal", "0"))
                result["date"] = row.get("date")
            elif "DII" in category or "DOMESTIC" in category:
                result["dii_net_cash"] = _parse_crore(row.get("netVal", "0"))
        return result
    except Exception as e:
        logger.warning("fii_dii_fetch_failed", error=str(e))
        return {"fii_net_cash": None, "dii_net_cash": None, "date": None}


def _parse_crore(val) -> Optional[float]:
    try:
        clean = re.sub(r"[^\d.\-]", "", str(val))
        return float(clean) if clean else None
    except ValueError:
        return None


def fetch_gift_nifty_gap(nifty_prev_close: float) -> dict:
    """
    Gift Nifty (pre-market gap) analysis.

    NOTE: There is no wired-up real-time Gift Nifty / NSE IX data source here.
    The previous implementation fetched ^NSEI (the NSE cash index itself) via
    yfinance as a "proxy" — but at 9:00 AM pre-market, before the cash market
    opens, that index's last daily bar IS still nifty_prev_close, so the gap
    always came out ~0% and this signal silently never fired. Returning a
    fabricated near-zero gap is worse than admitting the data isn't available:
    it looked like a working signal while actually being dead weight.
    Wire in a real Gift Nifty / NSE IX (or a paid vendor) feed here before
    relying on this signal — until then it correctly reports "unavailable".
    """
    logger.warning("gift_nifty_data_source_not_configured")
    return {"gift_nifty": None, "gap_pct": None, "signal": 0}


def get_full_global_context(nifty_prev_close: Optional[float] = None) -> dict:
    """
    Master pre-market intelligence fetch.
    Returns a single context dict consumed by the regime detector and signal engine.
    """
    logger.info("fetching_global_market_context")

    snapshot = fetch_global_snapshot()
    vix = fetch_india_vix()
    fii_dii = fetch_fii_dii_flows()
    gift = fetch_gift_nifty_gap(nifty_prev_close) if nifty_prev_close else {}

    vix_threshold = _cfg["capital"]["vix_size_reduction_threshold"]
    vix_extreme_low = _cfg["global_market"]["vix_extreme_low"]
    vix_extreme_high = _cfg["global_market"]["vix_extreme_high"]

    gm_cfg = _cfg["global_market"]
    dxy_chg = snapshot.get("dxy_chg_pct", 0) or 0
    crude_chg = snapshot.get("crude_chg_pct", 0) or 0

    # Directional scoring
    global_score = 0
    signals = {}

    # DXY — inverse correlation
    if abs(dxy_chg) > gm_cfg["dxy_inverse_threshold"]:
        signals["dxy"] = -1 if dxy_chg > 0 else 1
        global_score += signals["dxy"]

    # FII flows
    fii_net = fii_dii.get("fii_net_cash")
    if fii_net is not None:
        signals["fii"] = 1 if fii_net > 0 else -1
        global_score += signals["fii"]

    # US markets overnight
    sp_chg = snapshot.get("sp500_chg_pct", 0) or 0
    if abs(sp_chg) > 0.3:
        signals["sp500"] = 1 if sp_chg > 0 else -1
        global_score += signals["sp500"]

    # Gift Nifty gap
    if gift.get("signal"):
        signals["gift_nifty"] = gift["signal"]
        global_score += gift["signal"]

    # A failed VIX fetch (None) must never be treated as "VIX == 0" — that
    # silently flagged a missing data point as an "extreme low volatility"
    # regime. When VIX is unavailable, these flags stay neutral/off instead.
    vix_high_vol_regime = vix is not None and vix > vix_threshold
    vix_extreme = vix is not None and (vix > vix_extreme_high or vix < vix_extreme_low)
    position_size_factor = 0.5 if (vix is not None and vix > vix_threshold) else 1.0

    context = {
        "timestamp": datetime.now().isoformat(),
        "india_vix": vix,
        "vix_high_vol_regime": vix_high_vol_regime,
        "vix_extreme": vix_extreme,
        "position_size_factor": position_size_factor,
        "fii_net_cash_cr": fii_dii.get("fii_net_cash"),
        "dii_net_cash_cr": fii_dii.get("dii_net_cash"),
        "dxy": snapshot.get("dxy"),
        "dxy_chg_pct": dxy_chg,
        "crude": snapshot.get("crude"),
        "crude_chg_pct": crude_chg,
        "gold": snapshot.get("gold"),
        "sp500": snapshot.get("sp500"),
        "sp500_chg_pct": sp_chg,
        "dow_chg_pct": snapshot.get("dow_chg_pct"),
        "nasdaq_chg_pct": snapshot.get("nasdaq_chg_pct"),
        "gift_nifty_gap_pct": gift.get("gap_pct"),
        "global_score": global_score,     # net directional signal (-4 to +4)
        "individual_signals": signals,
    }

    logger.info(
        "global_context_ready",
        vix=vix,
        fii=fii_dii.get("fii_net_cash"),
        global_score=global_score,
        gift_gap=gift.get("gap_pct"),
    )
    return context

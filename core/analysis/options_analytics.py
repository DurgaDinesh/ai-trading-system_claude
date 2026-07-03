"""
Options chain analytics: OI, PCR, Max Pain, Greeks (Black-Scholes).
"""

import math
from datetime import date, datetime
from typing import Optional
from scipy.stats import norm
import pandas as pd
import structlog
import yaml

from core.market_data.kite_client import kite_client

logger = structlog.get_logger(__name__)
_full_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))
_cfg = _full_cfg["options"]


def _strike_interval_for(underlying: str) -> int:
    """Strike spacing differs by instrument (NIFTY=50, BANKNIFTY=100, ...)."""
    for key in ("primary", "secondary"):
        instr = _full_cfg["instruments"][key]
        if instr["name"] == underlying:
            return instr["strike_interval"]
    return 50


# ── Black-Scholes Greeks ───────────────────────────────────────────────────────

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "CE") -> float:
    """Standard Black-Scholes option price. T = time to expiry in years."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "CE" else max(K - S, 0)
        return intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "CE":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def compute_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "CE") -> dict:
    """Compute Delta, Gamma, Theta, Vega, Rho."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    delta = norm.cdf(d1) if option_type == "CE" else norm.cdf(d1) - 1
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    theta_CE = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    theta_PE = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
    theta = theta_CE if option_type == "CE" else theta_PE
    vega = S * pdf_d1 * math.sqrt(T) / 100   # per 1% change in IV
    rho = K * T * math.exp(-r * T) * (norm.cdf(d2) if option_type == "CE" else -norm.cdf(-d2)) / 100

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 2),
        "vega": round(vega, 2),
        "rho": round(rho, 2),
    }


def implied_volatility(
    market_price: float, S: float, K: float, T: float, r: float, option_type: str = "CE",
    tol: float = 1e-5, max_iter: int = 200
) -> Optional[float]:
    """Newton-Raphson IV solver. Returns IV as decimal (0.2 = 20%)."""
    if T <= 0 or market_price <= 0:
        return None
    sigma = 0.3
    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega = compute_greeks(S, K, T, r, sigma, option_type)["vega"] * 100
        if vega < 1e-10:
            return None
        diff = market_price - price
        if abs(diff) < tol:
            return round(sigma, 4)
        sigma += diff / vega
        if sigma <= 0:
            sigma = 0.001
    return None


# ── OI & PCR Analysis ─────────────────────────────────────────────────────────

def _build_synthetic_chain(spot_price: float, expiry: date, underlying: str) -> pd.DataFrame:
    """
    Black-Scholes estimated chain for when no live Kite session is available
    (paper mode without a broker login). OI is unknowable without a real
    chain, so it's left at 0 — OI/PCR/max-pain signals correctly stay silent
    (0/neutral) rather than fabricating open-interest data that doesn't exist
    anywhere else for free. Only strike/premium estimation is provided here.
    """
    interval = _strike_interval_for(underlying)
    atm = round(spot_price / interval) * interval
    strikes = [atm + i * interval for i in range(-6, 7)]
    T = max((expiry - date.today()).days, 1) / 365.0
    r = _cfg.get("risk_free_rate", 0.07)
    iv = _cfg.get("synthetic_iv", 0.15)

    rows = []
    for k in strikes:
        rows.append({
            "strike": k,
            "CE_oi": 0,
            "CE_ltp": round(black_scholes_price(spot_price, k, T, r, iv, "CE"), 2),
            "CE_token": None,
            "PE_oi": 0,
            "PE_ltp": round(black_scholes_price(spot_price, k, T, r, iv, "PE"), 2),
            "PE_token": None,
        })
    return pd.DataFrame(rows)


def fetch_option_chain_df(underlying: str, expiry: date, spot_price: Optional[float] = None) -> pd.DataFrame:
    """
    Fetch the full option chain from Kite and return as DataFrame.
    Columns: strike, CE_oi, CE_ltp, CE_iv, PE_oi, PE_ltp, PE_iv
    Falls back to a Black-Scholes estimated synthetic chain if no live Kite
    session is available (paper mode without a broker login) and a spot
    price was supplied.
    """
    if not kite_client.is_connected:
        if spot_price is None:
            return pd.DataFrame()
        logger.info("using_synthetic_chain_no_kite", underlying=underlying)
        return _build_synthetic_chain(spot_price, expiry, underlying)

    chain = kite_client.get_option_chain(underlying, expiry)
    if not chain:
        return pd.DataFrame()

    ce_data = {i["strike"]: i for i in chain if i["instrument_type"] == "CE"}
    pe_data = {i["strike"]: i for i in chain if i["instrument_type"] == "PE"}
    strikes = sorted(set(list(ce_data.keys()) + list(pe_data.keys())))

    rows = []
    for strike in strikes:
        ce = ce_data.get(strike, {})
        pe = pe_data.get(strike, {})
        rows.append({
            "strike": strike,
            "CE_oi": ce.get("oi", 0),
            "CE_ltp": ce.get("last_price", 0),
            "CE_token": ce.get("instrument_token"),
            "PE_oi": pe.get("oi", 0),
            "PE_ltp": pe.get("last_price", 0),
            "PE_token": pe.get("instrument_token"),
        })
    return pd.DataFrame(rows)


def compute_pcr(chain_df: pd.DataFrame) -> float:
    """Put-Call Ratio by OI. PCR > 1 = more put writing = bullish."""
    total_pe_oi = chain_df["PE_oi"].sum()
    total_ce_oi = chain_df["CE_oi"].sum()
    if total_ce_oi == 0:
        return 0.0
    return round(total_pe_oi / total_ce_oi, 3)


def compute_max_pain(chain_df: pd.DataFrame) -> float:
    """
    Max Pain: the strike at which option writers (sellers) experience minimum loss.
    = strike that maximizes total OI decay value for buyers.
    """
    strikes = chain_df["strike"].tolist()
    min_pain = float("inf")
    max_pain_strike = strikes[0]

    for exp_strike in strikes:
        total_pain = 0
        for _, row in chain_df.iterrows():
            total_pain += row["CE_oi"] * max(exp_strike - row["strike"], 0)
            total_pain += row["PE_oi"] * max(row["strike"] - exp_strike, 0)
        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = exp_strike

    return float(max_pain_strike)


def get_oi_signal(chain_df: pd.DataFrame, spot_price: float, underlying: str = "NIFTY") -> int:
    """
    Analyse OI buildup near ATM strikes.
    Bullish signal: CE OI unwinding + PE OI building.
    Bearish signal: PE OI unwinding + CE OI building.
    Returns +1, -1, or 0.
    """
    if chain_df.empty:
        return 0
    interval = _strike_interval_for(underlying)
    atm_strike = round(spot_price / interval) * interval
    band = chain_df[
        (chain_df["strike"] >= atm_strike - 4 * interval) &
        (chain_df["strike"] <= atm_strike + 4 * interval)
    ]
    if band.empty:
        return 0
    ce_oi = band["CE_oi"].sum()
    pe_oi = band["PE_oi"].sum()
    if ce_oi == 0:
        return 0
    pcr_local = pe_oi / ce_oi
    if pcr_local > _cfg["pcr_bullish_threshold"]:
        return 1
    if pcr_local < _cfg["pcr_bearish_threshold"]:
        return -1
    return 0


def select_option_strike(
    chain_df: pd.DataFrame,
    spot_price: float,
    direction: str,          # CE | PE
    preference: str = "ATM_PLUS_1",
    underlying: str = "NIFTY",
) -> Optional[dict]:
    """
    Select the best strike to trade.
    ATM_PLUS_1 = one strike OTM from ATM.
    Returns the row dict from chain_df.
    """
    interval = _strike_interval_for(underlying)
    atm_strike = round(spot_price / interval) * interval
    strikes = sorted(chain_df["strike"].unique())
    if not strikes:
        return None
    # If the exact computed ATM strike isn't listed in the chain, use the closest one available
    # instead of blindly falling back to the first (often deep ITM/OTM) strike in the list.
    idx = strikes.index(atm_strike) if atm_strike in strikes else min(
        range(len(strikes)), key=lambda i: abs(strikes[i] - atm_strike)
    )

    if direction == "CE":
        if preference == "ATM":
            target = strikes[idx]
        elif preference == "ATM_PLUS_1":
            target = strikes[min(idx + 1, len(strikes) - 1)]
        else:
            target = strikes[min(idx + 2, len(strikes) - 1)]
    else:  # PE
        if preference == "ATM":
            target = strikes[idx]
        elif preference == "ATM_PLUS_1":
            target = strikes[max(idx - 1, 0)]
        else:
            target = strikes[max(idx - 2, 0)]

    row = chain_df[chain_df["strike"] == target]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def get_options_analytics_summary(
    underlying: str, expiry: date, spot_price: float
) -> dict:
    """Single call that returns all OI analytics for the signal engine."""
    try:
        chain_df = fetch_option_chain_df(underlying, expiry, spot_price)
        if chain_df.empty:
            return {"pcr": None, "max_pain": None, "oi_signal": 0}
        # A synthetic (no-Kite) chain has no real open interest — reporting a
        # computed PCR/max-pain off all-zero OI would be a meaningless number
        # dressed up as a real signal. Report them honestly as unavailable.
        has_real_oi = chain_df["CE_oi"].sum() > 0 or chain_df["PE_oi"].sum() > 0
        return {
            "pcr": compute_pcr(chain_df) if has_real_oi else None,
            "max_pain": compute_max_pain(chain_df) if has_real_oi else None,
            "oi_signal": get_oi_signal(chain_df, spot_price, underlying) if has_real_oi else 0,
            "chain_df": chain_df,
        }
    except Exception as e:
        logger.error("options_analytics_failed", error=str(e))
        return {"pcr": None, "max_pain": None, "oi_signal": 0}

"""
Market regime detection.
Combines technical structure, VIX, global context, and news sentiment
to classify the current market state and choose the best strategy.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd
import structlog
import yaml

from core.analysis.technical import get_ema_stack_signal, get_vwap_signal, get_supertrend_signal

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))


class Regime(str, Enum):
    BULLISH_MOMENTUM = "BULLISH_MOMENTUM"
    BEARISH_MOMENTUM = "BEARISH_MOMENTUM"
    BULLISH_BREAKOUT = "BULLISH_BREAKOUT"
    BEARISH_BREAKDOWN = "BEARISH_BREAKDOWN"
    SIDEWAYS_LOW_VOL = "SIDEWAYS_LOW_VOL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class RegimeResult:
    regime: Regime
    direction: int          # +1 bullish, -1 bearish, 0 neutral
    confidence: float       # 0.0–1.0
    best_strategy: str
    rationale: list[str]
    vix: Optional[float]
    position_size_factor: float  # 0.5 if VIX > 20, else 1.0


_STRATEGY_MAP = {
    Regime.BULLISH_MOMENTUM:  "BUY_CE_ATM_PLUS_1",
    Regime.BEARISH_MOMENTUM:  "BUY_PE_ATM_PLUS_1",
    Regime.BULLISH_BREAKOUT:  "BUY_CE_ATM",
    Regime.BEARISH_BREAKDOWN: "BUY_PE_ATM",
    Regime.SIDEWAYS_LOW_VOL:  "IRON_CONDOR_OR_SKIP",
    Regime.HIGH_VOLATILITY:   "SKIP_OR_REDUCE_SIZE",
    Regime.UNCERTAIN:         "SKIP",
}


def detect_regime(
    df_5m: pd.DataFrame,
    global_context: dict,
    news_sentiment: dict,
    options_context: dict,
) -> RegimeResult:
    """
    Multi-factor regime detection.

    Parameters
    ----------
    df_5m         : 5-minute OHLCV with all indicators computed
    global_context: from global_market.get_full_global_context()
    news_sentiment: from news_sentiment.get_market_sentiment()
    options_context: from options_analytics (pcr, oi_signal, max_pain)
    """
    rationale = []
    direction_votes = []

    # ── VIX Check (override) ───────────────────────────────────────────────
    vix = global_context.get("india_vix")
    vix_threshold = _cfg["capital"]["vix_size_reduction_threshold"]
    vix_extreme_high = _cfg["global_market"]["vix_extreme_high"]
    position_size_factor = global_context.get("position_size_factor", 1.0)

    if vix and vix > vix_extreme_high:
        rationale.append(f"VIX EXTREME {vix:.1f} > {vix_extreme_high} — HIGH VOLATILITY regime")
        return RegimeResult(
            regime=Regime.HIGH_VOLATILITY,
            direction=0,
            confidence=0.9,
            best_strategy=_STRATEGY_MAP[Regime.HIGH_VOLATILITY],
            rationale=rationale,
            vix=vix,
            position_size_factor=0.5,
        )

    if vix and vix > vix_threshold:
        rationale.append(f"VIX {vix:.1f} > {vix_threshold} — position size halved")

    # ── Technical Signals ──────────────────────────────────────────────────
    ema_sig = get_ema_stack_signal(df_5m)
    vwap_sig = get_vwap_signal(df_5m)
    st_sig = get_supertrend_signal(df_5m)

    if ema_sig == 1:
        direction_votes.append(1)
        rationale.append("EMA stack: 9>21>50>200 (BULLISH)")
    elif ema_sig == -1:
        direction_votes.append(-1)
        rationale.append("EMA stack: 9<21<50<200 (BEARISH)")

    if vwap_sig == 1:
        direction_votes.append(1)
        rationale.append("Price ABOVE VWAP (bullish)")
    elif vwap_sig == -1:
        direction_votes.append(-1)
        rationale.append("Price BELOW VWAP (bearish)")

    if st_sig == 1:
        direction_votes.append(1)
        rationale.append("Supertrend: BULLISH direction")
    elif st_sig == -1:
        direction_votes.append(-1)
        rationale.append("Supertrend: BEARISH direction")

    # ── Global Market ──────────────────────────────────────────────────────
    global_score = global_context.get("global_score", 0)
    if global_score >= 2:
        direction_votes.append(1)
        rationale.append(f"Global markets: {global_score} bullish signals")
    elif global_score <= -2:
        direction_votes.append(-1)
        rationale.append(f"Global markets: {abs(global_score)} bearish signals")

    # FII specifically
    fii = global_context.get("fii_net_cash_cr")
    if fii and fii > 1000:
        direction_votes.append(1)
        rationale.append(f"FII net buyers ₹{fii:.0f}Cr (bullish)")
    elif fii and fii < -1000:
        direction_votes.append(-1)
        rationale.append(f"FII net sellers ₹{abs(fii):.0f}Cr (bearish)")

    # ── Options Sentiment ──────────────────────────────────────────────────
    oi_sig = options_context.get("oi_signal", 0)
    pcr = options_context.get("pcr")
    if oi_sig == 1:
        direction_votes.append(1)
        rationale.append(f"OI: CE unwinding + PE buildup (bullish), PCR={pcr}")
    elif oi_sig == -1:
        direction_votes.append(-1)
        rationale.append(f"OI: PE unwinding + CE buildup (bearish), PCR={pcr}")

    # ── News Sentiment ─────────────────────────────────────────────────────
    news_score = news_sentiment.get("score", 0)
    min_news_conf = _cfg["news"]["sentiment_min_score"]
    if abs(news_score) >= min_news_conf:
        direction_votes.append(1 if news_score > 0 else -1)
        rationale.append(f"News sentiment: {news_score:+.2f} ({news_sentiment.get('method', 'NLP')})")

    # ── Regime Decision ────────────────────────────────────────────────────
    if not direction_votes:
        return RegimeResult(
            regime=Regime.UNCERTAIN,
            direction=0,
            confidence=0.0,
            best_strategy="SKIP",
            rationale=["No clear directional signals"],
            vix=vix,
            position_size_factor=position_size_factor,
        )

    net_direction = sum(direction_votes)
    total_votes = len(direction_votes)
    confidence = abs(net_direction) / total_votes

    if net_direction > 0:
        direction = 1
        # Distinguish momentum from breakout based on RSI/MACD strength
        if "ema_9" in df_5m.columns:
            curr = df_5m.iloc[-1]
            rsi_val = curr.get("rsi", 50) if "rsi" in df_5m.columns else 50
            regime = Regime.BULLISH_MOMENTUM if rsi_val > 55 else Regime.BULLISH_BREAKOUT
        else:
            regime = Regime.BULLISH_MOMENTUM
    elif net_direction < 0:
        direction = -1
        if "rsi" in df_5m.columns:
            rsi_val = df_5m.iloc[-1].get("rsi", 50)
            regime = Regime.BEARISH_MOMENTUM if rsi_val < 45 else Regime.BEARISH_BREAKDOWN
        else:
            regime = Regime.BEARISH_MOMENTUM
    else:
        regime = Regime.SIDEWAYS_LOW_VOL
        direction = 0

    rationale.append(f"Net votes: {net_direction}/{total_votes} → confidence {confidence:.0%}")

    return RegimeResult(
        regime=regime,
        direction=direction,
        confidence=confidence,
        best_strategy=_STRATEGY_MAP.get(regime, "SKIP"),
        rationale=rationale,
        vix=vix,
        position_size_factor=position_size_factor,
    )

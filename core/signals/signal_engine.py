"""
Composite signal engine.
Aggregates all indicator signals into a single scored trade signal.
Enforces confluence rules, R:R filter, and session time rules.
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional
import pytz
import pandas as pd
import structlog
import yaml

from core.analysis.technical import (
    compute_all, get_ema_stack_signal, get_rsi_signal, get_macd_signal,
    get_vwap_signal, get_supertrend_signal,
)
from core.signals.regime_detector import RegimeResult, Regime
from core.execution.risk_manager import risk_manager

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))
IST = pytz.timezone("Asia/Kolkata")

# SL/TP are only known in real premium terms once a strike is selected
# (see strategy_selector.resolve_tradeable_instrument). The ratio between the
# configured premium percentages must still satisfy the minimum R:R — check
# once at import time so a bad config fails fast instead of silently.
_STATIC_RR = _cfg["signals"]["tp_premium_pct"]["tp1"] / _cfg["signals"]["sl_premium_pct"]
if _STATIC_RR < _cfg["signals"]["min_rr_ratio"]:
    raise ValueError(
        f"Configured tp1/sl premium percentages give R:R={_STATIC_RR:.2f}, "
        f"below min_rr_ratio={_cfg['signals']['min_rr_ratio']}"
    )

# Time rules
OBS_END = time(9, 30)
SQUAREOFF_TIME = time(15, 10)
NO_NEW_TRADE_AFTER = time(14, 30)   # No new trades in last 40 min to avoid forced exit


@dataclass
class TradeSignal:
    direction: str               # CE | PE | NONE
    composite_score: float       # 0–100
    confluence_count: int
    indicators_triggered: list[str]
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    atr: float
    rr_ratio: float
    regime: str
    strategy: str
    timestamp: datetime
    is_valid: bool
    invalidation_reason: str = ""
    quantity: int = 0
    order_value: float = 0.0


def _get_weights() -> dict:
    """Load current adaptive weights from DB or config default."""
    try:
        from database.trade_journal import journal
        with journal._get_db() as db:
            from database.models import IndicatorWeight
            row = db.query(IndicatorWeight).order_by(IndicatorWeight.id.desc()).first()
            if row:
                return {
                    "ema_stack": row.ema_stack,
                    "rsi": row.rsi,
                    "macd": row.macd,
                    "vwap": row.vwap,
                    "oi_analysis": row.oi_analysis,
                    "pcr": row.pcr,
                    "global_market": row.global_market,
                }
    except Exception:
        pass
    return _cfg["adaptive_weights"]["initial"]


def _is_trading_allowed(now: datetime) -> tuple[bool, str]:
    """Enforce session time rules."""
    ist_time = now.astimezone(IST).time()
    weekday = now.astimezone(IST).weekday() + 1  # Mon=1
    allowed_days = _cfg["session"]["allowed_trading_days"]

    if weekday not in allowed_days:
        return False, f"Not a trading day (weekday={weekday})"
    if ist_time < OBS_END:
        return False, f"Observation window — no trades before {OBS_END}"
    if ist_time >= NO_NEW_TRADE_AFTER:
        return False, f"Too close to square-off — no new positions after {NO_NEW_TRADE_AFTER}"
    return True, ""


def compute_composite_score(
    df: pd.DataFrame,
    regime: RegimeResult,
    options_context: dict,
    global_context: dict,
    news_sentiment: dict,
    weights: Optional[dict] = None,
) -> tuple[float, list[str], int]:
    """
    Compute weighted composite score (0–100).
    Returns (score, triggered_indicators, confluence_count).
    """
    if weights is None:
        weights = _get_weights()

    direction = regime.direction
    score = 0.0
    triggered = []
    counted_keys = set()   # Supertrend/News share a weight bucket with EMA/Global — count each bucket once

    def _add(name: str, signal: int, weight_key: str):
        nonlocal score
        if signal == direction and direction != 0:
            triggered.append(name)
            if weight_key not in counted_keys:
                score += weights.get(weight_key, 0.10)
                counted_keys.add(weight_key)

    # EMA stack
    _add("EMA_STACK", get_ema_stack_signal(df), "ema_stack")

    # RSI crossover
    _add("RSI_CROSSOVER", get_rsi_signal(df), "rsi")

    # MACD crossover
    _add("MACD_CROSSOVER", get_macd_signal(df), "macd")

    # VWAP
    _add("VWAP_POSITION", get_vwap_signal(df), "vwap")

    # Supertrend
    st_sig = get_supertrend_signal(df)
    _add("SUPERTREND", st_sig, "ema_stack")  # grouped with trend signals

    # OI Analysis
    _add("OI_ANALYSIS", options_context.get("oi_signal", 0), "oi_analysis")

    # PCR signal
    pcr = options_context.get("pcr")
    if pcr:
        pcr_sig = 1 if pcr >= _cfg["options"]["pcr_bullish_threshold"] else (-1 if pcr <= _cfg["options"]["pcr_bearish_threshold"] else 0)
        _add("PCR", pcr_sig, "pcr")

    # Global market alignment
    global_score = global_context.get("global_score", 0)
    global_sig = 1 if global_score > 0 else (-1 if global_score < 0 else 0)
    _add("GLOBAL_MARKET", global_sig, "global_market")

    # News sentiment
    news_score = news_sentiment.get("score", 0)
    news_min = _cfg["news"]["sentiment_min_score"]
    if abs(news_score) >= news_min:
        news_sig = 1 if news_score > 0 else -1
        _add("NEWS_SENTIMENT", news_sig, "global_market")

    # Normalize to 0–100
    total_possible_weight = sum(weights.values())
    composite = round((score / total_possible_weight) * 100, 1)
    return composite, triggered, len(triggered)


def generate_signal(
    df_5m: pd.DataFrame,
    regime: RegimeResult,
    options_context: dict,
    global_context: dict,
    news_sentiment: dict,
    spot_price: float,
    available_capital: float,
) -> TradeSignal:
    """
    Main signal generation entry point.
    Returns a TradeSignal with is_valid=True if all filters pass.
    """
    now = datetime.now(IST)

    # ── Session time check ─────────────────────────────────────────────────
    allowed, reason = _is_trading_allowed(now)
    if not allowed:
        return _invalid_signal(reason, regime, now)

    # ── Regime check ───────────────────────────────────────────────────────
    if regime.regime in (Regime.UNCERTAIN, Regime.HIGH_VOLATILITY, Regime.SIDEWAYS_LOW_VOL):
        return _invalid_signal(f"Regime={regime.regime.value} — not tradeable", regime, now)

    direction_str = "CE" if regime.direction == 1 else "PE"

    # ── Compute indicators ─────────────────────────────────────────────────
    df = compute_all(df_5m)
    composite, triggered, confluence = compute_composite_score(
        df, regime, options_context, global_context, news_sentiment
    )

    min_conf = _cfg["signals"]["min_confluence"]
    min_score = _cfg["signals"]["min_composite_score"]

    if confluence < min_conf:
        return _invalid_signal(
            f"Confluence {confluence} < {min_conf} required", regime, now
        )
    if composite < min_score:
        return _invalid_signal(
            f"Composite score {composite} < {min_score} threshold", regime, now
        )

    # ── ATR (informational — real SL/TP are computed on the option premium
    #    once a strike is selected in strategy_selector.py) ─────────────────
    atr = df["atr"].iloc[-1] if "atr" in df.columns else spot_price * 0.01
    if pd.isna(atr) or atr == 0:
        atr = spot_price * 0.01

    # ── Position sizing ────────────────────────────────────────────────────
    # Score-adjusted and VIX-adjusted internally — a signal that barely
    # clears min_composite_score gets a smaller allocation than a near-100
    # score, and high VIX cuts size further (see risk_manager.compute_position_size).
    # `quantity` here is informational only (index-price terms) — the real,
    # lot-size-aware quantity is computed in strategy_selector.py from the
    # actual option premium once a strike is picked. order_value is the
    # figure that actually carries forward as the trade's capital budget.
    actual_order_value = risk_manager.compute_position_size(
        composite, global_context.get("india_vix"), available_capital
    )
    quantity = max(1, int(actual_order_value / spot_price))

    # ── Max pain check ─────────────────────────────────────────────────────
    max_pain = options_context.get("max_pain")
    if max_pain:
        deviation_pct = abs(spot_price - max_pain) / max_pain * 100
        if deviation_pct < _cfg["options"]["max_pain_deviation_pct"]:
            return _invalid_signal(
                f"Price within {deviation_pct:.1f}% of max pain ({max_pain:.0f}) — caution",
                regime, now,
            )

    logger.info(
        "signal_generated",
        direction=direction_str,
        score=composite,
        confluence=confluence,
        triggered=triggered,
        rr=round(_STATIC_RR, 2),
        regime=regime.regime.value,
    )

    return TradeSignal(
        direction=direction_str,
        composite_score=composite,
        confluence_count=confluence,
        indicators_triggered=triggered,
        entry_price=spot_price,
        # SL/TP are computed on the actual option premium in
        # strategy_selector.resolve_tradeable_instrument, not here.
        stop_loss=0.0,
        tp1=0.0,
        tp2=0.0,
        tp3=0.0,
        atr=round(atr, 2),
        rr_ratio=round(_STATIC_RR, 2),
        regime=regime.regime.value,
        strategy=regime.best_strategy,
        timestamp=now,
        is_valid=True,
        quantity=quantity,
        order_value=round(actual_order_value, 2),
    )


def _invalid_signal(reason: str, regime: RegimeResult, now: datetime) -> TradeSignal:
    logger.info("signal_invalid", reason=reason)
    return TradeSignal(
        direction="NONE", composite_score=0, confluence_count=0,
        indicators_triggered=[], entry_price=0, stop_loss=0,
        tp1=0, tp2=0, tp3=0, atr=0, rr_ratio=0,
        regime=regime.regime.value if regime else "UNKNOWN",
        strategy="SKIP", timestamp=now, is_valid=False,
        invalidation_reason=reason,
    )

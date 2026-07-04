"""Shared fixtures for core/strategies tests."""
from datetime import datetime
import pandas as pd
import pytest
import pytz

from core.analysis.technical import compute_all
from core.signals.regime_detector import Regime, RegimeResult

IST = pytz.timezone("Asia/Kolkata")

# EMA-200 needs real warmup to produce a meaningful stacked signal — use 250 bars
# by default so trend fixtures aren't flat/NaN on the longest configured EMA.
DEFAULT_ROWS = 250


def make_ohlcv_df(
    rows: int = DEFAULT_ROWS,
    start_price: float = 22000.0,
    trend: float = 0.0,
    start_time: str = "2026-01-05 09:35",
    freq: str = "5min",
) -> pd.DataFrame:
    """Synthetic 5-min OHLCV df with a datetime index and optional linear drift. NOT enriched."""
    idx = pd.date_range(start=start_time, periods=rows, freq=freq, tz=IST)
    closes = [start_price + trend * i for i in range(rows)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 5 for c in closes],
            "low": [c - 5 for c in closes],
            "close": closes,
            "volume": [1000] * rows,
        },
        index=idx,
    )


def enriched_df(
    rows: int = DEFAULT_ROWS,
    trend: float = 0.0,
    start_price: float = 22000.0,
    start_time: str = "2026-01-05 09:35",
) -> pd.DataFrame:
    """Synthetic df already run through compute_all(), as strategies expect."""
    return compute_all(make_ohlcv_df(rows=rows, start_price=start_price, trend=trend, start_time=start_time))


def make_regime_result(direction: int = 0, confidence: float = 0.6) -> RegimeResult:
    if direction == 1:
        regime = Regime.BULLISH_MOMENTUM
    elif direction == -1:
        regime = Regime.BEARISH_MOMENTUM
    else:
        regime = Regime.UNCERTAIN
    return RegimeResult(
        regime=regime,
        direction=direction,
        confidence=confidence,
        best_strategy="BUY_CE_ATM_PLUS_1",
        rationale=["test fixture"],
        vix=15.0,
        position_size_factor=1.0,
    )


def make_options_context(pcr=1.0, max_pain=22000.0, oi_signal=0) -> dict:
    return {"pcr": pcr, "max_pain": max_pain, "oi_signal": oi_signal, "chain_df": pd.DataFrame()}


def make_global_context(**overrides) -> dict:
    ctx = {
        "india_vix": 15.0,
        "vix_high_vol_regime": False,
        "vix_extreme": False,
        "fii_net_cash_cr": 0.0,
        "dii_net_cash_cr": 0.0,
        "global_score": 0,
        "gift_nifty_gap_pct": None,
    }
    ctx.update(overrides)
    return ctx


def make_news_sentiment(score: float = 0.0) -> dict:
    return {"score": score, "summary": "test", "risk_events": [], "method": "keyword_fallback"}


@pytest.fixture
def flat_df():
    return enriched_df(trend=0.0)


@pytest.fixture
def uptrend_df():
    return enriched_df(trend=5.0)


@pytest.fixture
def downtrend_df():
    return enriched_df(trend=-5.0)


@pytest.fixture
def neutral_regime():
    return make_regime_result(direction=0, confidence=0.3)


@pytest.fixture
def bullish_regime():
    return make_regime_result(direction=1, confidence=0.8)


@pytest.fixture
def bearish_regime():
    return make_regime_result(direction=-1, confidence=0.8)


@pytest.fixture
def neutral_options_context():
    return make_options_context()


@pytest.fixture
def neutral_global_context():
    return make_global_context()


@pytest.fixture
def neutral_news_sentiment():
    return make_news_sentiment()

"""select_best_signal: promoted strategies compete; fallback only when none promoted."""
from datetime import datetime

import pandas as pd
import pytz

from core.analysis.technical import compute_all
from core.signals.regime_detector import Regime, RegimeResult
from core.signals.strategy_selector import select_best_signal
from core.strategies.base import Strategy

IST = pytz.timezone("Asia/Kolkata")
# Monday 2026-07-06 11:00 IST — inside the 09:30–14:30 strategy trading window
NOW = IST.localize(datetime(2026, 7, 6, 11, 0))


class _HighScore(Strategy):
    name = "sel_high"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "CE", 80.0, ["TEST"], 1, ""


class _LowScore(Strategy):
    name = "sel_low"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "CE", 60.0, ["TEST"], 1, ""


class _TieFirst(Strategy):
    name = "sel_tie_first"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "CE", 70.0, ["TEST"], 1, ""


class _TieSecond(Strategy):
    name = "sel_tie_second"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "PE", 70.0, ["TEST"], 1, ""


class _Boom(Strategy):
    name = "sel_boom"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        raise RuntimeError("boom")


class _Never(Strategy):
    name = "sel_never"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "NONE", 0.0, [], 0, "never"


def _enriched_df(rows=250):
    idx = pd.date_range("2026-01-05 09:35", periods=rows, freq="5min", tz=IST)
    closes = [22000.0 + 5 * i for i in range(rows)]
    return compute_all(pd.DataFrame({
        "open": closes, "high": [c + 5 for c in closes], "low": [c - 5 for c in closes],
        "close": closes, "volume": [1000] * rows,
    }, index=idx))


def _regime(direction=1):
    return RegimeResult(
        regime=Regime.BULLISH_MOMENTUM if direction == 1 else Regime.UNCERTAIN,
        direction=direction, confidence=0.8,
        best_strategy="BUY_CE_ATM_PLUS_1", rationale=["test"],
        vix=15.0, position_size_factor=1.0,
    )


def _contexts():
    options_context = {"pcr": 1.0, "max_pain": None, "oi_signal": 0, "chain_df": pd.DataFrame()}
    global_context = {
        "india_vix": 15.0, "vix_high_vol_regime": False, "vix_extreme": False,
        "fii_net_cash_cr": 0.0, "dii_net_cash_cr": 0.0, "global_score": 0,
        "gift_nifty_gap_pct": None,
    }
    news_sentiment = {"score": 0.0, "summary": "test", "risk_events": [], "method": "test"}
    return options_context, global_context, news_sentiment


def _run(strategies, direction=1):
    oc, gc, ns = _contexts()
    return select_best_signal(
        _enriched_df(), _regime(direction), oc, gc, ns,
        spot_price=22000.0, available_capital=100000.0,
        strategies=strategies, now=NOW,
    )


def test_picks_highest_score_and_sizes_position():
    sig = _run([_LowScore(), _HighScore()])
    assert sig.is_valid
    assert sig.strategy == "sel_high"
    assert sig.direction == "CE"
    assert sig.order_value > 0
    assert sig.quantity >= 1
    assert sig.entry_price == 22000.0


def test_tie_broken_by_rank_order():
    # list order == tournament rank order; equal scores -> earlier wins
    sig = _run([_TieFirst(), _TieSecond()])
    assert sig.strategy == "sel_tie_first"


def test_broken_strategy_is_isolated():
    sig = _run([_Boom(), _LowScore()])
    assert sig.is_valid
    assert sig.strategy == "sel_low"


def test_no_firing_strategy_returns_invalid_without_fallback():
    sig = _run([_Never()])
    assert sig.is_valid is False
    assert sig.invalidation_reason == "No promoted strategy produced a valid signal"


def test_empty_promoted_list_falls_back_to_pipeline():
    # signal_engine path: invalid here either by session time or by
    # Regime=UNCERTAIN — both mark strategy="SKIP", proving the fallback ran.
    sig = _run([], direction=0)
    assert sig.is_valid is False
    assert sig.strategy == "SKIP"

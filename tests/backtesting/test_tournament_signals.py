"""Per-strategy replay: evaluate() over a rolling window -> +1/-1/0 signal series."""
import pandas as pd
import pytz

from backtesting.tournament import neutral_contexts, precompute_regimes, generate_strategy_signals
from core.analysis.technical import compute_all
from core.strategies.base import Strategy

IST = pytz.timezone("Asia/Kolkata")


class _AlwaysCE(Strategy):
    name = "always_ce"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "CE", 60.0, ["TEST"], 1, ""


class _NeverFires(Strategy):
    name = "never_fires"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "NONE", 0.0, [], 0, "never"


def _enriched(rows=220):
    idx = pd.date_range("2026-01-05 09:15", periods=rows, freq="5min", tz=IST)
    closes = [22000.0 + 2 * i for i in range(rows)]
    df = pd.DataFrame({
        "open": closes, "high": [c + 5 for c in closes], "low": [c - 5 for c in closes],
        "close": closes, "volume": [1000] * rows,
    }, index=idx)
    return compute_all(df)


def test_always_ce_gives_plus_one_after_warmup():
    df = _enriched()
    regimes = precompute_regimes(df, warmup=200)
    signals = generate_strategy_signals(_AlwaysCE(), df, regimes, warmup=200)
    assert len(signals) == len(df)
    assert (signals.iloc[:200] == 0).all()
    assert (signals.iloc[200:] == 1).all()


def test_never_fires_gives_all_zero():
    df = _enriched()
    regimes = precompute_regimes(df, warmup=200)
    signals = generate_strategy_signals(_NeverFires(), df, regimes, warmup=200)
    assert (signals == 0).all()


def test_regimes_are_none_before_warmup_and_set_after():
    df = _enriched()
    regimes = precompute_regimes(df, warmup=200)
    assert regimes[0] is None
    assert regimes[199] is None
    assert regimes[200] is not None
    assert hasattr(regimes[200], "direction")

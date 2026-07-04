import pandas as pd

from core.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _with_rsi(df, rsi_values):
    df = df.copy()
    df.loc[df.index[-len(rsi_values):], "rsi"] = rsi_values
    return df


def test_fires_ce_on_oversold_bounce():
    strat = RSIMeanReversionStrategy()
    df = _with_rsi(enriched_df(trend=0.0), [25.0, 28.0])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"
    assert "RSI" in indicators


def test_fires_pe_on_overbought_reversal():
    strat = RSIMeanReversionStrategy()
    df = _with_rsi(enriched_df(trend=0.0), [75.0, 72.0])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_in_neutral_zone():
    strat = RSIMeanReversionStrategy()
    df = _with_rsi(enriched_df(trend=0.0), [48.0, 50.0])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"

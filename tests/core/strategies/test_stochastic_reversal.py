from core.strategies.stochastic_reversal import StochasticReversalStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _with_stoch(df, k, d):
    df = df.copy()
    df.loc[df.index[-len(k):], "stoch_k"] = k
    df.loc[df.index[-len(d):], "stoch_d"] = d
    return df


def test_fires_ce_on_bullish_stochastic_cross():
    strat = StochasticReversalStrategy()
    df = _with_stoch(enriched_df(), [15.0, 25.0], [18.0, 20.0])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_no_signal_without_cross():
    strat = StochasticReversalStrategy()
    df = _with_stoch(enriched_df(), [50.0, 51.0], [50.0, 50.5])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"

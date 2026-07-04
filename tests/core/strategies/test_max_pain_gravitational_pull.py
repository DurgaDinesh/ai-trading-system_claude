from core.strategies.max_pain_gravitational_pull import MaxPainGravitationalPullStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _df_with_close(close):
    df = enriched_df()
    df.loc[df.index[-1], "close"] = close
    return df


def test_fires_ce_when_price_below_max_pain():
    strat = MaxPainGravitationalPullStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        _df_with_close(21800.0), make_regime_result(), make_options_context(max_pain=22000.0),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_fires_pe_when_price_above_max_pain():
    strat = MaxPainGravitationalPullStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        _df_with_close(22200.0), make_regime_result(), make_options_context(max_pain=22000.0),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_near_max_pain():
    strat = MaxPainGravitationalPullStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        _df_with_close(22005.0), make_regime_result(), make_options_context(max_pain=22000.0),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"

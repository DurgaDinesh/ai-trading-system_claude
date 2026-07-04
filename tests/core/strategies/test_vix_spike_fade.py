from core.strategies.vix_spike_fade import VIXSpikeFadeStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_fading_a_downtrend_during_vix_extreme():
    strat = VIXSpikeFadeStrategy()
    df = enriched_df(trend=-5.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=-1), make_options_context(),
        make_global_context(vix_extreme=True), make_news_sentiment(),
    )
    assert direction == "CE"


def test_no_signal_when_vix_not_extreme():
    strat = VIXSpikeFadeStrategy()
    df = enriched_df(trend=-5.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=-1), make_options_context(),
        make_global_context(vix_extreme=False), make_news_sentiment(),
    )
    assert direction == "NONE"

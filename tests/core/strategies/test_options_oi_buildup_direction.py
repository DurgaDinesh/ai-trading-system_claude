from core.strategies.options_oi_buildup_direction import OptionsOIBuildupDirectionStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_on_bullish_oi_buildup():
    strat = OptionsOIBuildupDirectionStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(oi_signal=1),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_fires_pe_on_bearish_oi_buildup():
    strat = OptionsOIBuildupDirectionStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(oi_signal=-1),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_without_oi_buildup():
    strat = OptionsOIBuildupDirectionStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(oi_signal=0),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"

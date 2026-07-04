from core.strategies.pcr_contrarian import PCRContrarianStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_on_high_pcr():
    strat = PCRContrarianStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(pcr=1.8),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_fires_pe_on_low_pcr():
    strat = PCRContrarianStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(pcr=0.4),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_in_neutral_band():
    strat = PCRContrarianStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(pcr=1.0),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"

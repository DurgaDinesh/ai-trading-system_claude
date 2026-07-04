from core.strategies.fii_dii_flow_following import FIIDIIFlowFollowingStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_on_strong_fii_buying():
    strat = FIIDIIFlowFollowingStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(fii_net_cash_cr=1200.0), make_news_sentiment(),
    )
    assert direction == "CE"


def test_fires_pe_on_strong_fii_selling():
    strat = FIIDIIFlowFollowingStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(fii_net_cash_cr=-1200.0), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_on_insignificant_flow():
    strat = FIIDIIFlowFollowingStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(fii_net_cash_cr=50.0), make_news_sentiment(),
    )
    assert direction == "NONE"

from core.strategies.global_cues_gap_trading import GlobalCuesGapTradingStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_on_strong_positive_global_score():
    strat = GlobalCuesGapTradingStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(global_score=3), make_news_sentiment(),
    )
    assert direction == "CE"


def test_fires_pe_on_strong_negative_global_score():
    strat = GlobalCuesGapTradingStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(global_score=-3), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_on_neutral_global_score():
    strat = GlobalCuesGapTradingStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(global_score=0), make_news_sentiment(),
    )
    assert direction == "NONE"

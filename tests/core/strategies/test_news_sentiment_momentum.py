from core.strategies.news_sentiment_momentum import NewsSentimentMomentumStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_on_strongly_positive_sentiment():
    strat = NewsSentimentMomentumStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(score=0.7),
    )
    assert direction == "CE"


def test_fires_pe_on_strongly_negative_sentiment():
    strat = NewsSentimentMomentumStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(score=-0.7),
    )
    assert direction == "PE"


def test_no_signal_on_weak_sentiment():
    strat = NewsSentimentMomentumStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(score=0.1),
    )
    assert direction == "NONE"

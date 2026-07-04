from core.strategies.multi_timeframe_confluence import MultiTimeframeConfluenceStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_when_both_timeframes_agree_bullish():
    strat = MultiTimeframeConfluenceStrategy()
    df = enriched_df(trend=5.0)  # strong sustained uptrend visible on both 5m and resampled 15m
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"
    assert confluence == 2


def test_no_signal_on_flat_market():
    strat = MultiTimeframeConfluenceStrategy()
    df = enriched_df(trend=0.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=0), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"

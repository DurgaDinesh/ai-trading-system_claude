from core.strategies.ema_trend_following import EMATrendFollowingStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_on_uptrend_breakout():
    strat = EMATrendFollowingStrategy()
    df = enriched_df(trend=5.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"
    assert score > 0
    assert "EMA_STACK" in indicators


def test_fires_pe_on_downtrend_breakout():
    strat = EMATrendFollowingStrategy()
    df = enriched_df(trend=-5.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=-1), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "PE"
    assert "EMA_STACK" in indicators


def test_no_signal_on_flat_market():
    strat = EMATrendFollowingStrategy()
    df = enriched_df(trend=0.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=0), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"
    assert reason != ""

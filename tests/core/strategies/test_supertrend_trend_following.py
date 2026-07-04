from core.strategies.supertrend_trend_following import SupertrendTrendFollowingStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)

# Actual `ta`-library contract (see technical.get_supertrend_signal):
# supertrend_dir == -1 means price ABOVE supertrend (bullish),
# supertrend_dir == +1 means price BELOW supertrend (bearish).


def _with_supertrend_flip(df, dir_now, dir_prev):
    df = df.copy()
    df.loc[df.index[-2], "supertrend_dir"] = dir_prev
    df.loc[df.index[-1], "supertrend_dir"] = dir_now
    return df


def test_fires_ce_on_bullish_flip_with_ema_agreement():
    strat = SupertrendTrendFollowingStrategy()
    df = _with_supertrend_flip(enriched_df(trend=5.0), dir_now=-1, dir_prev=1)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"
    assert "SUPERTREND" in indicators


def test_fires_pe_on_bearish_flip():
    strat = SupertrendTrendFollowingStrategy()
    df = _with_supertrend_flip(enriched_df(trend=-5.0), dir_now=1, dir_prev=-1)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=-1), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_without_flip():
    strat = SupertrendTrendFollowingStrategy()
    df = _with_supertrend_flip(enriched_df(trend=5.0), dir_now=-1, dir_prev=-1)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"

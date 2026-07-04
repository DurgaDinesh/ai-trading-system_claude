from core.strategies.regime_aligned_pullback import RegimeAlignedPullbackStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _with_rsi(df, rsi_values):
    df = df.copy()
    df.loc[df.index[-len(rsi_values):], "rsi"] = rsi_values
    return df


def test_fires_ce_on_bullish_regime_with_rsi_pullback():
    strat = RegimeAlignedPullbackStrategy()
    df = _with_rsi(enriched_df(), [25.0, 28.0])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1, confidence=0.8), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_no_signal_when_regime_low_confidence():
    strat = RegimeAlignedPullbackStrategy()
    df = _with_rsi(enriched_df(), [25.0, 28.0])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1, confidence=0.2), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"

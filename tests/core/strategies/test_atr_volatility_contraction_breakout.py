from core.strategies.atr_volatility_contraction_breakout import ATRVolatilityContractionBreakoutStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _with_atr_squeeze_then_expand(df, ema_direction: int):
    df = df.copy()
    df["atr"] = 10.0  # flat/contracted baseline for the whole lookback window
    df.loc[df.index[-1], "atr"] = 20.0  # sharp expansion on the last bar
    if ema_direction == 1:
        df.loc[df.index[-1], "close"] = df["close"].iloc[-1] + 500  # push EMA stack bullish
    return df


def test_fires_on_atr_expansion_after_contraction_with_ema_bias():
    strat = ATRVolatilityContractionBreakoutStrategy()
    df = _with_atr_squeeze_then_expand(enriched_df(trend=5.0), ema_direction=1)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"
    assert "ATR_EXPANSION" in indicators


def test_no_signal_without_prior_contraction():
    strat = ATRVolatilityContractionBreakoutStrategy()
    df = enriched_df(trend=5.0)
    df["atr"] = 15.0  # flat ATR with no expansion on the last bar
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"

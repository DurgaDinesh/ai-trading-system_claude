from core.strategies.macd_momentum_crossover import MACDMomentumCrossoverStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _with_macd(df, macd, macd_signal, macd_hist):
    df = df.copy()
    df.loc[df.index[-len(macd):], "macd"] = macd
    df.loc[df.index[-len(macd_signal):], "macd_signal"] = macd_signal
    df.loc[df.index[-len(macd_hist):], "macd_hist"] = macd_hist
    return df


def test_fires_ce_on_bullish_crossover_with_expanding_histogram():
    strat = MACDMomentumCrossoverStrategy()
    df = _with_macd(enriched_df(), [1.0, 2.0], [1.5, 1.5], [0.2, 0.5])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"
    assert "MACD" in indicators


def test_no_signal_without_crossover():
    strat = MACDMomentumCrossoverStrategy()
    df = _with_macd(enriched_df(), [1.0, 1.1], [1.5, 1.5], [-0.5, -0.4])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"

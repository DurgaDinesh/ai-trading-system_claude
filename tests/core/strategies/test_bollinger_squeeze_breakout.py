import pandas as pd

from core.strategies.bollinger_squeeze_breakout import BollingerSqueezeBreakoutStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _squeeze_then_breakout(df, direction: str):
    df = df.copy()
    df["bb_width"] = 4.0
    df["bb_upper"] = 22050.0
    df["bb_lower"] = 21950.0
    if direction == "up":
        df.loc[df.index[-1], "close"] = 22100.0
    elif direction == "down":
        df.loc[df.index[-1], "close"] = 21900.0
    return df


def test_fires_ce_on_upside_squeeze_breakout():
    strat = BollingerSqueezeBreakoutStrategy()
    df = _squeeze_then_breakout(enriched_df(), "up")
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_no_signal_without_squeeze_breakout():
    strat = BollingerSqueezeBreakoutStrategy()
    df = _squeeze_then_breakout(enriched_df(), "none")
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"

from core.strategies.vwap_reversion import VWAPReversionStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _with_vwap_gap(df, close, vwap):
    df = df.copy()
    df.loc[df.index[-1], "close"] = close
    df.loc[df.index[-1], "vwap"] = vwap
    return df


def test_fires_ce_when_price_well_below_vwap():
    strat = VWAPReversionStrategy()
    df = _with_vwap_gap(enriched_df(), close=21900.0, vwap=22000.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_fires_pe_when_price_well_above_vwap():
    strat = VWAPReversionStrategy()
    df = _with_vwap_gap(enriched_df(), close=22100.0, vwap=22000.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_near_vwap():
    strat = VWAPReversionStrategy()
    df = _with_vwap_gap(enriched_df(), close=22005.0, vwap=22000.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"

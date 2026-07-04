import pandas as pd
import pytz

from core.strategies.gap_and_go_momentum import GapAndGoMomentumStrategy
from tests.core.strategies.conftest import (
    make_regime_result, make_options_context, make_global_context, make_news_sentiment,
)
from core.analysis.technical import compute_all

IST = pytz.timezone("Asia/Kolkata")


def _two_day_df(gap: str = "none") -> pd.DataFrame:
    # >= 14 prior bars so `ta`'s ATR (window 14) doesn't IndexError in compute_all()
    prior_idx = pd.date_range("2026-01-04 09:15", periods=20, freq="5min", tz=IST)
    prior_closes = [22000.0, 22005.0] * 9 + [22005.0, 22000.0]  # prev close = 22000

    today_open = 22150.0 if gap == "up" else (21850.0 if gap == "down" else 22001.0)
    today_idx = pd.date_range("2026-01-05 09:15", periods=2, freq="5min", tz=IST)
    today_closes = [today_open, today_open + (30.0 if gap == "up" else (-30.0 if gap == "down" else 0.0))]

    idx = prior_idx.append(today_idx)
    closes = prior_closes + today_closes
    df = pd.DataFrame({
        "open": closes, "high": [c + 3 for c in closes], "low": [c - 3 for c in closes],
        "close": closes, "volume": [1000] * len(closes),
    }, index=idx)
    return compute_all(df)


def test_fires_ce_on_gap_up_and_go():
    strat = GapAndGoMomentumStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        _two_day_df("up"), make_regime_result(), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_fires_pe_on_gap_down_and_go():
    strat = GapAndGoMomentumStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        _two_day_df("down"), make_regime_result(), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_without_gap():
    strat = GapAndGoMomentumStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        _two_day_df("none"), make_regime_result(), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"

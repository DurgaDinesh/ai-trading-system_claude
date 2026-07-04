import pandas as pd

from core.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from tests.core.strategies.conftest import (
    make_regime_result, make_options_context, make_global_context, make_news_sentiment,
)
from core.analysis.technical import compute_all
import pytz

IST = pytz.timezone("Asia/Kolkata")


def _session_df(breakout: str = "none") -> pd.DataFrame:
    """9:15-9:30 opening range at 22000 +/- 5, filler bars inside the range,
    then a final bar that breaks out (or not).

    Needs >= 14 bars total: `ta`'s ATR indexes true_range[0:14] directly and
    raises IndexError on shorter frames when compute_all() runs.
    """
    range_closes = [22000.0, 22002.0, 21998.0, 22001.0]  # 9:15-9:30
    filler_closes = [22001.0] * 17                        # 9:35-10:55, inside range
    closes = range_closes + filler_closes
    if breakout == "up":
        closes.append(22050.0)
    elif breakout == "down":
        closes.append(21950.0)
    else:
        closes.append(22001.0)
    idx = pd.date_range("2026-01-05 09:15", periods=len(closes), freq="5min", tz=IST)
    df = pd.DataFrame({
        "open": closes, "high": [c + 3 for c in closes], "low": [c - 3 for c in closes],
        "close": closes, "volume": [1000] * len(closes),
    }, index=idx)
    return compute_all(df)


def test_fires_ce_on_breakout_above_opening_range():
    strat = OpeningRangeBreakoutStrategy()
    df = _session_df("up")
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_fires_pe_on_breakdown_below_opening_range():
    strat = OpeningRangeBreakoutStrategy()
    df = _session_df("down")
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_inside_range():
    strat = OpeningRangeBreakoutStrategy()
    df = _session_df("none")
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"

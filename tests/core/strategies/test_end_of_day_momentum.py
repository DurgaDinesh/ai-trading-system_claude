import pandas as pd
import pytz

from core.strategies.end_of_day_momentum import EndOfDayMomentumStrategy
from tests.core.strategies.conftest import (
    make_regime_result, make_options_context, make_global_context, make_news_sentiment,
)
from core.analysis.technical import compute_all

IST = pytz.timezone("Asia/Kolkata")


def _late_session_uptrend_df() -> pd.DataFrame:
    idx = pd.date_range("2026-01-05 13:00", periods=260, freq="1min", tz=IST)
    closes = [22000.0 + i * 0.5 for i in range(260)]
    df = pd.DataFrame({
        "open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
        "close": closes, "volume": [1000] * 260,
    }, index=idx)
    return compute_all(df)


def _mid_session_uptrend_df() -> pd.DataFrame:
    # 250 bars from 09:15 end at 13:24 — genuinely BEFORE the 13:30 window.
    # (The plan's original 260 bars from 10:00 ended at 14:19, inside the window.)
    idx = pd.date_range("2026-01-05 09:15", periods=250, freq="1min", tz=IST)
    closes = [22000.0 + i * 0.5 for i in range(250)]
    df = pd.DataFrame({
        "open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
        "close": closes, "volume": [1000] * 250,
    }, index=idx)
    return compute_all(df)


def test_fires_ce_in_late_session_uptrend():
    strat = EndOfDayMomentumStrategy()
    df = _late_session_uptrend_df()
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_no_signal_before_end_of_day_window():
    strat = EndOfDayMomentumStrategy()
    df = _mid_session_uptrend_df()
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"
    assert "window" in reason.lower()

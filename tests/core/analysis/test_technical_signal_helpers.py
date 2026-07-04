"""Tests for get_bollinger_signal and get_stochastic_signal in technical.py."""
import pandas as pd

from core.analysis.technical import get_bollinger_signal, get_stochastic_signal


def _bollinger_df(breakout: str = "none") -> pd.DataFrame:
    """21 rows: 20 bars of a tight squeeze, then 1 bar that breaks out (or not)."""
    n = 21
    rows = []
    for i in range(n - 1):
        rows.append({"close": 100.0, "bb_upper": 102.0, "bb_lower": 98.0, "bb_width": 4.0})
    if breakout == "up":
        rows.append({"close": 103.0, "bb_upper": 102.0, "bb_lower": 98.0, "bb_width": 4.0})
    elif breakout == "down":
        rows.append({"close": 97.0, "bb_upper": 102.0, "bb_lower": 98.0, "bb_width": 4.0})
    else:
        rows.append({"close": 100.0, "bb_upper": 102.0, "bb_lower": 98.0, "bb_width": 4.0})
    return pd.DataFrame(rows)


def test_get_bollinger_signal_breakout_up_after_squeeze():
    assert get_bollinger_signal(_bollinger_df("up")) == 1


def test_get_bollinger_signal_breakout_down_after_squeeze():
    assert get_bollinger_signal(_bollinger_df("down")) == -1


def test_get_bollinger_signal_no_breakout():
    assert get_bollinger_signal(_bollinger_df("none")) == 0


def _stochastic_df(cross: str = "none") -> pd.DataFrame:
    if cross == "up":
        return pd.DataFrame({"stoch_k": [15.0, 25.0], "stoch_d": [18.0, 20.0]})
    if cross == "down":
        return pd.DataFrame({"stoch_k": [85.0, 75.0], "stoch_d": [82.0, 80.0]})
    return pd.DataFrame({"stoch_k": [50.0, 51.0], "stoch_d": [50.0, 50.5]})


def test_get_stochastic_signal_bullish_cross_from_oversold():
    assert get_stochastic_signal(_stochastic_df("up")) == 1


def test_get_stochastic_signal_bearish_cross_from_overbought():
    assert get_stochastic_signal(_stochastic_df("down")) == -1


def test_get_stochastic_signal_no_cross():
    assert get_stochastic_signal(_stochastic_df("none")) == 0

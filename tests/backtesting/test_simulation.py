"""simulate_signals: next-bar-open entry, SL/TP1/TP2 exits, metric outputs."""
import pandas as pd
import pytest

from backtesting.simulation import simulate_signals


def _df(closes, opens=None):
    idx = pd.date_range("2026-01-05 09:15", periods=len(closes), freq="1h")
    opens = opens or closes
    return pd.DataFrame({
        "open": opens, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
        "close": closes, "volume": [1000] * len(closes),
        "atr": [10.0] * len(closes),
    }, index=idx)


def _series(df, values):
    return pd.Series(values, index=df.index, dtype=float)


def test_no_signals_returns_zero_trades():
    df = _df([100.0] * 10)
    signals = _series(df, [0] * 10)
    nan = float("nan")
    result = simulate_signals(df, signals, _series(df, [nan] * 10),
                              _series(df, [nan] * 10), _series(df, [nan] * 10),
                              initial_capital=100000.0, period_days=10)
    assert result["total_trades"] == 0
    assert result["final_capital"] == 100000.0
    assert result["trades"] == []


def test_long_tp2_win_books_profit():
    # Signal at bar 1 (close 100) -> entry at bar 2 open (100).
    # TP2=120 is hit when close reaches 121 at bar 4.
    closes = [100.0, 100.0, 100.0, 110.0, 121.0, 121.0]
    df = _df(closes)
    signals = _series(df, [0, 1, 0, 0, 0, 0])
    nan = float("nan")
    sl = _series(df, [nan, 90.0, nan, nan, nan, nan])
    tp1 = _series(df, [nan, 110.0, nan, nan, nan, nan])
    tp2 = _series(df, [nan, 120.0, nan, nan, nan, nan])
    result = simulate_signals(df, signals, sl, tp1, tp2,
                              initial_capital=100000.0, period_days=6)
    assert result["total_trades"] == 1
    assert result["trades"][0]["exit_reason"] == "TP2"
    assert result["total_net_pnl"] > 0
    assert result["final_capital"] > 100000.0


def test_long_sl_loss_books_loss():
    closes = [100.0, 100.0, 100.0, 89.0, 89.0]
    df = _df(closes)
    signals = _series(df, [0, 1, 0, 0, 0])
    nan = float("nan")
    sl = _series(df, [nan, 90.0, nan, nan, nan])
    tp1 = _series(df, [nan, 110.0, nan, nan, nan])
    tp2 = _series(df, [nan, 120.0, nan, nan, nan])
    result = simulate_signals(df, signals, sl, tp1, tp2,
                              initial_capital=100000.0, period_days=5)
    assert result["total_trades"] == 1
    assert result["trades"][0]["exit_reason"] == "SL"
    assert result["total_net_pnl"] < 0

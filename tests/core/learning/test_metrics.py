"""Tests for the shared trade-metric helpers."""
import math

from core.learning.metrics import annualization_from_span, compute_trade_metrics


def test_annualization_from_span_calendar():
    # 100 trades over 365.25 days -> 100 trades/year -> sqrt(100) = 10
    assert math.isclose(annualization_from_span(100, 365.25), 10.0)


def test_annualization_from_span_trading_days():
    # 63 trades over 63 trading days at 252/year -> 252 trades/year -> sqrt(252)
    assert math.isclose(annualization_from_span(63, 63, days_per_year=252), 252 ** 0.5)


def test_annualization_zero_guards():
    assert annualization_from_span(0, 10) == 0.0
    assert annualization_from_span(10, 0) == 0.0


def test_compute_trade_metrics_basic():
    pnls = [100.0, -50.0, 200.0, -100.0]
    m = compute_trade_metrics(pnls)
    assert m["total_trades"] == 4
    assert m["winning_trades"] == 2
    assert m["losing_trades"] == 2
    assert math.isclose(m["win_rate"], 0.5)
    assert math.isclose(m["profit_factor"], 300.0 / 150.0)
    assert math.isclose(m["total_net_pnl"], 150.0)
    assert math.isclose(m["avg_pnl_per_trade"], 37.5)
    assert m["max_win"] == 200.0
    assert m["max_loss"] == -100.0
    assert math.isclose(m["avg_win"], 150.0)
    assert math.isclose(m["avg_loss"], -75.0)
    # cumulative: 100, 50, 250, 150 -> running max 100,100,250,250 -> dd 0,50,0,100
    assert math.isclose(m["max_drawdown"], 100.0)


def test_compute_trade_metrics_all_wins_profit_factor_inf():
    m = compute_trade_metrics([10.0, 20.0])
    assert m["profit_factor"] == float("inf")
    assert m["max_drawdown"] == 0.0


def test_compute_trade_metrics_empty():
    m = compute_trade_metrics([])
    assert m["total_trades"] == 0
    assert m["win_rate"] == 0.0
    assert m["profit_factor"] == 0.0
    assert m["sharpe"] == 0.0


def test_sharpe_uses_annualization_factor():
    pnls = [1.0, 2.0, 3.0, 2.0]
    m_raw = compute_trade_metrics(pnls, annualization=1.0)
    m_ann = compute_trade_metrics(pnls, annualization=2.0)
    assert math.isclose(m_ann["sharpe"], m_raw["sharpe"] * 2.0)


def test_sharpe_zero_when_fewer_than_two_trades_or_zero_std():
    assert compute_trade_metrics([5.0], annualization=1.0)["sharpe"] == 0.0
    assert compute_trade_metrics([5.0, 5.0], annualization=1.0)["sharpe"] == 0.0

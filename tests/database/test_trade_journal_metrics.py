"""_approx_sharpe must keep its trading-day annualization after the metrics refactor."""
import math
from datetime import datetime, timedelta
from types import SimpleNamespace

from database.trade_journal import _approx_sharpe
from core.learning.metrics import annualization_from_span, compute_trade_metrics


def _fake_trades(pnls, span_days):
    t0 = datetime(2026, 1, 5, 10, 0)
    trades = []
    for i, p in enumerate(pnls):
        ts = t0 + timedelta(days=span_days * i / max(len(pnls) - 1, 1))
        trades.append(SimpleNamespace(net_pnl=p, entry_time=ts, created_at=ts))
    return trades


def test_approx_sharpe_matches_shared_metrics_math():
    pnls = [100.0, -50.0, 200.0, -100.0, 80.0]
    span = 10
    trades = _fake_trades(pnls, span)
    expected_ann = annualization_from_span(len(pnls), span, days_per_year=252)
    expected = compute_trade_metrics(pnls, annualization=expected_ann)["sharpe"]
    assert math.isclose(_approx_sharpe(trades), expected, rel_tol=1e-9)


def test_approx_sharpe_zero_for_fewer_than_two():
    assert _approx_sharpe(_fake_trades([50.0], 1)) == 0.0

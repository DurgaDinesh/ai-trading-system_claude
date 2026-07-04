# Tournament Backtest Runner + Shared Metrics (AI Brain Plan 2 of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the weekly strategy tournament: one shared metrics module, a reusable signal-simulation function extracted from `backtest_engine`, a `StrategyRanking` DB table, and `backtesting/tournament.py` that backtests all 20 registered strategies, ranks them by a config-weighted composite score, and marks the top N as promoted.

**Architecture:** Metric math is extracted into pure functions in `core/learning/metrics.py` (consumed by `trade_journal`, `backtest_engine`, and the tournament — one implementation, pluggable annualization). The bar-by-bar trade simulation loop is extracted from `run_backtest` into `backtesting/simulation.py` so the tournament can reuse it per strategy. `backtesting/tournament.py` precomputes per-bar regimes once, replays each strategy's `evaluate()` over the enriched df, simulates its signals, computes metrics, min-max-normalizes across strategies, and writes `StrategyRanking` rows. All I/O (data fetch, DB session) is injectable for tests.

**Tech Stack:** Python 3.14, pandas, `ta` library (NOT pandas_ta), SQLAlchemy, pytest 9.

## Global Constraints

- Do NOT touch `core/signals/strategy_selector.py`, `scheduler/`, or `dashboard/` — that's Plan 4. Do NOT create `mistake_analyzer.py` or the `MissedOpportunity` table — that's Plan 3.
- `run_backtest()`'s public signature and result-dict keys must not change (dashboard/reports consume them).
- `journal.get_performance_stats()` must keep its exact key names (`total_trades, win_rate, profit_factor, total_net_pnl, avg_pnl_per_trade, max_win, max_loss, avg_win, avg_loss, sharpe_approx`) — `performance_tracker.check_live_escalation_readiness()` reads them.
- `opportunity_capture_rate` is stored as a column but always `0.0` in this plan; Plan 3 wires the real value into the composite score.
- Every strategy failure inside the tournament is isolated: `status="errored"`, never a crashed run (spec "Error handling").
- Strategies with fewer than `min_backtest_trades_for_ranking` (config, default 15) simulated trades get `status="insufficient_data"` and are excluded from ranking/promotion.
- Composite weights come from `config/settings.yaml` `strategy_tournament.score_weights`, defaults exactly: profit_factor 0.35, sharpe 0.25, win_rate 0.20, max_drawdown 0.20 (inverted).
- Run tests with `python -m pytest <path> -v` from the repo root (`E:\Algo Trading`).

---

## Task 1: `core/learning/metrics.py` — shared metric math

**Files:**
- Create: `core/learning/metrics.py`
- Test: `tests/core/learning/test_metrics.py`
- Create: `tests/core/learning/__init__.py` (empty)

**Interfaces:**
- Produces (consumed by Tasks 2, 3, 6):
  - `annualization_from_span(n_trades: int, span_days: float, days_per_year: float = 365.25) -> float`
  - `compute_trade_metrics(pnls: list[float], annualization: float = 0.0) -> dict` with keys
    `total_trades, winning_trades, losing_trades, win_rate, profit_factor, total_net_pnl, avg_pnl_per_trade, max_win, max_loss, avg_win, avg_loss, max_drawdown, sharpe, sortino`.
    For an empty list every value is 0 (win_rate 0.0, profit_factor 0.0).

- [ ] **Step 1: Write the failing test**

`tests/core/learning/test_metrics.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/learning/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.learning.metrics'`

- [ ] **Step 3: Write the implementation**

`core/learning/metrics.py`:

```python
"""
Shared trade-metric calculations.

One implementation used by: the live-escalation stats (database/trade_journal.py),
the single-pipeline backtest (backtesting/), and the strategy tournament.
Annualization is passed in by the caller so each call site keeps its own
convention (calendar days for backtests, trading days for the live journal).
"""
import statistics

import numpy as np


def annualization_from_span(n_trades: int, span_days: float, days_per_year: float = 365.25) -> float:
    """sqrt(trades-per-year) scaling factor for a per-trade Sharpe/Sortino."""
    if n_trades <= 0 or span_days <= 0:
        return 0.0
    trades_per_year = n_trades / span_days * days_per_year
    return trades_per_year ** 0.5


def compute_trade_metrics(pnls: list[float], annualization: float = 0.0) -> dict:
    """All per-trade-P&L metrics the system reports, from a list of net P&Ls."""
    if not pnls:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0.0, "profit_factor": 0.0, "total_net_pnl": 0.0,
            "avg_pnl_per_trade": 0.0, "max_win": 0.0, "max_loss": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "max_drawdown": 0.0,
            "sharpe": 0.0, "sortino": 0.0,
        }

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses))
    if gross_losses > 0:
        profit_factor = gross_wins / gross_losses
    else:
        profit_factor = float("inf") if gross_wins > 0 else 0.0

    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    max_drawdown = float(np.max(running_max - cumulative))

    if len(pnls) > 1:
        mean_pnl = statistics.mean(pnls)
        std_pnl = statistics.stdev(pnls)
        sharpe = (mean_pnl / std_pnl) * annualization if std_pnl else 0.0
    else:
        sharpe = 0.0

    neg = [p for p in pnls if p < 0]
    if len(neg) > 1:
        sortino_std = statistics.stdev(neg)
        sortino = (statistics.mean(pnls) / sortino_std) * annualization if sortino_std else 0.0
    else:
        sortino = 0.0

    return {
        "total_trades": len(pnls),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(pnls),
        "profit_factor": profit_factor,
        "total_net_pnl": sum(pnls),
        "avg_pnl_per_trade": sum(pnls) / len(pnls),
        "max_win": max(pnls),
        "max_loss": min(pnls),
        "avg_win": gross_wins / len(wins) if wins else 0.0,
        "avg_loss": (gross_losses / len(losses)) * -1 if losses else 0.0,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/learning/test_metrics.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add core/learning/metrics.py tests/core/learning/__init__.py tests/core/learning/test_metrics.py
git commit -m "Add shared trade-metrics module (core/learning/metrics.py)"
```

---

## Task 2: Refactor `trade_journal` to delegate metric math to `metrics.py`

**Files:**
- Modify: `database/trade_journal.py` (functions `_approx_sharpe` at line ~218 and `TradeJournal.get_performance_stats` at line ~182)
- Test: `tests/database/test_trade_journal_metrics.py`
- Create: `tests/database/__init__.py` (empty)

**Interfaces:**
- Consumes: `compute_trade_metrics`, `annualization_from_span` from Task 1.
- Produces: unchanged public behavior — `get_performance_stats()` returns the exact same keys as before (see Global Constraints); `_approx_sharpe(trades)` keeps its signature (list of ORM `Trade`-like objects with `.net_pnl`, `.entry_time`, `.created_at`).

- [ ] **Step 1: Write the failing test**

`tests/database/test_trade_journal_metrics.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_trade_journal_metrics.py -v`
Expected: PASS is possible only by luck; typically the first test PASSES already if math is identical — that is fine. The point of the test is to lock behavior BEFORE the refactor. If it passes pre-refactor, continue (this is a characterization test, not strict TDD red).

- [ ] **Step 3: Refactor `_approx_sharpe`**

Replace the whole `_approx_sharpe` function body in `database/trade_journal.py` with:

```python
def _approx_sharpe(trades: list) -> float:
    """
    Annualizes using this trade set's own actual frequency (trades per trading
    day, scaled to 252 trading days/year) rather than assuming one trade per
    day — this is an intraday system that can take several trades per day.
    Math lives in core.learning.metrics; only the annualization convention
    (252 trading days) is chosen here.
    """
    from core.learning.metrics import annualization_from_span, compute_trade_metrics

    pnls = [t.net_pnl for t in trades]
    timestamps = [t.entry_time or t.created_at for t in trades if t.entry_time or t.created_at]
    span_days = max((max(timestamps) - min(timestamps)).days, 1) if timestamps else 1
    ann = annualization_from_span(len(pnls), span_days, days_per_year=252)
    return compute_trade_metrics(pnls, annualization=ann)["sharpe"]
```

- [ ] **Step 4: Refactor `get_performance_stats`**

In `TradeJournal.get_performance_stats`, replace everything AFTER the `trades` query/`if not trades` guard (the manual wins/losses/profit_factor block and the return dict) with:

```python
        from core.learning.metrics import compute_trade_metrics

        pnls = [t.net_pnl for t in trades]
        m = compute_trade_metrics(pnls)  # sharpe comes from _approx_sharpe below
        return {
            "total_trades": m["total_trades"],
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
            "total_net_pnl": m["total_net_pnl"],
            "avg_pnl_per_trade": m["avg_pnl_per_trade"],
            "max_win": m["max_win"],
            "max_loss": m["max_loss"],
            "avg_win": m["avg_win"],
            "avg_loss": m["avg_loss"],
            "sharpe_approx": _approx_sharpe(trades),
        }
```

Keep the existing empty-trades early-return exactly as it is today (whatever dict it currently returns for zero trades — do not change it; read the function before editing).

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/database/test_trade_journal_metrics.py tests/core/learning/test_metrics.py -v`
Expected: PASS (10 passed)

Run: `python -m pytest tests/ -q`
Expected: all pass (regression check on the whole suite)

- [ ] **Step 6: Commit**

```bash
git add database/trade_journal.py tests/database/__init__.py tests/database/test_trade_journal_metrics.py
git commit -m "Delegate trade_journal metric math to shared core/learning/metrics"
```

---

## Task 3: Extract the simulation loop into `backtesting/simulation.py`

**Files:**
- Create: `backtesting/simulation.py`
- Modify: `backtesting/backtest_engine.py` (`run_backtest`, lines ~110–276)
- Test: `tests/backtesting/test_simulation.py`
- Create: `tests/backtesting/__init__.py` (empty)

**Interfaces:**
- Consumes: `compute_trade_metrics`, `annualization_from_span` (Task 1); `_apply_atr_levels` stays in `backtest_engine.py`.
- Produces (consumed by Task 6):
  `simulate_signals(df: pd.DataFrame, signals: pd.Series, sl_series, tp1_series, tp2_series, initial_capital: float, period_days: float) -> dict`
  with keys: `final_capital, total_return_pct, total_trades, winning_trades, losing_trades, win_rate, profit_factor, total_net_pnl, avg_pnl_per_trade, max_win, max_loss, max_drawdown, max_drawdown_pct, sharpe_ratio, sortino_ratio, trades, pnl_curve`. Zero trades → returns the same dict with `total_trades: 0`, empty `trades`/`pnl_curve`, all numeric metrics 0, `final_capital == initial_capital`.

- [ ] **Step 1: Write the failing test**

`tests/backtesting/test_simulation.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/backtesting/test_simulation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtesting.simulation'`

- [ ] **Step 3: Write `backtesting/simulation.py`**

Move the simulation loop + metric assembly out of `run_backtest` VERBATIM (it is the code from `# Simple simulation loop` down to the end of the `result = {...}` dict in `backtest_engine.py` lines ~114–269), wrapped as below. The only changes: metrics come from `compute_trade_metrics`, and the function takes explicit params instead of closing over `run_backtest` locals.

```python
"""
Bar-by-bar signal simulation, extracted from backtest_engine.run_backtest so
the strategy tournament can reuse the exact same execution model.
"""
import numpy as np
import pandas as pd
import yaml

from core.learning.metrics import annualization_from_span, compute_trade_metrics

_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))


def simulate_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    sl_series: pd.Series,
    tp1_series: pd.Series,
    tp2_series: pd.Series,
    initial_capital: float,
    period_days: float,
) -> dict:
    commission = _cfg["backtesting"]["commission_pct"]
    slippage = _cfg["backtesting"]["slippage_pct"]
    tp1_alloc = _cfg["signals"]["tp_allocation"]["tp1"]

    capital = initial_capital
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_idx = None
    entry_sl = entry_tp1 = entry_tp2 = 0.0
    direction = 0
    tp1_booked = False
    remaining_fraction = 1.0

    # A signal on bar i can only be filled at bar i+1's open — the earliest
    # realistic execution point — never at the same bar's own close.
    pending_direction = None
    pending_sl = pending_tp1 = pending_tp2 = 0.0

    for i in range(len(df)):
        row = df.iloc[i]
        close = row["close"]
        idx = df.index[i]

        if pending_direction is not None and not in_trade:
            direction = pending_direction
            entry_price = row["open"] * (1 + slippage if direction == 1 else 1 - slippage)
            entry_sl = pending_sl
            entry_tp1 = pending_tp1
            entry_tp2 = pending_tp2
            entry_idx = idx
            in_trade = True
            tp1_booked = False
            remaining_fraction = 1.0
            pending_direction = None

        if in_trade:
            tp1_hit = close >= entry_tp1 if direction == 1 else close <= entry_tp1
            tp2_hit = close >= entry_tp2 if direction == 1 else close <= entry_tp2
            sl_hit = close <= entry_sl if direction == 1 else close >= entry_sl
            base_qty = capital * _cfg["capital"]["max_per_trade_pct"] / entry_price

            if sl_hit:
                fill = entry_sl * (1 - slippage if direction == 1 else 1 + slippage)
                pnl = (fill - entry_price) * direction
                gross = pnl * remaining_fraction * base_qty
                net = gross - gross * commission * 2
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": idx,
                    "direction": direction, "entry": entry_price, "exit": fill,
                    "pnl": net, "exit_reason": "SL",
                })
                capital += net
                in_trade = False

            elif tp2_hit:
                fill = entry_tp2 * (1 - slippage if direction == 1 else 1 + slippage)
                pnl = (fill - entry_price) * direction
                gross = pnl * remaining_fraction * base_qty
                net = gross - gross * commission * 2
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": idx,
                    "direction": direction, "entry": entry_price, "exit": fill,
                    "pnl": net, "exit_reason": "TP2",
                })
                capital += net
                in_trade = False

            elif tp1_hit and not tp1_booked:
                # Book the configured TP1 fraction once, move SL to breakeven,
                # and leave the rest of the position open.
                tp1_booked = True
                entry_sl = entry_price
                fill = entry_tp1
                pnl = (fill - entry_price) * direction
                gross = pnl * tp1_alloc * base_qty
                net = gross - gross * commission
                capital += net
                remaining_fraction -= tp1_alloc

        elif signals.iloc[i] != 0 and not pd.isna(sl_series.iloc[i]) and pending_direction is None:
            pending_direction = int(signals.iloc[i])
            pending_sl = sl_series.iloc[i]
            pending_tp1 = tp1_series.iloc[i]
            pending_tp2 = tp2_series.iloc[i]

    if not trades:
        return {
            "final_capital": round(capital, 2), "total_return_pct": 0.0,
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0.0, "profit_factor": 0.0, "total_net_pnl": 0.0,
            "avg_pnl_per_trade": 0.0, "max_win": 0.0, "max_loss": 0.0,
            "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
            "trades": [], "pnl_curve": [],
        }

    pnls = [t["pnl"] for t in trades]
    ann = annualization_from_span(len(pnls), max(period_days, 1))
    m = compute_trade_metrics(pnls, annualization=ann)
    cumulative = np.cumsum(pnls)

    return {
        "final_capital": round(capital, 2),
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100, 2),
        "total_trades": m["total_trades"],
        "winning_trades": m["winning_trades"],
        "losing_trades": m["losing_trades"],
        "win_rate": round(m["win_rate"], 3),
        "profit_factor": round(m["profit_factor"], 3) if m["profit_factor"] != float("inf") else float("inf"),
        "total_net_pnl": round(m["total_net_pnl"], 2),
        "avg_pnl_per_trade": round(m["avg_pnl_per_trade"], 2),
        "max_win": round(m["max_win"], 2),
        "max_loss": round(m["max_loss"], 2),
        "max_drawdown": round(m["max_drawdown"], 2),
        "max_drawdown_pct": round(m["max_drawdown"] / initial_capital * 100, 2),
        "sharpe_ratio": round(m["sharpe"], 3),
        "sortino_ratio": round(m["sortino"], 3),
        "trades": trades,
        "pnl_curve": cumulative.tolist(),
    }
```

- [ ] **Step 4: Rewire `run_backtest` to call it**

In `backtesting/backtest_engine.py`, delete everything from the `# Simple simulation loop` comment (line ~114) through the end of the `result = {...}` assembly (line ~269), and replace with:

```python
    period_days = max((end - start).days, 1)
    sim = simulate_signals(df, signals, sl_series, tp1_series, tp2_series,
                           initial_capital, period_days)
    if sim["total_trades"] == 0:
        return {
            "symbol": symbol, "period": f"{start} → {end}", "total_trades": 0,
            "error": "No trades generated — check indicator parameters",
        }

    result = {
        "symbol": symbol,
        "period": f"{start} → {end}",
        "interval": interval,
        "initial_capital": initial_capital,
        **sim,
    }
```

Add the import at the top of `backtest_engine.py`:

```python
from backtesting.simulation import simulate_signals
```

Keep the final `logger.info("backtest_complete", ...)` block as-is (it reads keys that still exist).

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/backtesting/test_simulation.py -v`
Expected: PASS (3 passed)

Run: `python -m pytest tests/ -q`
Expected: all pass

Run: `python -c "from backtesting.backtest_engine import run_backtest; print('imports ok')"`
Expected: `imports ok`

- [ ] **Step 6: Commit**

```bash
git add backtesting/simulation.py backtesting/backtest_engine.py tests/backtesting/__init__.py tests/backtesting/test_simulation.py
git commit -m "Extract bar-by-bar simulation into backtesting/simulation.py"
```

---

## Task 4: `StrategyRanking` model + tournament config block

**Files:**
- Modify: `database/models.py` (append after `MLModelMetric`)
- Modify: `config/settings.yaml` (append at end)
- Test: `tests/database/test_strategy_ranking_model.py`

**Interfaces:**
- Produces (consumed by Task 6): `StrategyRanking` ORM class with columns
  `id, strategy_name, category, period_start, period_end, win_rate, profit_factor, sharpe_approx, max_drawdown, opportunity_capture_rate, composite_score, rank, promoted, status, created_at`.
- Config keys (consumed by Task 6): `strategy_tournament.enabled/run_day/backtest_lookback_days/min_backtest_trades_for_ranking/promote_top_n/score_weights/missed_opportunity`.

- [ ] **Step 1: Write the failing test**

`tests/database/test_strategy_ranking_model.py`:

```python
"""StrategyRanking table: schema round-trip on an in-memory SQLite DB."""
from datetime import datetime

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, StrategyRanking


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_strategy_ranking_round_trip():
    db = _session()
    row = StrategyRanking(
        strategy_name="ema_trend_following", category="trend",
        period_start=datetime(2026, 1, 1), period_end=datetime(2026, 6, 30),
        win_rate=0.61, profit_factor=1.8, sharpe_approx=1.2, max_drawdown=12000.0,
        opportunity_capture_rate=0.0, composite_score=0.74, rank=1,
        promoted=True, status="ranked",
    )
    db.add(row)
    db.commit()
    got = db.query(StrategyRanking).one()
    assert got.strategy_name == "ema_trend_following"
    assert got.promoted is True
    assert got.status == "ranked"
    assert got.created_at is not None


def test_tournament_config_block_present():
    cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))
    st = cfg["strategy_tournament"]
    assert st["enabled"] is True
    assert st["backtest_lookback_days"] == 180
    assert st["min_backtest_trades_for_ranking"] == 15
    assert st["promote_top_n"] == 3
    w = st["score_weights"]
    assert abs(w["profit_factor"] + w["sharpe"] + w["win_rate"] + w["max_drawdown"] - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_strategy_ranking_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'StrategyRanking'`

- [ ] **Step 3: Add the model**

Append to `database/models.py`:

```python
class StrategyRanking(Base):
    """One row per strategy per tournament run (weekly)."""
    __tablename__ = "strategy_rankings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    strategy_name = Column(String(60), nullable=False)
    category = Column(String(30))
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    sharpe_approx = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    opportunity_capture_rate = Column(Float, default=0.0)  # wired by Plan 3
    composite_score = Column(Float, default=0.0)
    rank = Column(Integer)                                  # 1 = best; NULL if not ranked
    promoted = Column(Boolean, default=False)
    status = Column(String(20), default="ranked")           # ranked | insufficient_data | errored
```

- [ ] **Step 4: Add the config block**

Append to `config/settings.yaml`:

```yaml
strategy_tournament:
  enabled: true
  run_day: "sunday"                     # scheduler wiring is Plan 4
  backtest_lookback_days: 180
  backtest_interval: "1h"               # yfinance intraday history limit ~730d at 1h
  min_backtest_trades_for_ranking: 15
  promote_top_n: 3
  score_weights:                        # must sum to 1.0
    profit_factor: 0.35
    sharpe: 0.25
    win_rate: 0.20
    max_drawdown: 0.20                  # inverted: lower drawdown scores higher
  profit_factor_cap: 10.0               # inf/huge PF capped before normalization
  missed_opportunity:                   # consumed by Plan 3
    enabled: true
    min_move_pct_threshold: 0.5
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/database/test_strategy_ranking_model.py -v`
Expected: PASS (2 passed)

Run: `python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add database/models.py config/settings.yaml tests/database/test_strategy_ranking_model.py
git commit -m "Add StrategyRanking table and strategy_tournament config block"
```

---

## Task 5: Tournament signal generation (per-strategy replay)

**Files:**
- Create: `backtesting/tournament.py` (first half — replay machinery; Task 6 adds orchestration to the same file)
- Test: `tests/backtesting/test_tournament_signals.py`

**Interfaces:**
- Consumes: `Strategy` (core/strategies/base.py), `detect_regime(df_5m, global_context, news_sentiment, options_context) -> RegimeResult`, `compute_all` from `core.analysis.technical`.
- Produces (consumed by Task 6, same file):
  - `neutral_contexts() -> tuple[dict, dict, dict]` — (options_context, global_context, news_sentiment) with the same neutral shapes the strategy tests use.
  - `precompute_regimes(df: pd.DataFrame, warmup: int) -> list` — index-aligned list, `None` before warmup, `RegimeResult` after.
  - `generate_strategy_signals(strategy, df, regimes, warmup: int = 200) -> pd.Series` — +1/-1/0 per bar; any exception inside `strategy.evaluate` propagates (caller isolates per-strategy in Task 6).

- [ ] **Step 1: Write the failing test**

`tests/backtesting/test_tournament_signals.py`:

```python
"""Per-strategy replay: evaluate() over a rolling window -> +1/-1/0 signal series."""
import pandas as pd
import pytz

from backtesting.tournament import neutral_contexts, precompute_regimes, generate_strategy_signals
from core.analysis.technical import compute_all
from core.strategies.base import Strategy

IST = pytz.timezone("Asia/Kolkata")


class _AlwaysCE(Strategy):
    name = "always_ce"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "CE", 60.0, ["TEST"], 1, ""


class _NeverFires(Strategy):
    name = "never_fires"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "NONE", 0.0, [], 0, "never"


def _enriched(rows=220):
    idx = pd.date_range("2026-01-05 09:15", periods=rows, freq="5min", tz=IST)
    closes = [22000.0 + 2 * i for i in range(rows)]
    df = pd.DataFrame({
        "open": closes, "high": [c + 5 for c in closes], "low": [c - 5 for c in closes],
        "close": closes, "volume": [1000] * rows,
    }, index=idx)
    return compute_all(df)


def test_always_ce_gives_plus_one_after_warmup():
    df = _enriched()
    regimes = precompute_regimes(df, warmup=200)
    signals = generate_strategy_signals(_AlwaysCE(), df, regimes, warmup=200)
    assert len(signals) == len(df)
    assert (signals.iloc[:200] == 0).all()
    assert (signals.iloc[200:] == 1).all()


def test_never_fires_gives_all_zero():
    df = _enriched()
    regimes = precompute_regimes(df, warmup=200)
    signals = generate_strategy_signals(_NeverFires(), df, regimes, warmup=200)
    assert (signals == 0).all()


def test_regimes_are_none_before_warmup_and_set_after():
    df = _enriched()
    regimes = precompute_regimes(df, warmup=200)
    assert regimes[0] is None
    assert regimes[199] is None
    assert regimes[200] is not None
    assert hasattr(regimes[200], "direction")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/backtesting/test_tournament_signals.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtesting.tournament'`

- [ ] **Step 3: Write the replay machinery**

`backtesting/tournament.py`:

```python
"""
Weekly strategy tournament: backtests every registered strategy over a rolling
lookback window, ranks them by a config-weighted composite score, and marks
the top N as promoted. Orchestration entry point: run_tournament().
"""
from datetime import date, datetime, timedelta

import pandas as pd
import structlog
import yaml

from core.signals.regime_detector import detect_regime
from core.strategies.base import Strategy

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))

WARMUP_BARS = 200  # same warmup the single-pipeline backtest uses


def neutral_contexts() -> tuple[dict, dict, dict]:
    """Neutral options/global/news contexts for historical replay.

    Historical values for PCR, OI, FII flow, VIX, news are not stored, so the
    replay feeds neutral values. Flow/news strategies therefore never fire in
    the tournament and end up status=insufficient_data — by design (spec:
    strategies with too few backtest trades are excluded from ranking).
    """
    options_context = {"pcr": 1.0, "max_pain": None, "oi_signal": 0, "chain_df": pd.DataFrame()}
    global_context = {
        "india_vix": 15.0, "vix_high_vol_regime": False, "vix_extreme": False,
        "fii_net_cash_cr": 0.0, "dii_net_cash_cr": 0.0, "global_score": 0,
        "gift_nifty_gap_pct": None,
    }
    news_sentiment = {"score": 0.0, "summary": "historical replay", "risk_events": [],
                      "method": "neutral_replay"}
    return options_context, global_context, news_sentiment


def precompute_regimes(df: pd.DataFrame, warmup: int = WARMUP_BARS) -> list:
    """One regime per bar, shared by all 20 strategies (computed once)."""
    options_context, global_context, news_sentiment = neutral_contexts()
    regimes: list = [None] * len(df)
    for i in range(warmup, len(df)):
        window = df.iloc[max(0, i - warmup):i + 1]
        try:
            regimes[i] = detect_regime(window, global_context, news_sentiment, options_context)
        except Exception as e:  # regime failure on one bar must not kill the run
            logger.warning("tournament_regime_failed", bar=str(df.index[i]), error=str(e))
            regimes[i] = None
    return regimes


def generate_strategy_signals(
    strategy: Strategy,
    df: pd.DataFrame,
    regimes: list,
    warmup: int = WARMUP_BARS,
) -> pd.Series:
    """Replay one strategy over the enriched df. Exceptions propagate to the caller."""
    options_context, global_context, news_sentiment = neutral_contexts()
    signals = pd.Series(0, index=df.index)
    for i in range(warmup, len(df)):
        regime = regimes[i]
        if regime is None:
            continue
        window = df.iloc[max(0, i - warmup):i + 1]
        direction, _score, _inds, _conf, _reason = strategy.evaluate(
            window, regime, options_context, global_context, news_sentiment
        )
        if direction == "CE":
            signals.iloc[i] = 1
        elif direction == "PE":
            signals.iloc[i] = -1
    return signals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/backtesting/test_tournament_signals.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backtesting/tournament.py tests/backtesting/test_tournament_signals.py
git commit -m "Add tournament replay machinery (regimes + per-strategy signals)"
```

---

## Task 6: Tournament orchestration — scoring, ranking, persistence, promotion

**Files:**
- Modify: `backtesting/tournament.py` (append to the Task 5 file)
- Test: `tests/backtesting/test_tournament_run.py`

**Interfaces:**
- Consumes: `simulate_signals` (Task 3), `_apply_atr_levels` from `backtesting.backtest_engine`, `compute_trade_metrics`/`annualization_from_span` (Task 1), `StrategyRanking` (Task 4), `get_all_strategies` from `core.strategies.registry`, `fetch_historical_yfinance` from `core.market_data.historical`, `compute_all`.
- Produces:
  - `compute_composite_scores(results: list[dict], weights: dict, pf_cap: float) -> list[dict]` — adds `composite_score` to each ranked result via min-max normalization (drawdown inverted).
  - `run_tournament(strategies=None, df=None, session_factory=None, now=None) -> list[dict]` — full run; every param injectable for tests; returns the result rows it wrote.
  - `python -m backtesting.tournament` manual entry point.

- [ ] **Step 1: Write the failing test**

`tests/backtesting/test_tournament_run.py`:

```python
"""End-to-end tournament: ranking, statuses, failure isolation, promotion."""
import pandas as pd
import pytz
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backtesting.tournament import compute_composite_scores, run_tournament
from core.analysis.technical import compute_all
from core.strategies.base import Strategy
from database.models import Base, StrategyRanking

IST = pytz.timezone("Asia/Kolkata")


class _GoodStrategy(Strategy):
    """Fires CE every 10th bar in an uptrend -> plenty of winning trades."""
    name = "good_strategy"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if len(df) % 10 == 0:
            return "CE", 70.0, ["TEST"], 1, ""
        return "NONE", 0.0, [], 0, "off-cycle"


class _QuietStrategy(Strategy):
    """Never fires -> insufficient_data."""
    name = "quiet_strategy"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "NONE", 0.0, [], 0, "never"


class _BrokenStrategy(Strategy):
    """Raises -> errored, must not kill the run."""
    name = "broken_strategy"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        raise RuntimeError("boom")


def _uptrend_df(rows=420):
    idx = pd.date_range("2026-01-05 09:15", periods=rows, freq="1h", tz=IST)
    closes = [22000.0 + 5 * i for i in range(rows)]
    df = pd.DataFrame({
        "open": closes, "high": [c + 8 for c in closes], "low": [c - 8 for c in closes],
        "close": closes, "volume": [1000] * rows,
    }, index=idx)
    return compute_all(df)


def _mem_sessions():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_run_tournament_ranks_and_isolates_failures():
    sessions = _mem_sessions()
    results = run_tournament(
        strategies=[_GoodStrategy(), _QuietStrategy(), _BrokenStrategy()],
        df=_uptrend_df(),
        session_factory=sessions,
    )
    by_name = {r["strategy_name"]: r for r in results}
    assert by_name["broken_strategy"]["status"] == "errored"
    assert by_name["quiet_strategy"]["status"] == "insufficient_data"
    assert by_name["good_strategy"]["status"] == "ranked"
    assert by_name["good_strategy"]["rank"] == 1
    assert by_name["good_strategy"]["promoted"] is True
    # non-ranked strategies are never promoted
    assert by_name["quiet_strategy"]["promoted"] is False
    assert by_name["broken_strategy"]["promoted"] is False

    db = sessions()
    rows = db.query(StrategyRanking).all()
    assert len(rows) == 3
    promoted = [r for r in rows if r.promoted]
    assert len(promoted) == 1  # only 1 ranked strategy, promote_top_n=3 caps at available


def test_compute_composite_scores_weighting_and_inversion():
    results = [
        {"strategy_name": "a", "status": "ranked", "win_rate": 0.6,
         "profit_factor": 2.0, "sharpe_approx": 1.5, "max_drawdown": 1000.0},
        {"strategy_name": "b", "status": "ranked", "win_rate": 0.4,
         "profit_factor": 1.0, "sharpe_approx": 0.5, "max_drawdown": 5000.0},
    ]
    weights = {"profit_factor": 0.35, "sharpe": 0.25, "win_rate": 0.20, "max_drawdown": 0.20}
    scored = compute_composite_scores(results, weights, pf_cap=10.0)
    a = next(r for r in scored if r["strategy_name"] == "a")
    b = next(r for r in scored if r["strategy_name"] == "b")
    # a is better on every dimension (incl. lower drawdown) -> normalized 1.0 vs 0.0
    assert abs(a["composite_score"] - 1.0) < 1e-9
    assert abs(b["composite_score"] - 0.0) < 1e-9


def test_compute_composite_scores_caps_infinite_profit_factor():
    results = [
        {"strategy_name": "a", "status": "ranked", "win_rate": 0.5,
         "profit_factor": float("inf"), "sharpe_approx": 1.0, "max_drawdown": 100.0},
        {"strategy_name": "b", "status": "ranked", "win_rate": 0.5,
         "profit_factor": 1.0, "sharpe_approx": 1.0, "max_drawdown": 100.0},
    ]
    weights = {"profit_factor": 1.0, "sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0}
    scored = compute_composite_scores(results, weights, pf_cap=10.0)
    a = next(r for r in scored if r["strategy_name"] == "a")
    assert a["composite_score"] == 1.0  # capped, not inf/nan
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/backtesting/test_tournament_run.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_composite_scores'`

- [ ] **Step 3: Append orchestration to `backtesting/tournament.py`**

```python
def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def compute_composite_scores(results: list[dict], weights: dict, pf_cap: float) -> list[dict]:
    """Min-max normalize each metric across ranked strategies, blend by weights.
    Drawdown is inverted (lower is better). Mutates and returns `results`."""
    ranked = [r for r in results if r["status"] == "ranked"]
    if not ranked:
        return results

    pfs = _minmax([min(r["profit_factor"], pf_cap) for r in ranked])
    sharpes = _minmax([r["sharpe_approx"] for r in ranked])
    win_rates = _minmax([r["win_rate"] for r in ranked])
    drawdowns = _minmax([r["max_drawdown"] for r in ranked])

    for r, pf, sh, wr, dd in zip(ranked, pfs, sharpes, win_rates, drawdowns):
        r["composite_score"] = (
            weights["profit_factor"] * pf
            + weights["sharpe"] * sh
            + weights["win_rate"] * wr
            + weights["max_drawdown"] * (1.0 - dd)
        )
    return results


def run_tournament(
    strategies: list | None = None,
    df: pd.DataFrame | None = None,
    session_factory=None,
    now: datetime | None = None,
) -> list[dict]:
    """Backtest every strategy, rank, persist StrategyRanking rows, promote top N.

    All parameters are injectable for tests; production callers pass nothing.
    """
    from backtesting.backtest_engine import _apply_atr_levels
    from backtesting.simulation import simulate_signals
    from core.analysis.technical import compute_all
    from database.models import StrategyRanking

    t_cfg = _cfg["strategy_tournament"]
    now = now or datetime.utcnow()
    lookback_days = t_cfg["backtest_lookback_days"]
    period_end = now
    period_start = now - timedelta(days=lookback_days)

    if strategies is None:
        from core.strategies.registry import get_all_strategies
        strategies = get_all_strategies()

    if df is None:
        from core.market_data.historical import fetch_historical_yfinance
        raw = fetch_historical_yfinance(
            "NIFTY",
            start=period_start.date(),
            end=period_end.date(),
            interval=t_cfg.get("backtest_interval", "1h"),
        )
        if raw.empty:
            logger.error("tournament_no_data")
            return []
        df = compute_all(raw)

    if session_factory is None:
        from database.trade_journal import SessionLocal, init_db
        init_db()  # ensures the new strategy_rankings table exists
        session_factory = SessionLocal

    initial_capital = _cfg["backtesting"]["initial_capital"]
    min_trades = t_cfg["min_backtest_trades_for_ranking"]
    period_days = max((df.index[-1] - df.index[0]).days, 1)

    regimes = precompute_regimes(df)
    results: list[dict] = []

    for strategy in strategies:
        base = {
            "strategy_name": strategy.name, "category": strategy.category,
            "win_rate": 0.0, "profit_factor": 0.0, "sharpe_approx": 0.0,
            "max_drawdown": 0.0, "opportunity_capture_rate": 0.0,
            "composite_score": 0.0, "rank": None, "promoted": False,
        }
        try:
            signals = generate_strategy_signals(strategy, df, regimes)
            sl_s, tp1_s, tp2_s = _apply_atr_levels(df, signals)
            sim = simulate_signals(df, signals, sl_s, tp1_s, tp2_s,
                                   initial_capital, period_days)
            if sim["total_trades"] < min_trades:
                results.append({**base, "status": "insufficient_data"})
                logger.info("tournament_insufficient_data",
                            strategy=strategy.name, trades=sim["total_trades"])
                continue
            results.append({
                **base, "status": "ranked",
                "win_rate": sim["win_rate"],
                "profit_factor": sim["profit_factor"],
                "sharpe_approx": sim["sharpe_ratio"],
                "max_drawdown": sim["max_drawdown"],
            })
        except Exception as e:
            logger.error("tournament_strategy_errored", strategy=strategy.name, error=str(e))
            results.append({**base, "status": "errored"})

    weights = t_cfg["score_weights"]
    pf_cap = t_cfg.get("profit_factor_cap", 10.0)
    compute_composite_scores(results, weights, pf_cap)

    ranked = sorted(
        [r for r in results if r["status"] == "ranked"],
        key=lambda r: r["composite_score"], reverse=True,
    )
    promote_top_n = t_cfg["promote_top_n"]
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
        r["promoted"] = i < promote_top_n

    db = session_factory()
    try:
        for r in results:
            db.add(StrategyRanking(period_start=period_start, period_end=period_end, **r))
        db.commit()
    finally:
        db.close()

    logger.info("tournament_complete",
                total=len(results),
                ranked=len(ranked),
                promoted=[r["strategy_name"] for r in ranked if r["promoted"]])
    return results


if __name__ == "__main__":
    rows = run_tournament()
    for r in sorted(rows, key=lambda x: (x["rank"] is None, x["rank"] or 0)):
        flag = "PROMOTED" if r["promoted"] else r["status"]
        print(f"{r['rank'] or '-':>3}  {r['strategy_name']:<40} "
              f"score={r['composite_score']:.3f}  {flag}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/backtesting/test_tournament_run.py -v`
Expected: PASS (3 passed)

Note: `run_tournament` in the first test passes `df` already enriched and `now=None`; `period_start/period_end` are derived from wall clock — fine, the test doesn't assert on them.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass (~85+ tests)

- [ ] **Step 6: Commit**

```bash
git add backtesting/tournament.py tests/backtesting/test_tournament_run.py
git commit -m "Add tournament orchestration: scoring, ranking, persistence, promotion"
```

---

## Task 7: Full verification + push

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `python -m pytest tests/ -q`
Expected: 0 failed.

- [ ] **Step 2: Smoke the manual entry point imports**

Run: `python -c "import backtesting.tournament as t; print(callable(t.run_tournament))"`
Expected: `True`

(Do NOT run a real tournament here — it fetches 180 days of yfinance data and takes minutes; the scheduler wiring in Plan 4 owns production runs.)

- [ ] **Step 3: Push**

```bash
git push origin main
```

---

## Self-Review Notes

- **Spec coverage:** spec §2 (tournament.py, lookback config, shared metrics, composite weights, StrategyRanking writes, insufficient_data/errored statuses) → Tasks 4–6. Metrics extraction ("one implementation for live-escalation and tournament") → Tasks 1–2. Reuse of `backtest_engine` execution model → Task 3. `opportunity_capture_rate` column exists but stays 0.0 (Plan 3 wires it) — noted in Global Constraints. Scheduler/promotion consumption (`get_active_strategies`) is Plans 3–4, not silently dropped.
- **Known simplification (documented in code):** historical replay uses neutral options/global/news contexts, so pure flow/news strategies won't fire in tournaments and will sit at `insufficient_data` until Plan 3+ adds historical context capture. This matches the spec's insufficient-data handling rather than fabricating fake historical PCR/FII data.
- **Type consistency:** `simulate_signals` return keys checked against `run_backtest`'s existing result dict; `StrategyRanking` kwargs in `run_tournament` match the Task 4 columns exactly (`rank`, `promoted`, `status`, etc.); `generate_strategy_signals(strategy, df, regimes, warmup)` matches Task 5's signature when called in Task 6 (uses default warmup).
- **Placeholder scan:** none — every step has complete runnable code.

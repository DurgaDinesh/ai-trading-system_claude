# Mistake Learning + System Integration (AI Brain Plans 3 & 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `core/learning/mistake_analyzer.py` (per-strategy score learning + daily missed-opportunity detection into a new `MissedOpportunity` table), wire `opportunity_capture_rate` into the tournament composite score, and integrate everything into the live system: promoted-strategy signal selection in `strategy_selector.py`, weekly tournament + daily missed-opportunity scheduler jobs, and a dashboard Strategy Leaderboard page.

**Architecture:** The mistake analyzer reuses the tournament's replay machinery (`precompute_regimes` / `generate_strategy_signals`) to replay each day's bars through all 20 strategies after close; a day with a move ≥ `min_move_pct_threshold` that no *promoted* strategy signaled but some *non-promoted* strategy would have becomes a `MissedOpportunity` row. The weekly tournament reads those rows back as a per-strategy capture rate and blends it into the composite score via a separate `opportunity_capture_weight` (the existing 4-metric `score_weights` block stays untouched — a committed test asserts it sums to 1.0). Live selection: `registry.get_active_strategies()` returns the latest tournament's promoted strategies in rank order; `strategy_selector.select_best_signal()` runs them per bar and picks the best valid signal, falling back to the original `signal_engine` pipeline only when no tournament has ever promoted anything.

**Tech Stack:** Python 3.14, pandas, SQLAlchemy, APScheduler, FastAPI + Jinja2, pytest 9.

## Global Constraints

- Run tests with `python -m pytest <path> -v` from the repo root (`E:\Algo Trading`).
- Do NOT change: `run_backtest()` signature/result keys, `journal.get_performance_stats()` keys, the `strategy_tournament.score_weights` config block (4 keys, sum 1.0 — locked by `tests/database/test_strategy_ranking_model.py::test_tournament_config_block_present`).
- `compute_composite_scores` gains `capture_weight` as a keyword arg with default `0.0` so the two existing call-sites/tests in `tests/backtesting/test_tournament_run.py` keep passing unchanged.
- Composite formula with capture: `composite = (1 - capture_weight) * four_metric_blend + capture_weight * minmax(opportunity_capture_rate)`.
- Per-strategy live learning adjusts `composite_score` on the LATEST `StrategyRanking` row for that strategy, clamped to [0.0, 1.0], step `strategy_tournament.live_score_adjustment_step` (default 0.02) — the same clamp-and-normalize spirit as `adaptive_weights.py`. Hardcoded-pipeline trades store a regime label (e.g. `BUY_CE_ATM_PLUS_1`) in `Trade.strategy` which matches no ranking row — the update is a harmless no-op returning `None`.
- `select_best_signal` falls back to `signal_engine.generate_signal` ONLY when zero strategies are promoted. If promoted strategies exist but none fire, it returns an invalid signal (no fallback) — spec §4.
- Every strategy failure (live selection, missed-opp replay) is isolated per strategy: logged and skipped, never a crashed cycle (spec "Error handling").
- Missed-opportunity scan writes at most one row per direction (CE/PE) per day.
- `config/settings.yaml` already has `strategy_tournament.missed_opportunity` — only ADD new keys, never duplicate the block.

---

## Task 1: `MissedOpportunity` model + new config keys

**Files:**
- Modify: `database/models.py` (append after `StrategyRanking`)
- Modify: `config/settings.yaml` (add 2 keys inside the existing `strategy_tournament:` block)
- Test: `tests/database/test_missed_opportunity_model.py`

**Interfaces:**
- Produces (consumed by Tasks 4, 5, 8): `MissedOpportunity` ORM class with columns
  `id, date, underlying, move_pct, direction, would_have_matched (JSON), reason, created_at`.
- Config keys (consumed by Tasks 3, 5): `strategy_tournament.opportunity_capture_weight` (0.10), `strategy_tournament.live_score_adjustment_step` (0.02).

- [ ] **Step 1: Write the failing test**

`tests/database/test_missed_opportunity_model.py`:

```python
"""MissedOpportunity table: schema round-trip + new tournament config keys."""
from datetime import datetime

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, MissedOpportunity


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_missed_opportunity_round_trip():
    db = _session()
    row = MissedOpportunity(
        date=datetime(2026, 7, 3),
        underlying="NIFTY",
        move_pct=0.85,
        direction="CE",
        would_have_matched=["rsi_mean_reversion", "vwap_reversion"],
        reason="no_promoted_strategy_signaled",
    )
    db.add(row)
    db.commit()
    got = db.query(MissedOpportunity).one()
    assert got.underlying == "NIFTY"
    assert got.direction == "CE"
    assert got.would_have_matched == ["rsi_mean_reversion", "vwap_reversion"]
    assert got.reason == "no_promoted_strategy_signaled"
    assert got.created_at is not None


def test_new_tournament_config_keys():
    cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))
    st = cfg["strategy_tournament"]
    assert st["opportunity_capture_weight"] == 0.10
    assert st["live_score_adjustment_step"] == 0.02
    # the original 4-weight block must remain untouched
    w = st["score_weights"]
    assert abs(w["profit_factor"] + w["sharpe"] + w["win_rate"] + w["max_drawdown"] - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_missed_opportunity_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'MissedOpportunity'`

- [ ] **Step 3: Add the model**

Append to `database/models.py`:

```python
class MissedOpportunity(Base):
    """A profitable move no promoted strategy signaled — logged daily after close."""
    __tablename__ = "missed_opportunities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    date = Column(DateTime, nullable=False)                # trading day of the move
    underlying = Column(String(20), default="NIFTY")
    move_pct = Column(Float, default=0.0)                  # size of the missed move (%)
    direction = Column(String(2))                          # CE | PE
    would_have_matched = Column(JSON)                      # non-promoted strategy names that fired
    reason = Column(String(60), default="no_promoted_strategy_signaled")
```

- [ ] **Step 4: Add the config keys**

In `config/settings.yaml`, inside the existing `strategy_tournament:` block, add these two lines directly after `profit_factor_cap: 10.0`:

```yaml
  opportunity_capture_weight: 0.10      # blend share for missed-opp capture rate (score_weights stays sum=1.0)
  live_score_adjustment_step: 0.02      # per-trade nudge to a strategy's latest ranking score
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/database/test_missed_opportunity_model.py -v`
Expected: PASS (2 passed)

Run: `python -m pytest tests/ -q`
Expected: all pass (regression — especially `test_tournament_config_block_present`)

- [ ] **Step 6: Commit**

```bash
git add database/models.py config/settings.yaml tests/database/test_missed_opportunity_model.py
git commit -m "Add MissedOpportunity table and tournament learning config keys"
```

---

## Task 2: Registry — promoted-strategy lookup

**Files:**
- Modify: `core/strategies/registry.py` (append two functions)
- Test: `tests/core/strategies/test_registry_active.py`

**Interfaces:**
- Consumes: `StrategyRanking` (existing), `get_all_strategies()` (existing, same file).
- Produces (consumed by Tasks 4, 6):
  - `get_promoted_strategy_names(session_factory=None) -> list[str]` — promoted names from the latest tournament run (rows sharing the max `period_end`), ordered by `rank` ascending. Returns `[]` on no rows or ANY error (DB down → spec: last-known set / fallback handled by caller).
  - `get_active_strategies(session_factory=None) -> list[Strategy]` — those names mapped to instances via `get_all_strategies()`, preserving rank order; names with no matching class are silently dropped.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_registry_active.py`:

```python
"""get_active_strategies: promoted names from the latest tournament run -> instances."""
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.strategies.registry import (
    get_active_strategies,
    get_all_strategies,
    get_promoted_strategy_names,
)
from database.models import Base, StrategyRanking


def _mem_sessions():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_promoted_names_from_latest_run_in_rank_order():
    all_strats = get_all_strategies()
    n1, n2 = all_strats[0].name, all_strats[1].name
    sessions = _mem_sessions()
    db = sessions()
    db.add_all([
        # latest run (period_end 2026-07-05)
        StrategyRanking(strategy_name=n2, period_end=datetime(2026, 7, 5), promoted=True, rank=1, status="ranked"),
        StrategyRanking(strategy_name=n1, period_end=datetime(2026, 7, 5), promoted=True, rank=2, status="ranked"),
        StrategyRanking(strategy_name="ghost_strategy", period_end=datetime(2026, 7, 5), promoted=True, rank=3, status="ranked"),
        StrategyRanking(strategy_name=n1, period_end=datetime(2026, 7, 5), promoted=False, rank=None, status="errored"),
        # an older run that must be ignored
        StrategyRanking(strategy_name=n1, period_end=datetime(2026, 6, 28), promoted=True, rank=1, status="ranked"),
    ])
    db.commit()
    db.close()

    names = get_promoted_strategy_names(session_factory=sessions)
    assert names == [n2, n1, "ghost_strategy"]

    active = get_active_strategies(session_factory=sessions)
    assert [s.name for s in active] == [n2, n1]  # ghost has no class -> dropped


def test_promoted_names_empty_when_no_rows():
    assert get_promoted_strategy_names(session_factory=_mem_sessions()) == []
    assert get_active_strategies(session_factory=_mem_sessions()) == []


def test_promoted_names_empty_on_db_error():
    def broken_factory():
        raise RuntimeError("db unavailable")
    assert get_promoted_strategy_names(session_factory=broken_factory) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/strategies/test_registry_active.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_promoted_strategy_names'`

- [ ] **Step 3: Write the implementation**

Append to `core/strategies/registry.py`:

```python
def get_promoted_strategy_names(session_factory=None) -> list[str]:
    """Promoted strategy names from the latest tournament run, ordered by rank.

    Returns [] if no tournament has run yet or the DB is unavailable — callers
    treat an empty list as "fall back to the hardcoded pipeline" (spec §4).
    """
    try:
        from sqlalchemy import func

        from database.models import StrategyRanking

        if session_factory is None:
            from database.trade_journal import SessionLocal
            session_factory = SessionLocal
        db = session_factory()
        try:
            latest_period_end = db.query(func.max(StrategyRanking.period_end)).scalar()
            if latest_period_end is None:
                return []
            rows = (
                db.query(StrategyRanking)
                .filter(
                    StrategyRanking.period_end == latest_period_end,
                    StrategyRanking.promoted.is_(True),
                )
                .order_by(StrategyRanking.rank)
                .all()
            )
            return [r.strategy_name for r in rows]
        finally:
            db.close()
    except Exception:
        return []


def get_active_strategies(session_factory=None) -> list[Strategy]:
    """Instances of the currently promoted strategies, in tournament-rank order."""
    names = get_promoted_strategy_names(session_factory)
    by_name = {s.name: s for s in get_all_strategies()}
    return [by_name[n] for n in names if n in by_name]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/strategies/test_registry_active.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/registry.py tests/core/strategies/test_registry_active.py
git commit -m "Add promoted-strategy lookup to registry (get_active_strategies)"
```

---

## Task 3: mistake_analyzer — per-strategy score learning on trade close

**Files:**
- Create: `core/learning/mistake_analyzer.py`
- Modify: `core/learning/performance_tracker.py` (`on_trade_closed`, after the adaptive-weights call at line ~41)
- Test: `tests/core/learning/test_mistake_analyzer_scores.py`

**Interfaces:**
- Consumes: `StrategyRanking` (existing), config key `strategy_tournament.live_score_adjustment_step` (Task 1).
- Produces (consumed by `performance_tracker`):
  - `update_strategy_score_on_trade_close(strategy_name: str, is_win: bool, session_factory=None) -> float | None` — adjusts the latest ranking row's `composite_score` by ±step, clamped to [0.0, 1.0]; returns the new score, or `None` if no row exists for that strategy.

- [ ] **Step 1: Write the failing test**

`tests/core/learning/test_mistake_analyzer_scores.py`:

```python
"""Per-strategy score learning: latest StrategyRanking row nudged on trade close."""
import math
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.learning.mistake_analyzer import update_strategy_score_on_trade_close
from database.models import Base, StrategyRanking


def _mem_sessions():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _add_row(sessions, name, score, created_at):
    db = sessions()
    db.add(StrategyRanking(
        strategy_name=name, composite_score=score, status="ranked",
        created_at=created_at, period_end=created_at,
    ))
    db.commit()
    db.close()


def test_win_bumps_latest_row_only():
    sessions = _mem_sessions()
    _add_row(sessions, "ema_trend_following", 0.50, datetime(2026, 6, 28))
    _add_row(sessions, "ema_trend_following", 0.70, datetime(2026, 7, 5))

    new_score = update_strategy_score_on_trade_close(
        "ema_trend_following", is_win=True, session_factory=sessions
    )
    assert math.isclose(new_score, 0.72)

    db = sessions()
    rows = {r.created_at: r.composite_score for r in db.query(StrategyRanking).all()}
    assert math.isclose(rows[datetime(2026, 7, 5)], 0.72)   # latest updated
    assert math.isclose(rows[datetime(2026, 6, 28)], 0.50)  # older untouched
    db.close()


def test_loss_reduces_and_clamps_at_zero():
    sessions = _mem_sessions()
    _add_row(sessions, "vwap_reversion", 0.01, datetime(2026, 7, 5))
    new_score = update_strategy_score_on_trade_close(
        "vwap_reversion", is_win=False, session_factory=sessions
    )
    assert new_score == 0.0


def test_win_clamps_at_one():
    sessions = _mem_sessions()
    _add_row(sessions, "vwap_reversion", 0.995, datetime(2026, 7, 5))
    new_score = update_strategy_score_on_trade_close(
        "vwap_reversion", is_win=True, session_factory=sessions
    )
    assert new_score == 1.0


def test_unknown_strategy_returns_none():
    sessions = _mem_sessions()
    assert update_strategy_score_on_trade_close(
        "BUY_CE_ATM_PLUS_1", is_win=True, session_factory=sessions
    ) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/learning/test_mistake_analyzer_scores.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.learning.mistake_analyzer'`

- [ ] **Step 3: Write the implementation**

Create `core/learning/mistake_analyzer.py`:

```python
"""
Mistake learning (spec §3):

1. Losing-trade learning — per-strategy win/loss outcomes nudge that strategy's
   latest StrategyRanking.composite_score (clamped to [0, 1]), alongside the
   existing global adaptive_weights update which continues unchanged.
2. Missed-opportunity detection — a daily post-close replay of the day's bars
   through every registered strategy; moves nobody promoted signaled are
   logged to MissedOpportunity and feed the next tournament's capture rate.
"""
from datetime import datetime

import pandas as pd
import structlog
import yaml

from database.models import MissedOpportunity, StrategyRanking

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))

# A single day of 5-min bars (~75) is far below the tournament's 200-bar
# warmup, so the daily scan is fed several days of history and only the
# scan-date bars are evaluated. This floor guards against degenerate input.
MIN_WARMUP_BARS = 50


def update_strategy_score_on_trade_close(
    strategy_name: str,
    is_win: bool,
    session_factory=None,
):
    """Nudge the strategy's latest ranking score by ±live_score_adjustment_step.

    Returns the new score, or None when the strategy has no ranking row —
    which is always the case for hardcoded-pipeline trades whose Trade.strategy
    holds a regime label like "BUY_CE_ATM_PLUS_1", not an archetype name.
    """
    step = _cfg["strategy_tournament"].get("live_score_adjustment_step", 0.02)
    if session_factory is None:
        from database.trade_journal import SessionLocal
        session_factory = SessionLocal

    db = session_factory()
    try:
        row = (
            db.query(StrategyRanking)
            .filter(StrategyRanking.strategy_name == strategy_name)
            .order_by(StrategyRanking.created_at.desc(), StrategyRanking.id.desc())
            .first()
        )
        if row is None:
            return None
        adjusted = row.composite_score + (step if is_win else -step)
        row.composite_score = min(1.0, max(0.0, adjusted))
        db.commit()
        logger.info(
            "strategy_score_adjusted",
            strategy=strategy_name, is_win=is_win, new_score=row.composite_score,
        )
        return row.composite_score
    finally:
        db.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/learning/test_mistake_analyzer_scores.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Wire into performance_tracker**

In `core/learning/performance_tracker.py`, inside `on_trade_closed`, directly after the line `update_weights_on_trade_close(indicators_triggered, is_win, trade_number)`, insert:

```python
        # Per-strategy score learning (AI brain): nudge the closed trade's
        # strategy in the latest tournament ranking. No-op for hardcoded-
        # pipeline trades (their strategy field is a regime label, not an
        # archetype), and never allowed to break the trade-close flow.
        try:
            trade = journal.get_trade(trade_id)
            if trade and trade.strategy:
                from core.learning.mistake_analyzer import update_strategy_score_on_trade_close
                update_strategy_score_on_trade_close(trade.strategy, is_win)
        except Exception as e:
            logger.warning("strategy_score_update_failed", trade_id=trade_id, error=str(e))
```

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add core/learning/mistake_analyzer.py core/learning/performance_tracker.py tests/core/learning/test_mistake_analyzer_scores.py
git commit -m "Add per-strategy score learning on trade close (mistake_analyzer)"
```

---

## Task 4: mistake_analyzer — missed-opportunity detection

**Files:**
- Modify: `core/learning/mistake_analyzer.py` (append)
- Test: `tests/core/learning/test_missed_opportunities.py`

**Interfaces:**
- Consumes: `precompute_regimes`, `generate_strategy_signals` (backtesting/tournament.py), `get_all_strategies`, `get_promoted_strategy_names` (Task 2), `MissedOpportunity` (Task 1), config `strategy_tournament.missed_opportunity.min_move_pct_threshold`.
- Produces (consumed by Task 7's scheduler job):
  - `detect_missed_opportunities(df: pd.DataFrame, scan_date: date, strategies=None, promoted_names=None, underlying: str = "NIFTY", session_factory=None) -> list[dict]` — `df` must be enriched (`compute_all`) and include several days of history before `scan_date` for indicator warmup; returns the rows written (possibly empty), each with keys `date, underlying, move_pct, direction, would_have_matched, reason`.

- [ ] **Step 1: Write the failing test**

`tests/core/learning/test_missed_opportunities.py`:

```python
"""Missed-opportunity detection: big move + silent promoted set -> logged row."""
from datetime import date

import pandas as pd
import pytz
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.analysis.technical import compute_all
from core.learning.mistake_analyzer import detect_missed_opportunities
from core.strategies.base import Strategy
from database.models import Base, MissedOpportunity

IST = pytz.timezone("Asia/Kolkata")
SCAN_DATE = date(2026, 1, 5)


class _AlwaysCE(Strategy):
    name = "mo_always_ce"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "CE", 60.0, ["TEST"], 1, ""


class _Quiet(Strategy):
    name = "mo_quiet"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "NONE", 0.0, [], 0, "never"


def _mem_sessions():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _replay_df(day_trend: float = 30.0):
    """200 flat warmup bars on 2026-01-02 + 75 scan-day bars on 2026-01-05."""
    hist_idx = pd.date_range("2026-01-02 09:15", periods=200, freq="5min", tz=IST)
    day_idx = pd.date_range("2026-01-05 09:15", periods=75, freq="5min", tz=IST)
    idx = hist_idx.append(day_idx)
    closes = [22000.0] * 200 + [22000.0 + day_trend * i for i in range(75)]
    df = pd.DataFrame({
        "open": closes,
        "high": [c + 5 for c in closes],
        "low": [c - 5 for c in closes],
        "close": closes,
        "volume": [1000] * len(closes),
    }, index=idx)
    return compute_all(df)


def test_big_move_missed_by_promoted_is_logged():
    sessions = _mem_sessions()
    rows = detect_missed_opportunities(
        _replay_df(), SCAN_DATE,
        strategies=[_AlwaysCE(), _Quiet()],
        promoted_names=["mo_quiet"],
        session_factory=sessions,
    )
    assert len(rows) == 1
    assert rows[0]["direction"] == "CE"
    assert rows[0]["move_pct"] >= 0.5
    assert rows[0]["would_have_matched"] == ["mo_always_ce"]
    assert rows[0]["reason"] == "no_promoted_strategy_signaled"

    db = sessions()
    saved = db.query(MissedOpportunity).all()
    assert len(saved) == 1
    assert saved[0].would_have_matched == ["mo_always_ce"]
    db.close()


def test_move_caught_by_promoted_is_not_logged():
    sessions = _mem_sessions()
    rows = detect_missed_opportunities(
        _replay_df(), SCAN_DATE,
        strategies=[_AlwaysCE()],
        promoted_names=["mo_always_ce"],
        session_factory=sessions,
    )
    assert rows == []
    db = sessions()
    assert db.query(MissedOpportunity).count() == 0
    db.close()


def test_small_move_below_threshold_is_not_logged():
    rows = detect_missed_opportunities(
        _replay_df(day_trend=0.0), SCAN_DATE,
        strategies=[_AlwaysCE(), _Quiet()],
        promoted_names=["mo_quiet"],
        session_factory=_mem_sessions(),
    )
    assert rows == []


def test_insufficient_warmup_returns_empty():
    day_idx = pd.date_range("2026-01-05 09:15", periods=75, freq="5min", tz=IST)
    closes = [22000.0 + 30 * i for i in range(75)]
    df = compute_all(pd.DataFrame({
        "open": closes, "high": [c + 5 for c in closes],
        "low": [c - 5 for c in closes], "close": closes,
        "volume": [1000] * 75,
    }, index=day_idx))
    rows = detect_missed_opportunities(
        df, SCAN_DATE,
        strategies=[_AlwaysCE()],
        promoted_names=[],
        session_factory=_mem_sessions(),
    )
    assert rows == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/learning/test_missed_opportunities.py -v`
Expected: FAIL — `ImportError: cannot import name 'detect_missed_opportunities'`

- [ ] **Step 3: Write the implementation**

Append to `core/learning/mistake_analyzer.py`:

```python
def _max_forward_move(day_df: pd.DataFrame, direction: str) -> float:
    """Largest % move available after any bar's close, in the given direction.

    "Available" = the best high (CE) / worst low (PE) among all LATER bars —
    the move a perfectly-timed entry at that close could have captured.
    """
    closes = day_df["close"].to_numpy()
    if len(closes) < 2:
        return 0.0
    best = 0.0
    if direction == "CE":
        highs = day_df["high"].to_numpy()
        run_extreme = float("-inf")
        for i in range(len(closes) - 2, -1, -1):
            run_extreme = max(run_extreme, highs[i + 1])
            best = max(best, (run_extreme - closes[i]) / closes[i] * 100)
    else:
        lows = day_df["low"].to_numpy()
        run_extreme = float("inf")
        for i in range(len(closes) - 2, -1, -1):
            run_extreme = min(run_extreme, lows[i + 1])
            best = max(best, (closes[i] - run_extreme) / closes[i] * 100)
    return best


def detect_missed_opportunities(
    df: pd.DataFrame,
    scan_date,
    strategies: list | None = None,
    promoted_names: list[str] | None = None,
    underlying: str = "NIFTY",
    session_factory=None,
) -> list[dict]:
    """Replay scan_date's bars through every strategy; log moves nobody promoted caught.

    df must be enriched (compute_all) and contain several days of history
    BEFORE scan_date — those earlier bars become the indicator warmup. At most
    one CE and one PE row is written per day.
    """
    from backtesting.tournament import generate_strategy_signals, precompute_regimes
    from core.strategies.registry import get_all_strategies, get_promoted_strategy_names

    mo_cfg = _cfg["strategy_tournament"]["missed_opportunity"]
    threshold = mo_cfg["min_move_pct_threshold"]

    day_positions = [i for i, ts in enumerate(df.index) if ts.date() == scan_date]
    if not day_positions:
        logger.warning("missed_opp_no_bars_for_date", date=str(scan_date))
        return []
    day_start = day_positions[0]
    if day_start < MIN_WARMUP_BARS:
        logger.warning("missed_opp_insufficient_warmup", warmup_bars=day_start)
        return []

    if strategies is None:
        strategies = get_all_strategies()
    if promoted_names is None:
        promoted_names = get_promoted_strategy_names(session_factory)
    promoted_set = set(promoted_names)

    regimes = precompute_regimes(df, warmup=day_start)
    day_signals: dict[str, pd.Series] = {}
    for strat in strategies:
        try:
            signals = generate_strategy_signals(strat, df, regimes, warmup=day_start)
            day_signals[strat.name] = signals.iloc[day_start:]
        except Exception as e:  # one broken strategy never kills the scan
            logger.warning("missed_opp_strategy_failed", strategy=strat.name, error=str(e))

    day_df = df.iloc[day_start:]
    found: list[dict] = []
    for direction, sig_val in (("CE", 1), ("PE", -1)):
        move_pct = _max_forward_move(day_df, direction)
        if move_pct < threshold:
            continue
        promoted_fired = any(
            (day_signals[name] == sig_val).any()
            for name in promoted_set if name in day_signals
        )
        if promoted_fired:
            continue  # a promoted strategy caught it — not a miss
        matched = sorted(
            name for name, sig in day_signals.items()
            if name not in promoted_set and (sig == sig_val).any()
        )
        if not matched:
            continue  # no strategy would have caught it — nothing to learn from
        found.append({
            "date": datetime.combine(scan_date, datetime.min.time()),
            "underlying": underlying,
            "move_pct": round(move_pct, 3),
            "direction": direction,
            "would_have_matched": matched,
            "reason": "no_promoted_strategy_signaled",
        })

    if found:
        if session_factory is None:
            from database.trade_journal import SessionLocal, init_db
            init_db()
            session_factory = SessionLocal
        db = session_factory()
        try:
            for row in found:
                db.add(MissedOpportunity(**row))
            db.commit()
        finally:
            db.close()
        logger.info("missed_opportunities_logged", count=len(found), date=str(scan_date))
    return found
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/learning/test_missed_opportunities.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add core/learning/mistake_analyzer.py tests/core/learning/test_missed_opportunities.py
git commit -m "Add daily missed-opportunity detection to mistake_analyzer"
```

---

## Task 5: Tournament — opportunity capture rate in the composite score

**Files:**
- Modify: `backtesting/tournament.py` (`compute_composite_scores` + `run_tournament`)
- Test: `tests/backtesting/test_tournament_capture.py`

**Interfaces:**
- Consumes: `MissedOpportunity` (Task 1).
- Produces:
  - `compute_opportunity_capture_rates(period_start, period_end, session_factory=None) -> dict[str, float]` — per-strategy share of the window's `MissedOpportunity` rows naming that strategy in `would_have_matched`; `{}` when no rows.
  - `compute_composite_scores(results, weights, pf_cap, capture_weight: float = 0.0)` — same as today when `capture_weight == 0.0`; otherwise blends per the Global Constraints formula.
  - `run_tournament` now populates each result's real `opportunity_capture_rate` and passes `capture_weight` from config.

- [ ] **Step 1: Write the failing test**

`tests/backtesting/test_tournament_capture.py`:

```python
"""Opportunity capture rate: computed from MissedOpportunity rows, blended into score."""
import math
from datetime import datetime

import pandas as pd
import pytz
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backtesting.tournament import (
    compute_composite_scores,
    compute_opportunity_capture_rates,
    run_tournament,
)
from core.analysis.technical import compute_all
from core.strategies.base import Strategy
from database.models import Base, MissedOpportunity, StrategyRanking

IST = pytz.timezone("Asia/Kolkata")
WEIGHTS = {"profit_factor": 0.35, "sharpe": 0.25, "win_rate": 0.20, "max_drawdown": 0.20}


class _GoodStrategy(Strategy):
    """Fires CE every 10th bar in an uptrend -> plenty of winning trades."""
    name = "good_strategy"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if len(df) % 10 == 0:
            return "CE", 70.0, ["TEST"], 1, ""
        return "NONE", 0.0, [], 0, "off-cycle"


def _mem_sessions():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _uptrend_df(rows=420):
    idx = pd.date_range("2026-01-05 09:15", periods=rows, freq="1h", tz=IST)
    closes = [22000.0 + 5 * i for i in range(rows)]
    df = pd.DataFrame({
        "open": closes, "high": [c + 8 for c in closes], "low": [c - 8 for c in closes],
        "close": closes, "volume": [1000] * rows,
    }, index=idx)
    return compute_all(df)


def _add_missed(sessions, when, matched):
    db = sessions()
    db.add(MissedOpportunity(
        date=when, underlying="NIFTY", move_pct=1.0, direction="CE",
        would_have_matched=matched, reason="no_promoted_strategy_signaled",
    ))
    db.commit()
    db.close()


def test_capture_rates_counted_per_strategy():
    sessions = _mem_sessions()
    _add_missed(sessions, datetime(2026, 6, 10), ["a", "b"])
    _add_missed(sessions, datetime(2026, 6, 11), ["a"])
    _add_missed(sessions, datetime(2026, 6, 12), ["c"])
    _add_missed(sessions, datetime(2026, 6, 13), [])
    rates = compute_opportunity_capture_rates(
        datetime(2026, 6, 1), datetime(2026, 6, 30), session_factory=sessions
    )
    assert math.isclose(rates["a"], 0.5)
    assert math.isclose(rates["b"], 0.25)
    assert math.isclose(rates["c"], 0.25)


def test_capture_rates_empty_when_no_rows():
    assert compute_opportunity_capture_rates(
        datetime(2026, 6, 1), datetime(2026, 6, 30), session_factory=_mem_sessions()
    ) == {}


def test_composite_blends_capture_dimension():
    def _results():
        return [
            {"strategy_name": "a", "status": "ranked", "win_rate": 0.5,
             "profit_factor": 1.5, "sharpe_approx": 1.0, "max_drawdown": 1000.0,
             "opportunity_capture_rate": 1.0},
            {"strategy_name": "b", "status": "ranked", "win_rate": 0.5,
             "profit_factor": 1.5, "sharpe_approx": 1.0, "max_drawdown": 1000.0,
             "opportunity_capture_rate": 0.0},
        ]

    # weight 0 -> identical metrics give identical scores (backward compatible)
    flat = compute_composite_scores(_results(), WEIGHTS, pf_cap=10.0)
    assert math.isclose(flat[0]["composite_score"], flat[1]["composite_score"])

    # weight 0.5 -> the full-capture strategy scores exactly 0.5 higher
    blended = compute_composite_scores(_results(), WEIGHTS, pf_cap=10.0, capture_weight=0.5)
    a = next(r for r in blended if r["strategy_name"] == "a")
    b = next(r for r in blended if r["strategy_name"] == "b")
    assert math.isclose(a["composite_score"] - b["composite_score"], 0.5)


def test_run_tournament_populates_capture_rate():
    sessions = _mem_sessions()
    _add_missed(sessions, datetime(2026, 6, 15), ["good_strategy"])
    run_tournament(
        strategies=[_GoodStrategy()],
        df=_uptrend_df(),
        session_factory=sessions,
        now=datetime(2026, 7, 1),
    )
    db = sessions()
    row = (db.query(StrategyRanking)
           .filter(StrategyRanking.strategy_name == "good_strategy").one())
    assert row.opportunity_capture_rate == 1.0
    db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/backtesting/test_tournament_capture.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_opportunity_capture_rates'`

- [ ] **Step 3: Add `compute_opportunity_capture_rates`**

In `backtesting/tournament.py`, add after `_minmax`:

```python
def compute_opportunity_capture_rates(
    period_start,
    period_end,
    session_factory=None,
) -> dict[str, float]:
    """Per-strategy share of the window's missed opportunities it would have caught."""
    from database.models import MissedOpportunity

    if session_factory is None:
        from database.trade_journal import SessionLocal, init_db
        init_db()
        session_factory = SessionLocal
    db = session_factory()
    try:
        rows = (
            db.query(MissedOpportunity)
            .filter(
                MissedOpportunity.date >= period_start,
                MissedOpportunity.date <= period_end,
            )
            .all()
        )
    finally:
        db.close()
    if not rows:
        return {}
    counts: dict[str, int] = {}
    for row in rows:
        for name in (row.would_have_matched or []):
            counts[name] = counts.get(name, 0) + 1
    total = len(rows)
    return {name: c / total for name, c in counts.items()}
```

- [ ] **Step 4: Extend `compute_composite_scores`**

Replace the whole `compute_composite_scores` function in `backtesting/tournament.py` with:

```python
def compute_composite_scores(
    results: list[dict],
    weights: dict,
    pf_cap: float,
    capture_weight: float = 0.0,
) -> list[dict]:
    """Min-max normalize each metric across ranked strategies, blend by weights.

    Drawdown is inverted (lower is better). When capture_weight > 0, the
    missed-opportunity capture rate is blended in on top of the 4-metric
    score: composite = (1 - cw) * four_metric_blend + cw * minmax(capture).
    Mutates and returns `results`."""
    ranked = [r for r in results if r["status"] == "ranked"]
    if not ranked:
        return results

    pfs = _minmax([min(r["profit_factor"], pf_cap) for r in ranked])
    sharpes = _minmax([r["sharpe_approx"] for r in ranked])
    win_rates = _minmax([r["win_rate"] for r in ranked])
    drawdowns = _minmax([r["max_drawdown"] for r in ranked])
    captures = _minmax([r.get("opportunity_capture_rate", 0.0) for r in ranked])

    for r, pf, sh, wr, dd, cap in zip(ranked, pfs, sharpes, win_rates, drawdowns, captures):
        base = (
            weights["profit_factor"] * pf
            + weights["sharpe"] * sh
            + weights["win_rate"] * wr
            + weights["max_drawdown"] * (1.0 - dd)
        )
        r["composite_score"] = (1.0 - capture_weight) * base + capture_weight * cap
    return results
```

- [ ] **Step 5: Wire into `run_tournament`**

In `run_tournament`, replace these three lines:

```python
    weights = t_cfg["score_weights"]
    pf_cap = t_cfg.get("profit_factor_cap", 10.0)
    compute_composite_scores(results, weights, pf_cap)
```

with:

```python
    capture_rates = compute_opportunity_capture_rates(period_start, period_end, session_factory)
    for r in results:
        r["opportunity_capture_rate"] = round(capture_rates.get(r["strategy_name"], 0.0), 3)

    weights = t_cfg["score_weights"]
    pf_cap = t_cfg.get("profit_factor_cap", 10.0)
    capture_weight = t_cfg.get("opportunity_capture_weight", 0.0)
    compute_composite_scores(results, weights, pf_cap, capture_weight=capture_weight)
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/backtesting/test_tournament_capture.py tests/backtesting/test_tournament_run.py -v`
Expected: PASS (7 passed) — the pre-existing tournament tests must pass unchanged.

Run: `python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add backtesting/tournament.py tests/backtesting/test_tournament_capture.py
git commit -m "Blend missed-opportunity capture rate into tournament composite score"
```

---

## Task 6: `select_best_signal` — promoted strategies drive live signals

**Files:**
- Modify: `core/signals/strategy_selector.py` (append function + imports)
- Modify: `scheduler/job_scheduler.py` (`job_signal_scan`, lines ~104–149)
- Test: `tests/core/signals/test_select_best_signal.py`
- Create: `tests/core/signals/__init__.py` (empty)

**Interfaces:**
- Consumes: `get_active_strategies` (Task 2), `Strategy.generate_signal` (existing), `signal_engine.generate_signal` / `signal_engine._invalid_signal` (existing), `risk_manager.compute_position_size` (existing).
- Produces (consumed by `job_signal_scan`):
  - `select_best_signal(df, regime, options_context, global_context, news_sentiment, spot_price, available_capital, strategies=None, now=None) -> TradeSignal` — `df` must already be enriched via `compute_all` (job_signal_scan already does this). `strategies` and `now` are injectable for tests.

- [ ] **Step 1: Write the failing test**

`tests/core/signals/test_select_best_signal.py`:

```python
"""select_best_signal: promoted strategies compete; fallback only when none promoted."""
from datetime import datetime

import pandas as pd
import pytz

from core.analysis.technical import compute_all
from core.signals.regime_detector import Regime, RegimeResult
from core.signals.strategy_selector import select_best_signal
from core.strategies.base import Strategy

IST = pytz.timezone("Asia/Kolkata")
# Monday 2026-07-06 11:00 IST — inside the 09:30–14:30 strategy trading window
NOW = IST.localize(datetime(2026, 7, 6, 11, 0))


class _HighScore(Strategy):
    name = "sel_high"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "CE", 80.0, ["TEST"], 1, ""


class _LowScore(Strategy):
    name = "sel_low"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "CE", 60.0, ["TEST"], 1, ""


class _TieFirst(Strategy):
    name = "sel_tie_first"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "CE", 70.0, ["TEST"], 1, ""


class _TieSecond(Strategy):
    name = "sel_tie_second"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "PE", 70.0, ["TEST"], 1, ""


class _Boom(Strategy):
    name = "sel_boom"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        raise RuntimeError("boom")


class _Never(Strategy):
    name = "sel_never"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "NONE", 0.0, [], 0, "never"


def _enriched_df(rows=250):
    idx = pd.date_range("2026-01-05 09:35", periods=rows, freq="5min", tz=IST)
    closes = [22000.0 + 5 * i for i in range(rows)]
    return compute_all(pd.DataFrame({
        "open": closes, "high": [c + 5 for c in closes], "low": [c - 5 for c in closes],
        "close": closes, "volume": [1000] * rows,
    }, index=idx))


def _regime(direction=1):
    return RegimeResult(
        regime=Regime.BULLISH_MOMENTUM if direction == 1 else Regime.UNCERTAIN,
        direction=direction, confidence=0.8,
        best_strategy="BUY_CE_ATM_PLUS_1", rationale=["test"],
        vix=15.0, position_size_factor=1.0,
    )


def _contexts():
    options_context = {"pcr": 1.0, "max_pain": None, "oi_signal": 0, "chain_df": pd.DataFrame()}
    global_context = {
        "india_vix": 15.0, "vix_high_vol_regime": False, "vix_extreme": False,
        "fii_net_cash_cr": 0.0, "dii_net_cash_cr": 0.0, "global_score": 0,
        "gift_nifty_gap_pct": None,
    }
    news_sentiment = {"score": 0.0, "summary": "test", "risk_events": [], "method": "test"}
    return options_context, global_context, news_sentiment


def _run(strategies, direction=1):
    oc, gc, ns = _contexts()
    return select_best_signal(
        _enriched_df(), _regime(direction), oc, gc, ns,
        spot_price=22000.0, available_capital=100000.0,
        strategies=strategies, now=NOW,
    )


def test_picks_highest_score_and_sizes_position():
    sig = _run([_LowScore(), _HighScore()])
    assert sig.is_valid
    assert sig.strategy == "sel_high"
    assert sig.direction == "CE"
    assert sig.order_value > 0
    assert sig.quantity >= 1
    assert sig.entry_price == 22000.0


def test_tie_broken_by_rank_order():
    # list order == tournament rank order; equal scores -> earlier wins
    sig = _run([_TieFirst(), _TieSecond()])
    assert sig.strategy == "sel_tie_first"


def test_broken_strategy_is_isolated():
    sig = _run([_Boom(), _LowScore()])
    assert sig.is_valid
    assert sig.strategy == "sel_low"


def test_no_firing_strategy_returns_invalid_without_fallback():
    sig = _run([_Never()])
    assert sig.is_valid is False
    assert sig.invalidation_reason == "No promoted strategy produced a valid signal"


def test_empty_promoted_list_falls_back_to_pipeline():
    # signal_engine path: invalid here either by session time or by
    # Regime=UNCERTAIN — both mark strategy="SKIP", proving the fallback ran.
    sig = _run([], direction=0)
    assert sig.is_valid is False
    assert sig.strategy == "SKIP"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/signals/test_select_best_signal.py -v`
Expected: FAIL — `ImportError: cannot import name 'select_best_signal'`

- [ ] **Step 3: Write the implementation**

In `core/signals/strategy_selector.py`, extend the imports at the top of the file:

```python
from datetime import date, datetime, timedelta
from typing import Optional
import pandas as pd
import pytz
import structlog
import yaml
```

and add after the `_cfg = ...` line:

```python
IST = pytz.timezone("Asia/Kolkata")
```

Then append the function at the end of the file:

```python
def select_best_signal(
    df: pd.DataFrame,
    regime: RegimeResult,
    options_context: dict,
    global_context: dict,
    news_sentiment: dict,
    spot_price: float,
    available_capital: float,
    strategies: Optional[list] = None,
    now: Optional[datetime] = None,
) -> TradeSignal:
    """Promoted-strategy signal selection (AI brain spec §4).

    df must already be enriched via compute_all(). Behavior:
    - No strategies promoted yet (before the first tournament) -> the original
      hardcoded signal_engine pipeline runs unchanged.
    - Otherwise every promoted strategy evaluates the current bar (failures
      isolated per strategy) and the highest-composite_score valid signal
      wins, ties broken by tournament rank. If none fire there is NO
      fallback — the system trades nothing this bar.
    """
    from core.execution.risk_manager import risk_manager
    from core.signals.signal_engine import generate_signal, _invalid_signal
    from core.strategies.registry import get_active_strategies

    if strategies is None:
        strategies = get_active_strategies()

    if not strategies:
        return generate_signal(
            df_5m=df,
            regime=regime,
            options_context=options_context,
            global_context=global_context,
            news_sentiment=news_sentiment,
            spot_price=spot_price,
            available_capital=available_capital,
        )

    rank_order = {s.name: i for i, s in enumerate(strategies)}  # registry returns rank order
    candidates: list[TradeSignal] = []
    for strat in strategies:
        try:
            sig = strat.generate_signal(
                df, regime, options_context, global_context, news_sentiment, now=now
            )
        except Exception as e:  # one broken strategy never blocks the scan
            logger.warning("promoted_strategy_failed", strategy=strat.name, error=str(e))
            continue
        if sig.is_valid:
            candidates.append(sig)

    if not candidates:
        return _invalid_signal(
            "No promoted strategy produced a valid signal",
            regime,
            now or datetime.now(IST),
        )

    best = min(
        candidates,
        key=lambda s: (-s.composite_score, rank_order.get(s.strategy, len(rank_order))),
    )

    # Strategy archetypes detect patterns but don't size positions — apply the
    # same score- and VIX-adjusted sizing the hardcoded pipeline uses.
    order_value = risk_manager.compute_position_size(
        best.composite_score, global_context.get("india_vix"), available_capital
    )
    best.order_value = round(order_value, 2)
    if spot_price > 0:
        best.quantity = max(1, int(order_value / spot_price))
        best.entry_price = spot_price
    logger.info(
        "promoted_signal_selected",
        strategy=best.strategy, direction=best.direction,
        score=best.composite_score, candidates=len(candidates),
    )
    return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/signals/test_select_best_signal.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Rewire `job_signal_scan`**

In `scheduler/job_scheduler.py` inside `job_signal_scan`:

Replace the import lines:

```python
        from core.signals.signal_engine import generate_signal
        from core.signals.strategy_selector import resolve_tradeable_instrument, get_next_thursday_expiry
```

with:

```python
        from core.signals.strategy_selector import (
            resolve_tradeable_instrument, get_next_thursday_expiry, select_best_signal,
        )
```

Replace the signal-generation call:

```python
        # Signal generation
        signal = generate_signal(
            df_5m=df,
            regime=regime,
            options_context=options_ctx,
            global_context=_global_context,
            news_sentiment=_news_sentiment,
            spot_price=spot_price,
            available_capital=total_capital,
        )
```

with:

```python
        # Signal generation: promoted tournament strategies when available,
        # otherwise the original hardcoded pipeline (select_best_signal decides).
        signal = select_best_signal(
            df=df,
            regime=regime,
            options_context=options_ctx,
            global_context=_global_context,
            news_sentiment=_news_sentiment,
            spot_price=spot_price,
            available_capital=total_capital,
        )
```

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass

Run: `python -c "import scheduler.job_scheduler; print('imports ok')"`
Expected: `imports ok`

- [ ] **Step 7: Commit**

```bash
git add core/signals/strategy_selector.py scheduler/job_scheduler.py tests/core/signals/__init__.py tests/core/signals/test_select_best_signal.py
git commit -m "Route live signals through promoted strategies (select_best_signal)"
```

---

## Task 7: Scheduler — weekly tournament + daily missed-opportunity jobs

**Files:**
- Modify: `scheduler/job_scheduler.py` (two new job functions + `create_scheduler` registration)
- Test: `tests/scheduler/test_scheduler_jobs.py`
- Create: `tests/scheduler/__init__.py` (empty)

**Interfaces:**
- Consumes: `run_tournament` (existing), `detect_missed_opportunities` (Task 4), config `strategy_tournament.enabled/run_day/missed_opportunity.enabled`.
- Produces: APScheduler jobs `weekly_tournament` (run_day 10:00 IST) and `missed_opportunity_scan` (Mon–Thu 15:45 IST, after the 15:10 square-off and 15:30 report).

- [ ] **Step 1: Write the failing test**

`tests/scheduler/test_scheduler_jobs.py`:

```python
"""New AI-brain scheduler jobs are registered and config-gated."""
import scheduler.job_scheduler as js


def test_tournament_jobs_registered():
    s = js.create_scheduler()
    ids = {j.id for j in s.get_jobs()}
    assert "weekly_tournament" in ids
    assert "missed_opportunity_scan" in ids


def test_weekly_tournament_respects_disabled_flag(monkeypatch):
    monkeypatch.setitem(js._cfg["strategy_tournament"], "enabled", False)
    js.job_weekly_tournament()  # early-returns; must not raise or touch network


def test_missed_opp_scan_respects_disabled_flag(monkeypatch):
    monkeypatch.setitem(
        js._cfg["strategy_tournament"]["missed_opportunity"], "enabled", False
    )
    js.job_missed_opportunity_scan()  # early-returns; must not raise or touch network
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scheduler/test_scheduler_jobs.py -v`
Expected: FAIL — `AttributeError: module 'scheduler.job_scheduler' has no attribute 'job_weekly_tournament'`

- [ ] **Step 3: Add the job functions**

In `scheduler/job_scheduler.py`, add after `job_ml_retrain_check`:

```python
def job_weekly_tournament():
    """Weekly (run_day, default Sunday) — rank all strategies, promote top N."""
    if not _cfg.get("strategy_tournament", {}).get("enabled", False):
        return
    try:
        from backtesting.tournament import run_tournament
        from notifications.telegram_bot import telegram

        results = run_tournament()
        ranked = [r for r in results if r.get("status") == "ranked"]
        promoted = [r["strategy_name"] for r in ranked if r.get("promoted")]
        msg = (
            f"🏆 <b>Weekly Strategy Tournament</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Strategies ranked: {len(ranked)}/{len(results)}\n"
            f"Promoted: {', '.join(promoted) if promoted else 'none'}"
        )
        telegram.send_alert(msg, severity="INFO")
        logger.info("weekly_tournament_complete", promoted=promoted)
    except Exception as e:
        logger.error("weekly_tournament_failed", error=str(e))


def job_missed_opportunity_scan():
    """3:45 PM Mon–Thu — log today's moves that no promoted strategy signaled."""
    mo_cfg = _cfg.get("strategy_tournament", {}).get("missed_opportunity", {})
    if not mo_cfg.get("enabled", False):
        return
    try:
        from core.analysis.technical import compute_all
        from core.learning.mistake_analyzer import detect_missed_opportunities
        from core.market_data.historical import fetch_historical_yfinance

        df = fetch_historical_yfinance("NIFTY", period="10d", interval="5m")
        if df.empty:
            logger.warning("missed_opp_scan_no_data")
            return
        rows = detect_missed_opportunities(compute_all(df), scan_date=date.today())
        logger.info("missed_opportunity_scan_complete", found=len(rows))
        if rows:
            from notifications.telegram_bot import telegram
            lines = [
                f"• {r['direction']} move {r['move_pct']:.2f}% — would-have-caught: "
                f"{', '.join(r['would_have_matched'])}"
                for r in rows
            ]
            telegram.send_alert(
                "🔍 <b>Missed Opportunities Today</b>\n" + "\n".join(lines),
                severity="WARNING",
            )
    except Exception as e:
        logger.error("missed_opportunity_scan_failed", error=str(e))
```

- [ ] **Step 4: Register the jobs**

In `create_scheduler()`, before the final `logger.info("scheduler_configured", ...)` line, add:

```python
    # Weekly strategy tournament (AI brain): run_day 10:00 IST
    t_cfg = _cfg.get("strategy_tournament", {})
    run_day = str(t_cfg.get("run_day", "sunday"))[:3].lower()
    scheduler.add_job(
        job_weekly_tournament, CronTrigger(
            hour=10, minute=0, day_of_week=run_day, timezone=IST
        ), id="weekly_tournament", replace_existing=True
    )

    # Daily missed-opportunity scan: 3:45 PM Mon–Thu, after square-off (15:10)
    # and the daily report (15:30)
    scheduler.add_job(
        job_missed_opportunity_scan, CronTrigger(
            hour=15, minute=45, day_of_week="mon-thu", timezone=IST
        ), id="missed_opportunity_scan", replace_existing=True
    )
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/scheduler/test_scheduler_jobs.py -v`
Expected: PASS (3 passed)

Run: `python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add scheduler/job_scheduler.py tests/scheduler/__init__.py tests/scheduler/test_scheduler_jobs.py
git commit -m "Schedule weekly tournament and daily missed-opportunity scan"
```

---

## Task 8: Dashboard — Strategy Leaderboard page

**Files:**
- Create: `core/learning/leaderboard.py` (testable read-model, no FastAPI imports)
- Modify: `dashboard/app.py` (two routes, append at end)
- Create: `dashboard/templates/leaderboard.html`
- Modify: `dashboard/templates/dashboard.html` (sidebar nav link, line ~15)
- Test: `tests/core/learning/test_leaderboard.py`

**Interfaces:**
- Consumes: `StrategyRanking`, `MissedOpportunity`.
- Produces (consumed by the routes):
  - `get_leaderboard_data(session_factory=None, history_runs: int = 8, missed_limit: int = 20) -> dict` with keys `rankings` (latest run, rank order, un-ranked rows last), `history` (recent runs' promoted lists, newest first), `missed_opportunities` (newest first), `generated_at`.

- [ ] **Step 1: Write the failing test**

`tests/core/learning/test_leaderboard.py`:

```python
"""Leaderboard read-model: latest rankings, promotion history, missed-opp feed."""
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.learning.leaderboard import get_leaderboard_data
from database.models import Base, MissedOpportunity, StrategyRanking


def _mem_sessions():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _seed(sessions):
    db = sessions()
    db.add_all([
        # older run
        StrategyRanking(strategy_name="alpha", period_end=datetime(2026, 6, 28),
                        promoted=True, rank=1, status="ranked", composite_score=0.9),
        # latest run
        StrategyRanking(strategy_name="beta", period_end=datetime(2026, 7, 5),
                        promoted=True, rank=1, status="ranked", composite_score=0.8),
        StrategyRanking(strategy_name="alpha", period_end=datetime(2026, 7, 5),
                        promoted=False, rank=2, status="ranked", composite_score=0.6),
        StrategyRanking(strategy_name="gamma", period_end=datetime(2026, 7, 5),
                        promoted=False, rank=None, status="errored", composite_score=0.0),
    ])
    db.add(MissedOpportunity(
        date=datetime(2026, 7, 2), underlying="NIFTY", move_pct=0.9,
        direction="PE", would_have_matched=["alpha"],
        reason="no_promoted_strategy_signaled",
    ))
    db.commit()
    db.close()


def test_leaderboard_shape_and_ordering():
    sessions = _mem_sessions()
    _seed(sessions)
    data = get_leaderboard_data(session_factory=sessions)

    # latest run only, ranked rows first in rank order, un-ranked last
    assert [r["strategy_name"] for r in data["rankings"]] == ["beta", "alpha", "gamma"]
    assert data["rankings"][0]["promoted"] is True
    assert data["rankings"][2]["status"] == "errored"

    # history: newest run first, promoted names only
    assert data["history"][0]["promoted"] == ["beta"]
    assert data["history"][1]["promoted"] == ["alpha"]

    assert len(data["missed_opportunities"]) == 1
    assert data["missed_opportunities"][0]["direction"] == "PE"
    assert data["missed_opportunities"][0]["would_have_matched"] == ["alpha"]
    assert data["generated_at"]


def test_leaderboard_empty_db():
    data = get_leaderboard_data(session_factory=_mem_sessions())
    assert data["rankings"] == []
    assert data["history"] == []
    assert data["missed_opportunities"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/learning/test_leaderboard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.learning.leaderboard'`

- [ ] **Step 3: Write the read-model**

Create `core/learning/leaderboard.py`:

```python
"""Read-model for the dashboard Strategy Leaderboard page (no FastAPI imports
so it stays unit-testable with an in-memory DB)."""
from datetime import datetime

import structlog
from sqlalchemy import func

from database.models import MissedOpportunity, StrategyRanking

logger = structlog.get_logger(__name__)


def get_leaderboard_data(
    session_factory=None,
    history_runs: int = 8,
    missed_limit: int = 20,
) -> dict:
    if session_factory is None:
        from database.trade_journal import SessionLocal, init_db
        init_db()
        session_factory = SessionLocal

    db = session_factory()
    try:
        latest_period_end = db.query(func.max(StrategyRanking.period_end)).scalar()
        rankings = []
        if latest_period_end is not None:
            rows = (
                db.query(StrategyRanking)
                .filter(StrategyRanking.period_end == latest_period_end)
                .order_by(StrategyRanking.rank.is_(None), StrategyRanking.rank)
                .all()
            )
            rankings = [{
                "strategy_name": r.strategy_name,
                "category": r.category,
                "rank": r.rank,
                "promoted": bool(r.promoted),
                "status": r.status,
                "composite_score": r.composite_score,
                "win_rate": r.win_rate,
                "profit_factor": r.profit_factor,
                "sharpe_approx": r.sharpe_approx,
                "max_drawdown": r.max_drawdown,
                "opportunity_capture_rate": r.opportunity_capture_rate,
            } for r in rows]

        period_ends = [
            p[0] for p in (
                db.query(StrategyRanking.period_end)
                .filter(StrategyRanking.period_end.isnot(None))
                .distinct()
                .order_by(StrategyRanking.period_end.desc())
                .limit(history_runs)
                .all()
            )
        ]
        history = []
        for pe in period_ends:
            promoted_rows = (
                db.query(StrategyRanking)
                .filter(
                    StrategyRanking.period_end == pe,
                    StrategyRanking.promoted.is_(True),
                )
                .order_by(StrategyRanking.rank)
                .all()
            )
            history.append({
                "period_end": pe.isoformat(),
                "promoted": [r.strategy_name for r in promoted_rows],
            })

        missed_rows = (
            db.query(MissedOpportunity)
            .order_by(MissedOpportunity.date.desc(), MissedOpportunity.id.desc())
            .limit(missed_limit)
            .all()
        )
        missed = [{
            "date": m.date.isoformat() if m.date else None,
            "underlying": m.underlying,
            "direction": m.direction,
            "move_pct": m.move_pct,
            "would_have_matched": m.would_have_matched or [],
            "reason": m.reason,
        } for m in missed_rows]

        return {
            "rankings": rankings,
            "history": history,
            "missed_opportunities": missed,
            "generated_at": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
        }
    finally:
        db.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/learning/test_leaderboard.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add the routes**

Append to `dashboard/app.py`:

```python
# ── Strategy Leaderboard (AI brain) ────────────────────────────────────────────

@app.get("/strategies", response_class=HTMLResponse)
async def strategies_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")
    from core.learning.leaderboard import get_leaderboard_data
    data = get_leaderboard_data()
    return templates.TemplateResponse(request, "leaderboard.html", {
        "user": user,
        "mode": _system_mode,
        **data,
    })


@app.get("/api/strategy_leaderboard")
async def api_strategy_leaderboard(request: Request):
    require_auth(request)
    from core.learning.leaderboard import get_leaderboard_data
    return get_leaderboard_data()
```

- [ ] **Step 6: Create the template**

Create `dashboard/templates/leaderboard.html`:

```html
{% extends "base.html" %}
{% block title %}Strategy Leaderboard{% endblock %}
{% block content %}
<div class="app">

  <!-- Sidebar -->
  <nav class="sidebar">
    <div class="sidebar-brand">
      <span>🎯</span> NiftySniper
    </div>
    <ul class="nav-links">
      <li><a href="/">📊 Dashboard</a></li>
      <li class="active"><a href="/strategies">🏆 Strategy Leaderboard</a></li>
      <li><a href="/logout">🚪 Logout</a></li>
    </ul>
    <div class="mode-badge {{ 'badge-live' if mode == 'live' else 'badge-paper' }}">
      {{ mode.upper() }} MODE
    </div>
  </nav>

  <!-- Main content -->
  <main class="main-content">

    <header class="topbar">
      <div class="topbar-left">
        <span class="status-text">🏆 Strategy Leaderboard</span>
        <span class="topbar-time">{{ generated_at }}</span>
      </div>
    </header>

    <style>
      .lb-section { margin: 24px 0; }
      .lb-section h2 { margin-bottom: 12px; font-size: 1.1rem; }
      .lb-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
      .lb-table th, .lb-table td { padding: 8px 10px; text-align: left; border-bottom: 1px solid rgba(128,128,128,0.25); }
      .lb-table th { opacity: 0.7; font-weight: 600; }
      .lb-badge { padding: 2px 8px; border-radius: 10px; font-size: 0.78rem; font-weight: 700; }
      .lb-promoted { background: #14532d; color: #86efac; }
      .lb-status { opacity: 0.7; }
      .lb-empty { opacity: 0.6; padding: 12px 0; }
      .lb-history-item { padding: 6px 0; border-bottom: 1px solid rgba(128,128,128,0.15); }
    </style>

    <section class="lb-section">
      <h2>Current Rankings</h2>
      <table class="lb-table">
        <thead>
          <tr>
            <th>#</th><th>Strategy</th><th>Category</th><th>Score</th>
            <th>Win Rate</th><th>PF</th><th>Sharpe</th><th>Max DD</th>
            <th>Capture</th><th>Status</th>
          </tr>
        </thead>
        <tbody>
          {% for r in rankings %}
          <tr>
            <td>{{ r.rank if r.rank is not none else '—' }}</td>
            <td>{{ r.strategy_name }}</td>
            <td>{{ r.category or '—' }}</td>
            <td>{{ "%.3f"|format(r.composite_score or 0) }}</td>
            <td>{{ "%.1f"|format((r.win_rate or 0) * 100) }}%</td>
            <td>{{ "%.2f"|format(r.profit_factor or 0) }}</td>
            <td>{{ "%.2f"|format(r.sharpe_approx or 0) }}</td>
            <td>₹{{ "%.0f"|format(r.max_drawdown or 0) }}</td>
            <td>{{ "%.0f"|format((r.opportunity_capture_rate or 0) * 100) }}%</td>
            <td>
              {% if r.promoted %}<span class="lb-badge lb-promoted">PROMOTED</span>
              {% else %}<span class="lb-status">{{ r.status }}</span>{% endif %}
            </td>
          </tr>
          {% else %}
          <tr><td colspan="10" class="lb-empty">No tournament has run yet — rankings appear after the first weekly run.</td></tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

    <section class="lb-section">
      <h2>Promotion History</h2>
      {% for h in history %}
      <div class="lb-history-item">
        <strong>{{ h.period_end[:10] }}</strong> —
        {{ h.promoted | join(', ') if h.promoted else 'none promoted' }}
      </div>
      {% else %}
      <div class="lb-empty">No runs yet.</div>
      {% endfor %}
    </section>

    <section class="lb-section">
      <h2>Recent Missed Opportunities</h2>
      <table class="lb-table">
        <thead>
          <tr><th>Date</th><th>Underlying</th><th>Direction</th><th>Move</th><th>Would Have Caught</th></tr>
        </thead>
        <tbody>
          {% for m in missed_opportunities %}
          <tr>
            <td>{{ m.date[:10] if m.date else '—' }}</td>
            <td>{{ m.underlying }}</td>
            <td>{{ m.direction }}</td>
            <td>{{ "%.2f"|format(m.move_pct or 0) }}%</td>
            <td>{{ m.would_have_matched | join(', ') }}</td>
          </tr>
          {% else %}
          <tr><td colspan="5" class="lb-empty">No missed opportunities logged.</td></tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

  </main>
</div>
{% endblock %}
```

- [ ] **Step 7: Add the nav link on the main dashboard**

In `dashboard/templates/dashboard.html`, in the sidebar `<ul class="nav-links">`, after the `📈 Performance` line and before the `🚪 Logout` line, add:

```html
      <li><a href="/strategies">🏆 Strategy Leaderboard</a></li>
```

- [ ] **Step 8: Run tests + import smoke**

Run: `python -m pytest tests/ -q`
Expected: all pass

Run: `python -c "import dashboard.app; print('dashboard imports ok')"`
Expected: `dashboard imports ok` (verifies the new routes/template wiring parse; requires local vault/.env which exists on this machine)

- [ ] **Step 9: Commit**

```bash
git add core/learning/leaderboard.py dashboard/app.py dashboard/templates/leaderboard.html dashboard/templates/dashboard.html tests/core/learning/test_leaderboard.py
git commit -m "Add Strategy Leaderboard dashboard page with missed-opportunity feed"
```

---

## Task 9: Full verification + push

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `python -m pytest tests/ -q`
Expected: 0 failed (~110 tests).

- [ ] **Step 2: Import smokes**

Run: `python -c "from core.learning.mistake_analyzer import detect_missed_opportunities, update_strategy_score_on_trade_close; from core.strategies.registry import get_active_strategies; from core.signals.strategy_selector import select_best_signal; import scheduler.job_scheduler; print('all imports ok')"`
Expected: `all imports ok`

(Do NOT run a real tournament or missed-opp scan here — they fetch yfinance data; the scheduler owns production runs.)

- [ ] **Step 3: Push**

```bash
git push origin main
```

---

## Self-Review Notes

- **Spec coverage:** §3 losing-trade learning → Task 3 (clamp pattern, additive to unchanged adaptive_weights); §3 missed-opportunity detection → Task 4 (daily replay, MissedOpportunity rows, "which strategies would have caught it"); §3 "extra scoring dimension" → Task 5; §4 weekly job + daily job → Task 7; §4 strategy_selector extension with fallback → Task 6; §5 leaderboard page (rankings, metrics, promoted status, promotion history, missed-opp feed) → Task 8; §6 MissedOpportunity schema → Task 1 (exact columns); §7 config → keys already exist from Plan 2, two additions in Task 1. Error-handling section: per-strategy isolation in Tasks 4 and 6; DB-down → `get_promoted_strategy_names` returns [] → hardcoded-pipeline fallback (Task 2/6). Testing section: missed-opp fixture test (Task 4), failure-isolation tests (Tasks 4, 6).
- **Known simplifications (documented in code):** "the relevant time" for a missed move is approximated at day granularity — one CE + one PE row max per day, matched if a strategy fired that direction at any bar of the day; TP1-reachability is approximated by `min_move_pct_threshold` (config, spec §7). Capture rate = share of logged missed opportunities a strategy would have caught (opportunities caught by promoted strategies produce no row, per spec's logging rule).
- **Type consistency check:** `update_strategy_score_on_trade_close(strategy_name, is_win, session_factory)` matches the performance_tracker call (Task 3 Step 5); `detect_missed_opportunities(df, scan_date, strategies, promoted_names, underlying, session_factory)` matches the Task 7 job call (`compute_all(df)`, `scan_date=date.today()`); `compute_composite_scores(..., capture_weight=0.0)` keeps both existing positional call sites in tests valid; `select_best_signal` kwargs in Task 6 Step 5 match its Task 6 Step 3 signature; `get_leaderboard_data()` keys match template variables (`rankings/history/missed_opportunities/generated_at`).
- **Placeholder scan:** none — every step has complete runnable code.

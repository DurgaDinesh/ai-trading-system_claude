# Strategy Infrastructure (AI Brain Plan 1 of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `core/strategies/` package containing a shared `Strategy` base class, ~~20~~ 20 standalone strategy archetypes (each producing a `TradeSignal`), and a registry that auto-discovers them — fully testable in isolation, with zero live-trading wiring yet.

**Architecture:** Each strategy is a small class implementing `evaluate(df, regime, options_context, global_context, news_sentiment) -> (direction, score, indicators, confluence_count, invalidation_reason)`. A shared `Strategy.generate_signal(...)` template method (session-time gating + `TradeSignal` construction) lives once in the base class so archetypes only contain their pattern-detection logic. Strategies reuse existing functions in `core/analysis/technical.py` and `core/analysis/options_analytics.py` — no new indicator math except two small missing helpers (`get_bollinger_signal`, `get_stochastic_signal`) that just read columns `compute_all()` already produces.

**Tech Stack:** Python 3.11/3.14, pandas, pytest (new dev dependency — not currently in `requirements.txt`).

## Global Constraints

- This plan does **not** touch `strategy_selector.py`, the scheduler, the dashboard, or any DB tables — that's Plans 2–4. Nothing built here changes live/paper trading behavior.
- Every strategy leaves `stop_loss=0.0, tp1=0.0, tp2=0.0, tp3=0.0, quantity=0, order_value=0.0` on its `TradeSignal` — matching the existing `signal_engine.generate_signal` pattern where those are filled in later by `strategy_selector.resolve_tradeable_instrument` once a winning signal is chosen (Plan 4 will wire this).
- Callers are responsible for calling `core.analysis.technical.compute_all(df)` **once** before passing `df` into any strategy's `generate_signal`/`evaluate` — strategies assume the df is already enriched (avoids recomputing indicators 20× per bar). Tests must do the same.
- Multi-leg spread strategies (iron condor, strangle/straddle) are out of scope — every strategy here produces a single directional CE/PE signal only, per `docs/superpowers/specs/2026-07-03-ai-brain-strategy-tournament-design.md`.
- Three archetypes from the original spec list were swapped for feasibility, confirmed against the actual codebase (no Fibonacci/support-resistance/volume-spike functions exist, and adding them would be new indicator math, which the spec excludes): **Support/Resistance Breakout → Stochastic Oscillator Reversal**, **Fibonacci Retracement Bounce → ATR Volatility Contraction Breakout**, **Volume Spike Confirmation → Regime-Aligned Pullback**. The spec doc will be amended to match (Task 0).

---

## Task 0: Amend the design spec to reflect the verified strategy list

**Files:**
- Modify: `docs/superpowers/specs/2026-07-03-ai-brain-strategy-tournament-design.md`

- [ ] **Step 1: Update the strategy list and base-class description**

In the `### 1. core/strategies/ (new package)` section, replace items 15–17 in the numbered list:

```markdown
  15. Stochastic Oscillator Reversal (%K/%D cross from oversold/overbought)
  16. ATR Volatility Contraction Breakout (squeeze-then-expand)
  17. Regime-Aligned Pullback (regime direction + RSI pullback entry)
```

Replace the `generate_signal(market_ctx: MarketContext) -> TradeSignal` bullet with:

```markdown
  - `generate_signal(df, regime, options_context, global_context, news_sentiment, now=None) -> TradeSignal`
    — same explicit params as the existing `signal_engine.generate_signal` (no
    separate `MarketContext` type), so both call sites share one shape. `df` must
    already be enriched via `compute_all()` by the caller.
```

Add a short note after the list: "(See `docs/superpowers/plans/2026-07-03-strategy-infrastructure.md` for why 15–17 were swapped — the original archetypes needed indicator math that doesn't exist in `technical.py` and was out of scope to add.)"

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-07-03-ai-brain-strategy-tournament-design.md
git commit -m "Amend AI brain spec: swap 3 strategy archetypes for feasibility, fix generate_signal signature"
```

---

## Task 1: Test infrastructure — shared fixtures and pytest setup

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/core/__init__.py`
- Create: `tests/core/strategies/__init__.py`
- Create: `tests/core/strategies/conftest.py`

**Interfaces:**
- Produces: `make_ohlcv_df(rows, start_price, trend, start_time, freq) -> pd.DataFrame` (raw, NOT enriched), `enriched_df(rows, trend, start_price, start_time) -> pd.DataFrame` (pre-enriched via `compute_all`), `make_regime_result(direction, confidence) -> RegimeResult`, `make_options_context(pcr, max_pain, oi_signal) -> dict`, `make_global_context(**overrides) -> dict`, `make_news_sentiment(score) -> dict`, plus pytest fixtures `flat_df`, `uptrend_df`, `downtrend_df`, `neutral_regime`, `bullish_regime`, `bearish_regime`, `neutral_options_context`, `neutral_global_context`, `neutral_news_sentiment` — all subsequent tasks consume these.

- [ ] **Step 1: Add pytest to requirements.txt**

Append to `requirements.txt`:

```
pytest==8.3.3
```

- [ ] **Step 2: Install it**

Run: `pip install pytest==8.3.3`
Expected: successful install (or "already satisfied").

- [ ] **Step 3: Create package init files**

`tests/__init__.py` — empty file.
`tests/core/__init__.py` — empty file.
`tests/core/strategies/__init__.py` — empty file.

- [ ] **Step 4: Write the shared fixtures module**

`tests/core/strategies/conftest.py`:

```python
"""Shared fixtures for core/strategies tests."""
from datetime import datetime
import pandas as pd
import pytest
import pytz

from core.analysis.technical import compute_all
from core.signals.regime_detector import Regime, RegimeResult

IST = pytz.timezone("Asia/Kolkata")

# EMA-200 needs real warmup to produce a meaningful stacked signal — use 250 bars
# by default so trend fixtures aren't flat/NaN on the longest configured EMA.
DEFAULT_ROWS = 250


def make_ohlcv_df(
    rows: int = DEFAULT_ROWS,
    start_price: float = 22000.0,
    trend: float = 0.0,
    start_time: str = "2026-01-05 09:35",
    freq: str = "5min",
) -> pd.DataFrame:
    """Synthetic 5-min OHLCV df with a datetime index and optional linear drift. NOT enriched."""
    idx = pd.date_range(start=start_time, periods=rows, freq=freq, tz=IST)
    closes = [start_price + trend * i for i in range(rows)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 5 for c in closes],
            "low": [c - 5 for c in closes],
            "close": closes,
            "volume": [1000] * rows,
        },
        index=idx,
    )


def enriched_df(
    rows: int = DEFAULT_ROWS,
    trend: float = 0.0,
    start_price: float = 22000.0,
    start_time: str = "2026-01-05 09:35",
) -> pd.DataFrame:
    """Synthetic df already run through compute_all(), as strategies expect."""
    return compute_all(make_ohlcv_df(rows=rows, start_price=start_price, trend=trend, start_time=start_time))


def make_regime_result(direction: int = 0, confidence: float = 0.6) -> RegimeResult:
    if direction == 1:
        regime = Regime.BULLISH_MOMENTUM
    elif direction == -1:
        regime = Regime.BEARISH_MOMENTUM
    else:
        regime = Regime.UNCERTAIN
    return RegimeResult(
        regime=regime,
        direction=direction,
        confidence=confidence,
        best_strategy="BUY_CE_ATM_PLUS_1",
        rationale=["test fixture"],
        vix=15.0,
        position_size_factor=1.0,
    )


def make_options_context(pcr=1.0, max_pain=22000.0, oi_signal=0) -> dict:
    return {"pcr": pcr, "max_pain": max_pain, "oi_signal": oi_signal, "chain_df": pd.DataFrame()}


def make_global_context(**overrides) -> dict:
    ctx = {
        "india_vix": 15.0,
        "vix_high_vol_regime": False,
        "vix_extreme": False,
        "fii_net_cash_cr": 0.0,
        "dii_net_cash_cr": 0.0,
        "global_score": 0,
        "gift_nifty_gap_pct": None,
    }
    ctx.update(overrides)
    return ctx


def make_news_sentiment(score: float = 0.0) -> dict:
    return {"score": score, "summary": "test", "risk_events": [], "method": "keyword_fallback"}


@pytest.fixture
def flat_df():
    return enriched_df(trend=0.0)


@pytest.fixture
def uptrend_df():
    return enriched_df(trend=5.0)


@pytest.fixture
def downtrend_df():
    return enriched_df(trend=-5.0)


@pytest.fixture
def neutral_regime():
    return make_regime_result(direction=0, confidence=0.3)


@pytest.fixture
def bullish_regime():
    return make_regime_result(direction=1, confidence=0.8)


@pytest.fixture
def bearish_regime():
    return make_regime_result(direction=-1, confidence=0.8)


@pytest.fixture
def neutral_options_context():
    return make_options_context()


@pytest.fixture
def neutral_global_context():
    return make_global_context()


@pytest.fixture
def neutral_news_sentiment():
    return make_news_sentiment()
```

- [ ] **Step 5: Verify fixtures import cleanly**

Run: `python -c "from tests.core.strategies.conftest import enriched_df; df = enriched_df(); print(df.columns.tolist()); print(len(df))"`
Expected: prints a column list including `ema_9, ema_21, ema_50, ema_200, rsi, macd, macd_signal, macd_hist, atr, supertrend, supertrend_dir, vwap, bb_upper, bb_mid, bb_lower, bb_width, stoch_k, stoch_d` and `250`.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/__init__.py tests/core/__init__.py tests/core/strategies/__init__.py tests/core/strategies/conftest.py
git commit -m "Add pytest and shared strategy test fixtures"
```

---

## Task 2: `core/strategies/base.py` — Strategy abstract base class

**Files:**
- Create: `core/strategies/__init__.py`
- Create: `core/strategies/base.py`
- Test: `tests/core/strategies/test_base.py`

**Interfaces:**
- Consumes: `TradeSignal` from `core.signals.signal_engine`, `OBS_END`/`NO_NEW_TRADE_AFTER` constants from `core.signals.signal_engine`, `RegimeResult` from `core.signals.regime_detector`.
- Produces: `Strategy` ABC with `name: str`, `category: str`, abstract `evaluate(df, regime, options_context, global_context, news_sentiment) -> tuple[str, float, list[str], int, str]`, and concrete `generate_signal(df, regime, options_context, global_context, news_sentiment, now=None) -> TradeSignal`. All later strategy tasks subclass this.

- [ ] **Step 1: Create the package init**

`core/strategies/__init__.py` — empty file.

- [ ] **Step 2: Write the failing test**

`tests/core/strategies/test_base.py`:

```python
"""Tests for the Strategy base class template method."""
from datetime import datetime, time
import pytz

from core.strategies.base import Strategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)

IST = pytz.timezone("Asia/Kolkata")


class _AlwaysCEStrategy(Strategy):
    name = "always_ce_test_strategy"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "CE", 77.0, ["TEST_INDICATOR"], 1, ""


class _AlwaysNoneStrategy(Strategy):
    name = "always_none_test_strategy"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "NONE", 0.0, [], 0, "test never fires"


def _mid_session_time():
    return IST.localize(datetime(2026, 1, 5, 11, 0))


def _pre_market_time():
    return IST.localize(datetime(2026, 1, 5, 9, 15))


def test_generate_signal_returns_valid_ce_signal_during_session():
    strat = _AlwaysCEStrategy()
    df = enriched_df()
    signal = strat.generate_signal(
        df, make_regime_result(), make_options_context(), make_global_context(),
        make_news_sentiment(), now=_mid_session_time(),
    )
    assert signal.is_valid is True
    assert signal.direction == "CE"
    assert signal.composite_score == 77.0
    assert signal.indicators_triggered == ["TEST_INDICATOR"]
    assert signal.strategy == "always_ce_test_strategy"
    assert signal.stop_loss == 0.0
    assert signal.tp1 == 0.0
    assert signal.quantity == 0


def test_generate_signal_invalid_when_evaluate_returns_none():
    strat = _AlwaysNoneStrategy()
    df = enriched_df()
    signal = strat.generate_signal(
        df, make_regime_result(), make_options_context(), make_global_context(),
        make_news_sentiment(), now=_mid_session_time(),
    )
    assert signal.is_valid is False
    assert signal.direction == "NONE"
    assert signal.invalidation_reason == "test never fires"


def test_generate_signal_invalid_outside_session_window():
    strat = _AlwaysCEStrategy()
    df = enriched_df()
    signal = strat.generate_signal(
        df, make_regime_result(), make_options_context(), make_global_context(),
        make_news_sentiment(), now=_pre_market_time(),
    )
    assert signal.is_valid is False
    assert "window" in signal.invalidation_reason.lower()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.strategies.base'`

- [ ] **Step 4: Write the implementation**

`core/strategies/base.py`:

```python
"""Shared base class for all strategy archetypes in core/strategies/."""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz

from core.signals.signal_engine import TradeSignal, OBS_END, NO_NEW_TRADE_AFTER
from core.signals.regime_detector import RegimeResult

IST = pytz.timezone("Asia/Kolkata")


class Strategy(ABC):
    """
    A single directional (CE/PE) signal-generation archetype.

    Subclasses implement `evaluate()` with their pattern-detection logic only.
    `generate_signal()` (not overridden) handles session-time gating and
    TradeSignal construction, matching the shape of the existing
    core.signals.signal_engine.generate_signal function.
    """

    name: str = "unnamed_strategy"
    category: str = "uncategorized"

    @abstractmethod
    def evaluate(
        self,
        df: pd.DataFrame,
        regime: RegimeResult,
        options_context: dict,
        global_context: dict,
        news_sentiment: dict,
    ) -> tuple[str, float, list[str], int, str]:
        """
        Detect this archetype's pattern on an already-enriched df (caller must
        have called core.analysis.technical.compute_all(df) first).

        Returns (direction, score, indicators_triggered, confluence_count, invalidation_reason).
        direction is "CE", "PE", or "NONE". When direction == "NONE",
        invalidation_reason must be a non-empty explanation.
        """
        raise NotImplementedError

    def generate_signal(
        self,
        df: pd.DataFrame,
        regime: RegimeResult,
        options_context: dict,
        global_context: dict,
        news_sentiment: dict,
        now: Optional[datetime] = None,
    ) -> TradeSignal:
        now = now or datetime.now(pytz.UTC)
        ist_time = now.astimezone(IST).time()

        if ist_time < OBS_END or ist_time >= NO_NEW_TRADE_AFTER:
            return self._invalid(regime, now, "Outside strategy trading window")

        direction, score, indicators, confluence, reason = self.evaluate(
            df, regime, options_context, global_context, news_sentiment
        )

        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns and not df.empty else 0.0
        entry_price = float(df["close"].iloc[-1]) if not df.empty else 0.0

        if direction == "NONE" or not direction:
            return self._invalid(regime, now, reason or "No pattern match", entry_price, atr)

        return TradeSignal(
            direction=direction,
            composite_score=round(score, 1),
            confluence_count=confluence,
            indicators_triggered=indicators,
            entry_price=entry_price,
            stop_loss=0.0,
            tp1=0.0,
            tp2=0.0,
            tp3=0.0,
            atr=atr,
            rr_ratio=0.0,
            regime=regime.regime.value,
            strategy=self.name,
            timestamp=now,
            is_valid=True,
            invalidation_reason="",
        )

    def _invalid(
        self,
        regime: RegimeResult,
        now: datetime,
        reason: str,
        entry_price: float = 0.0,
        atr: float = 0.0,
    ) -> TradeSignal:
        return TradeSignal(
            direction="NONE",
            composite_score=0.0,
            confluence_count=0,
            indicators_triggered=[],
            entry_price=entry_price,
            stop_loss=0.0,
            tp1=0.0,
            tp2=0.0,
            tp3=0.0,
            atr=atr,
            rr_ratio=0.0,
            regime=regime.regime.value,
            strategy=self.name,
            timestamp=now,
            is_valid=False,
            invalidation_reason=reason,
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_base.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add core/strategies/__init__.py core/strategies/base.py tests/core/strategies/test_base.py
git commit -m "Add Strategy base class with session-gated generate_signal template method"
```

---

## Task 3: `core/analysis/technical.py` additions — Bollinger and Stochastic signal helpers

**Files:**
- Modify: `core/analysis/technical.py`
- Test: `tests/core/analysis/test_technical_signal_helpers.py`
- Create: `tests/core/analysis/__init__.py`

**Interfaces:**
- Produces: `get_bollinger_signal(df: pd.DataFrame, squeeze_lookback: int = 20) -> int`, `get_stochastic_signal(df: pd.DataFrame) -> int` — consumed by the Bollinger Squeeze Breakout and Stochastic Reversal strategy tasks below.

- [ ] **Step 1: Create test package init**

`tests/core/analysis/__init__.py` — empty file.

- [ ] **Step 2: Write the failing test**

`tests/core/analysis/test_technical_signal_helpers.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/core/analysis/test_technical_signal_helpers.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_bollinger_signal'`

- [ ] **Step 4: Add the two functions to technical.py**

Append to `core/analysis/technical.py` (after the existing `get_supertrend_signal` function):

```python
def get_bollinger_signal(df: pd.DataFrame, squeeze_lookback: int = 20) -> int:
    """
    1 if close breaks above bb_upper following a bandwidth squeeze,
    -1 if it breaks below bb_lower following a squeeze, else 0.
    Reads columns already produced by compute_all() — no new indicator math.
    """
    required = {"close", "bb_upper", "bb_lower", "bb_width"}
    if not required.issubset(df.columns) or len(df) < squeeze_lookback + 1:
        return 0
    close = df["close"].iloc[-1]
    bb_upper = df["bb_upper"].iloc[-1]
    bb_lower = df["bb_lower"].iloc[-1]
    width_now = df["bb_width"].iloc[-1]
    width_recent_min = df["bb_width"].iloc[-(squeeze_lookback + 1):-1].min()
    was_squeezed = width_now <= width_recent_min * 1.1
    if not was_squeezed:
        return 0
    if close > bb_upper:
        return 1
    if close < bb_lower:
        return -1
    return 0


def get_stochastic_signal(df: pd.DataFrame) -> int:
    """
    1 if %K crosses above %D from oversold (<20), -1 if %K crosses below %D
    from overbought (>80), else 0.
    """
    required = {"stoch_k", "stoch_d"}
    if not required.issubset(df.columns) or len(df) < 2:
        return 0
    k, d = df["stoch_k"].iloc[-1], df["stoch_d"].iloc[-1]
    prev_k, prev_d = df["stoch_k"].iloc[-2], df["stoch_d"].iloc[-2]
    crossed_up = prev_k <= prev_d and k > d
    crossed_down = prev_k >= prev_d and k < d
    if crossed_up and k < 20:
        return 1
    if crossed_down and k > 80:
        return -1
    return 0
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/core/analysis/test_technical_signal_helpers.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add core/analysis/technical.py tests/core/analysis/__init__.py tests/core/analysis/test_technical_signal_helpers.py
git commit -m "Add get_bollinger_signal and get_stochastic_signal helpers"
```

---

## Task 4: Strategy — EMA Trend Following

**Files:**
- Create: `core/strategies/ema_trend_following.py`
- Test: `tests/core/strategies/test_ema_trend_following.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `get_ema_stack_signal`/`get_rsi_signal` from `core.analysis.technical`, `enriched_df`/`make_regime_result`/etc. fixtures (Task 1).
- Produces: `EMATrendFollowingStrategy` class, `name = "ema_trend_following"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_ema_trend_following.py`:

```python
from core.strategies.ema_trend_following import EMATrendFollowingStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_on_uptrend_breakout():
    strat = EMATrendFollowingStrategy()
    df = enriched_df(trend=5.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"
    assert score > 0
    assert "EMA_STACK" in indicators


def test_fires_pe_on_downtrend_breakout():
    strat = EMATrendFollowingStrategy()
    df = enriched_df(trend=-5.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=-1), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "PE"
    assert "EMA_STACK" in indicators


def test_no_signal_on_flat_market():
    strat = EMATrendFollowingStrategy()
    df = enriched_df(trend=0.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=0), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"
    assert reason != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_ema_trend_following.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/ema_trend_following.py`:

```python
"""Turtle-style breakout confirmed by EMA stack alignment."""
import pandas as pd

from core.analysis.technical import get_ema_stack_signal, get_rsi_signal
from core.strategies.base import Strategy


class EMATrendFollowingStrategy(Strategy):
    name = "ema_trend_following"
    category = "trend"
    BREAKOUT_LOOKBACK = 20

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if len(df) < self.BREAKOUT_LOOKBACK + 1:
            return "NONE", 0.0, [], 0, "Insufficient bars for breakout lookback"

        ema_signal = get_ema_stack_signal(df)
        if ema_signal == 0:
            return "NONE", 0.0, [], 0, "EMA stack not aligned"

        close = df["close"].iloc[-1]
        lookback = df["close"].iloc[-(self.BREAKOUT_LOOKBACK + 1):-1]
        broke_high = close > lookback.max()
        broke_low = close < lookback.min()

        if ema_signal == 1 and broke_high:
            direction = "CE"
        elif ema_signal == -1 and broke_low:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "No confirmed EMA-aligned breakout"

        indicators = ["EMA_STACK"]
        score = 65.0
        rsi_signal = get_rsi_signal(df)
        if (direction == "CE" and rsi_signal == 1) or (direction == "PE" and rsi_signal == -1):
            indicators.append("RSI")
            score += 15.0

        return direction, score, indicators, len(indicators), ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_ema_trend_following.py -v`
Expected: PASS (3 passed). If the breakout tests fail with direction "NONE", check `config/settings.yaml`'s `ema.periods` — increase `enriched_df`'s default row count in `conftest.py` if a longer EMA period was configured after this plan was written.

- [ ] **Step 5: Commit**

```bash
git add core/strategies/ema_trend_following.py tests/core/strategies/test_ema_trend_following.py
git commit -m "Add EMA Trend Following strategy"
```

---

## Task 5: Strategy — RSI Mean-Reversion

**Files:**
- Create: `core/strategies/rsi_mean_reversion.py`
- Test: `tests/core/strategies/test_rsi_mean_reversion.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `get_vwap_signal` from `core.analysis.technical`.
- Produces: `RSIMeanReversionStrategy`, `name = "rsi_mean_reversion"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_rsi_mean_reversion.py`:

```python
import pandas as pd

from core.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _with_rsi(df, rsi_values):
    df = df.copy()
    df.loc[df.index[-len(rsi_values):], "rsi"] = rsi_values
    return df


def test_fires_ce_on_oversold_bounce():
    strat = RSIMeanReversionStrategy()
    df = _with_rsi(enriched_df(trend=0.0), [25.0, 28.0])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"
    assert "RSI" in indicators


def test_fires_pe_on_overbought_reversal():
    strat = RSIMeanReversionStrategy()
    df = _with_rsi(enriched_df(trend=0.0), [75.0, 72.0])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_in_neutral_zone():
    strat = RSIMeanReversionStrategy()
    df = _with_rsi(enriched_df(trend=0.0), [48.0, 50.0])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_rsi_mean_reversion.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/rsi_mean_reversion.py`:

```python
"""Classic RSI oversold/overbought reversal, confirmed by VWAP side."""
from core.analysis.technical import get_vwap_signal
from core.strategies.base import Strategy


class RSIMeanReversionStrategy(Strategy):
    name = "rsi_mean_reversion"
    category = "mean_reversion"
    OVERSOLD = 30
    OVERBOUGHT = 70

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if "rsi" not in df.columns or df.empty:
            return "NONE", 0.0, [], 0, "RSI column missing"

        rsi = df["rsi"].iloc[-1]
        prev_rsi = df["rsi"].iloc[-2] if len(df) > 1 else rsi

        if rsi < self.OVERSOLD and rsi > prev_rsi:
            direction = "CE"
        elif rsi > self.OVERBOUGHT and rsi < prev_rsi:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "RSI not in reversal zone"

        indicators = ["RSI"]
        score = 60.0
        vwap_signal = get_vwap_signal(df)
        if (direction == "CE" and vwap_signal >= 0) or (direction == "PE" and vwap_signal <= 0):
            indicators.append("VWAP")
            score += 10.0

        return direction, score, indicators, len(indicators), ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_rsi_mean_reversion.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/rsi_mean_reversion.py tests/core/strategies/test_rsi_mean_reversion.py
git commit -m "Add RSI Mean-Reversion strategy"
```

---

## Task 6: Strategy — MACD Momentum Crossover

**Files:**
- Create: `core/strategies/macd_momentum_crossover.py`
- Test: `tests/core/strategies/test_macd_momentum_crossover.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `get_macd_signal` from `core.analysis.technical`.
- Produces: `MACDMomentumCrossoverStrategy`, `name = "macd_momentum_crossover"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_macd_momentum_crossover.py`:

```python
from core.strategies.macd_momentum_crossover import MACDMomentumCrossoverStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _with_macd(df, macd, macd_signal, macd_hist):
    df = df.copy()
    df.loc[df.index[-len(macd):], "macd"] = macd
    df.loc[df.index[-len(macd_signal):], "macd_signal"] = macd_signal
    df.loc[df.index[-len(macd_hist):], "macd_hist"] = macd_hist
    return df


def test_fires_ce_on_bullish_crossover_with_expanding_histogram():
    strat = MACDMomentumCrossoverStrategy()
    df = _with_macd(enriched_df(), [1.0, 2.0], [1.5, 1.5], [0.2, 0.5])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"
    assert "MACD" in indicators


def test_no_signal_without_crossover():
    strat = MACDMomentumCrossoverStrategy()
    df = _with_macd(enriched_df(), [1.0, 1.1], [1.5, 1.5], [-0.5, -0.4])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_macd_momentum_crossover.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/macd_momentum_crossover.py`:

```python
"""MACD signal-line crossover, with a bonus for an expanding histogram."""
from core.analysis.technical import get_macd_signal
from core.strategies.base import Strategy


class MACDMomentumCrossoverStrategy(Strategy):
    name = "macd_momentum_crossover"
    category = "trend"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        macd_signal = get_macd_signal(df)
        if macd_signal == 0:
            return "NONE", 0.0, [], 0, "No MACD crossover"

        direction = "CE" if macd_signal == 1 else "PE"
        indicators = ["MACD"]
        score = 60.0

        if len(df) > 1 and "macd_hist" in df.columns:
            hist = df["macd_hist"].iloc[-1]
            prev_hist = df["macd_hist"].iloc[-2]
            if abs(hist) > abs(prev_hist):
                indicators.append("MACD_HIST_EXPANDING")
                score += 15.0

        return direction, score, indicators, len(indicators), ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_macd_momentum_crossover.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/macd_momentum_crossover.py tests/core/strategies/test_macd_momentum_crossover.py
git commit -m "Add MACD Momentum Crossover strategy"
```

---

## Task 7: Strategy — VWAP Reversion

**Files:**
- Create: `core/strategies/vwap_reversion.py`
- Test: `tests/core/strategies/test_vwap_reversion.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2).
- Produces: `VWAPReversionStrategy`, `name = "vwap_reversion"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_vwap_reversion.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_vwap_reversion.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/vwap_reversion.py`:

```python
"""Fade large intraday deviations from VWAP back toward it."""
from core.strategies.base import Strategy


class VWAPReversionStrategy(Strategy):
    name = "vwap_reversion"
    category = "mean_reversion"
    DEVIATION_PCT = 0.003

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if "vwap" not in df.columns or df.empty:
            return "NONE", 0.0, [], 0, "VWAP column missing"

        close = df["close"].iloc[-1]
        vwap = df["vwap"].iloc[-1]
        if not vwap:
            return "NONE", 0.0, [], 0, "VWAP unavailable"

        deviation = (close - vwap) / vwap
        if deviation < -self.DEVIATION_PCT:
            direction = "CE"
        elif deviation > self.DEVIATION_PCT:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "Price within VWAP band"

        score = 55.0 + min(abs(deviation) * 1000, 25.0)
        return direction, score, ["VWAP"], 1, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_vwap_reversion.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/vwap_reversion.py tests/core/strategies/test_vwap_reversion.py
git commit -m "Add VWAP Reversion strategy"
```

---

## Task 8: Strategy — Supertrend Trend-Following

**Files:**
- Create: `core/strategies/supertrend_trend_following.py`
- Test: `tests/core/strategies/test_supertrend_trend_following.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `get_supertrend_signal`/`get_ema_stack_signal` from `core.analysis.technical`.
- Produces: `SupertrendTrendFollowingStrategy`, `name = "supertrend_trend_following"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_supertrend_trend_following.py`:

```python
from core.strategies.supertrend_trend_following import SupertrendTrendFollowingStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _with_supertrend_flip(df, direction_col_value, dir_prev):
    df = df.copy()
    df.loc[df.index[-2], "supertrend_dir"] = dir_prev
    df.loc[df.index[-1], "supertrend_dir"] = direction_col_value
    return df


def test_fires_ce_on_uptrend_with_supertrend_and_ema_agreement():
    strat = SupertrendTrendFollowingStrategy()
    df = _with_supertrend_flip(enriched_df(trend=5.0), 1, -1)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"
    assert "SUPERTREND" in indicators


def test_no_signal_without_flip():
    strat = SupertrendTrendFollowingStrategy()
    df = _with_supertrend_flip(enriched_df(trend=5.0), 1, 1)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_supertrend_trend_following.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/supertrend_trend_following.py`:

```python
"""Trade Supertrend direction flips, confirmed by EMA stack agreement."""
from core.analysis.technical import get_supertrend_signal, get_ema_stack_signal
from core.strategies.base import Strategy


class SupertrendTrendFollowingStrategy(Strategy):
    name = "supertrend_trend_following"
    category = "trend"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        st_signal = get_supertrend_signal(df)
        if st_signal == 0:
            return "NONE", 0.0, [], 0, "No Supertrend flip"

        direction = "CE" if st_signal == 1 else "PE"
        indicators = ["SUPERTREND"]
        score = 65.0

        ema_signal = get_ema_stack_signal(df)
        if (direction == "CE" and ema_signal == 1) or (direction == "PE" and ema_signal == -1):
            indicators.append("EMA_STACK")
            score += 15.0

        return direction, score, indicators, len(indicators), ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_supertrend_trend_following.py -v`
Expected: PASS (2 passed). Note: `get_supertrend_signal`'s exact flip-detection logic wasn't fully inspected in this plan's research pass — if the first test fails because the function reads a different column shape than `supertrend_dir`, open `core/analysis/technical.py`'s `get_supertrend_signal` and `_add_supertrend` to match the test fixture to its actual column contract before changing the strategy code.

- [ ] **Step 5: Commit**

```bash
git add core/strategies/supertrend_trend_following.py tests/core/strategies/test_supertrend_trend_following.py
git commit -m "Add Supertrend Trend-Following strategy"
```

---

## Task 9: Strategy — Opening Range Breakout

**Files:**
- Create: `core/strategies/opening_range_breakout.py`
- Test: `tests/core/strategies/test_opening_range_breakout.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2).
- Produces: `OpeningRangeBreakoutStrategy`, `name = "opening_range_breakout"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_opening_range_breakout.py`:

```python
import pandas as pd

from core.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from tests.core.strategies.conftest import (
    make_regime_result, make_options_context, make_global_context, make_news_sentiment,
)
from core.analysis.technical import compute_all
import pytz

IST = pytz.timezone("Asia/Kolkata")


def _session_df(breakout: str = "none") -> pd.DataFrame:
    """9:15-9:30 opening range at 22000 +/- 5, then a later bar that breaks out (or not)."""
    idx = pd.date_range("2026-01-05 09:15", periods=4, freq="5min", tz=IST).tolist()
    idx.append(pd.Timestamp("2026-01-05 10:00", tz=IST))
    closes = [22000.0, 22002.0, 21998.0, 22001.0]
    if breakout == "up":
        closes.append(22050.0)
    elif breakout == "down":
        closes.append(21950.0)
    else:
        closes.append(22001.0)
    df = pd.DataFrame({
        "open": closes, "high": [c + 3 for c in closes], "low": [c - 3 for c in closes],
        "close": closes, "volume": [1000] * 5,
    }, index=pd.DatetimeIndex(idx))
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_opening_range_breakout.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/opening_range_breakout.py`:

```python
"""Break of the first 15 minutes' high/low, in either direction."""
from datetime import time

import pandas as pd

from core.strategies.base import Strategy


class OpeningRangeBreakoutStrategy(Strategy):
    name = "opening_range_breakout"
    category = "breakout"
    RANGE_END = time(9, 30)

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return "NONE", 0.0, [], 0, "DataFrame index is not datetime"

        today = df.index[-1].date()
        range_bars = df.loc[(df.index.date == today) & (df.index.time <= self.RANGE_END)]
        if range_bars.empty:
            return "NONE", 0.0, [], 0, "Opening range not yet formed"

        orb_high = range_bars["high"].max()
        orb_low = range_bars["low"].min()
        close = df["close"].iloc[-1]

        if close > orb_high:
            direction = "CE"
        elif close < orb_low:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "Price inside opening range"

        return direction, 62.0, ["OPENING_RANGE_BREAKOUT"], 1, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_opening_range_breakout.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/opening_range_breakout.py tests/core/strategies/test_opening_range_breakout.py
git commit -m "Add Opening Range Breakout strategy"
```

---

## Task 10: Strategy — Bollinger Band Squeeze Breakout

**Files:**
- Create: `core/strategies/bollinger_squeeze_breakout.py`
- Test: `tests/core/strategies/test_bollinger_squeeze_breakout.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `get_bollinger_signal` from `core.analysis.technical` (Task 3).
- Produces: `BollingerSqueezeBreakoutStrategy`, `name = "bollinger_squeeze_breakout"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_bollinger_squeeze_breakout.py`:

```python
import pandas as pd

from core.strategies.bollinger_squeeze_breakout import BollingerSqueezeBreakoutStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _squeeze_then_breakout(df, direction: str):
    df = df.copy()
    df["bb_width"] = 4.0
    df["bb_upper"] = 22050.0
    df["bb_lower"] = 21950.0
    if direction == "up":
        df.loc[df.index[-1], "close"] = 22100.0
    elif direction == "down":
        df.loc[df.index[-1], "close"] = 21900.0
    return df


def test_fires_ce_on_upside_squeeze_breakout():
    strat = BollingerSqueezeBreakoutStrategy()
    df = _squeeze_then_breakout(enriched_df(), "up")
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_no_signal_without_squeeze_breakout():
    strat = BollingerSqueezeBreakoutStrategy()
    df = _squeeze_then_breakout(enriched_df(), "none")
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_bollinger_squeeze_breakout.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/bollinger_squeeze_breakout.py`:

```python
"""Breakout of Bollinger Bands following a volatility squeeze."""
from core.analysis.technical import get_bollinger_signal
from core.strategies.base import Strategy


class BollingerSqueezeBreakoutStrategy(Strategy):
    name = "bollinger_squeeze_breakout"
    category = "breakout"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        bb_signal = get_bollinger_signal(df)
        if bb_signal == 0:
            return "NONE", 0.0, [], 0, "No squeeze breakout"

        direction = "CE" if bb_signal == 1 else "PE"
        return direction, 68.0, ["BOLLINGER_SQUEEZE"], 1, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_bollinger_squeeze_breakout.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/bollinger_squeeze_breakout.py tests/core/strategies/test_bollinger_squeeze_breakout.py
git commit -m "Add Bollinger Band Squeeze Breakout strategy"
```

---

## Task 11: Strategy — PCR Contrarian

**Files:**
- Create: `core/strategies/pcr_contrarian.py`
- Test: `tests/core/strategies/test_pcr_contrarian.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `options_context["pcr"]`.
- Produces: `PCRContrarianStrategy`, `name = "pcr_contrarian"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_pcr_contrarian.py`:

```python
from core.strategies.pcr_contrarian import PCRContrarianStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_on_high_pcr():
    strat = PCRContrarianStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(pcr=1.8),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_fires_pe_on_low_pcr():
    strat = PCRContrarianStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(pcr=0.4),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_in_neutral_band():
    strat = PCRContrarianStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(pcr=1.0),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_pcr_contrarian.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/pcr_contrarian.py`:

```python
"""Fade extreme Put-Call Ratio readings."""
from core.strategies.base import Strategy


class PCRContrarianStrategy(Strategy):
    name = "pcr_contrarian"
    category = "flow"
    HIGH_PCR = 1.5
    LOW_PCR = 0.6

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        pcr = options_context.get("pcr")
        if pcr is None:
            return "NONE", 0.0, [], 0, "PCR unavailable"

        if pcr >= self.HIGH_PCR:
            direction = "CE"
        elif pcr <= self.LOW_PCR:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "PCR within neutral band"

        return direction, 58.0, ["PCR"], 1, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_pcr_contrarian.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/pcr_contrarian.py tests/core/strategies/test_pcr_contrarian.py
git commit -m "Add PCR Contrarian strategy"
```

---

## Task 12: Strategy — Max Pain Gravitational Pull

**Files:**
- Create: `core/strategies/max_pain_gravitational_pull.py`
- Test: `tests/core/strategies/test_max_pain_gravitational_pull.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `options_context["max_pain"]`.
- Produces: `MaxPainGravitationalPullStrategy`, `name = "max_pain_gravitational_pull"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_max_pain_gravitational_pull.py`:

```python
from core.strategies.max_pain_gravitational_pull import MaxPainGravitationalPullStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _df_with_close(close):
    df = enriched_df()
    df.loc[df.index[-1], "close"] = close
    return df


def test_fires_ce_when_price_below_max_pain():
    strat = MaxPainGravitationalPullStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        _df_with_close(21800.0), make_regime_result(), make_options_context(max_pain=22000.0),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_fires_pe_when_price_above_max_pain():
    strat = MaxPainGravitationalPullStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        _df_with_close(22200.0), make_regime_result(), make_options_context(max_pain=22000.0),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_near_max_pain():
    strat = MaxPainGravitationalPullStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        _df_with_close(22005.0), make_regime_result(), make_options_context(max_pain=22000.0),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_max_pain_gravitational_pull.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/max_pain_gravitational_pull.py`:

```python
"""Price tends to drift toward the option chain's max-pain strike near expiry."""
from core.strategies.base import Strategy


class MaxPainGravitationalPullStrategy(Strategy):
    name = "max_pain_gravitational_pull"
    category = "flow"
    MIN_DEVIATION_PCT = 0.005

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        max_pain = options_context.get("max_pain")
        if max_pain is None or df.empty:
            return "NONE", 0.0, [], 0, "Max pain unavailable"

        close = df["close"].iloc[-1]
        if not close:
            return "NONE", 0.0, [], 0, "Invalid close price"

        deviation = (close - max_pain) / close
        if deviation > self.MIN_DEVIATION_PCT:
            direction = "PE"
        elif deviation < -self.MIN_DEVIATION_PCT:
            direction = "CE"
        else:
            return "NONE", 0.0, [], 0, "Price near max pain"

        return direction, 55.0, ["MAX_PAIN"], 1, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_max_pain_gravitational_pull.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/max_pain_gravitational_pull.py tests/core/strategies/test_max_pain_gravitational_pull.py
git commit -m "Add Max Pain Gravitational Pull strategy"
```

---

## Task 13: Strategy — Gap-and-Go Momentum

**Files:**
- Create: `core/strategies/gap_and_go_momentum.py`
- Test: `tests/core/strategies/test_gap_and_go_momentum.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2).
- Produces: `GapAndGoMomentumStrategy`, `name = "gap_and_go_momentum"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_gap_and_go_momentum.py`:

```python
import pandas as pd
import pytz

from core.strategies.gap_and_go_momentum import GapAndGoMomentumStrategy
from tests.core.strategies.conftest import (
    make_regime_result, make_options_context, make_global_context, make_news_sentiment,
)
from core.analysis.technical import compute_all

IST = pytz.timezone("Asia/Kolkata")


def _two_day_df(gap: str = "none") -> pd.DataFrame:
    prior_idx = pd.date_range("2026-01-04 09:15", periods=3, freq="5min", tz=IST)
    prior_closes = [22000.0, 22005.0, 22000.0]  # prev close = 22000

    today_open = 22150.0 if gap == "up" else (21850.0 if gap == "down" else 22001.0)
    today_idx = pd.date_range("2026-01-05 09:15", periods=2, freq="5min", tz=IST)
    today_closes = [today_open, today_open + (30.0 if gap == "up" else (-30.0 if gap == "down" else 0.0))]

    idx = prior_idx.append(today_idx)
    closes = prior_closes + today_closes
    df = pd.DataFrame({
        "open": closes, "high": [c + 3 for c in closes], "low": [c - 3 for c in closes],
        "close": closes, "volume": [1000] * 5,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_gap_and_go_momentum.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/gap_and_go_momentum.py`:

```python
"""Trade continuation in the direction of a significant overnight gap."""
import pandas as pd

from core.strategies.base import Strategy


class GapAndGoMomentumStrategy(Strategy):
    name = "gap_and_go_momentum"
    category = "breakout"
    MIN_GAP_PCT = 0.003

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return "NONE", 0.0, [], 0, "DataFrame index is not datetime"

        today = df.index[-1].date()
        todays_bars = df.loc[df.index.date == today]
        prior_bars = df.loc[df.index.date < today]
        if todays_bars.empty or prior_bars.empty:
            return "NONE", 0.0, [], 0, "Insufficient prior-day data for gap calc"

        prev_close = prior_bars["close"].iloc[-1]
        today_open = todays_bars["open"].iloc[0]
        if not prev_close:
            return "NONE", 0.0, [], 0, "Invalid previous close"

        gap_pct = (today_open - prev_close) / prev_close
        close = df["close"].iloc[-1]

        if gap_pct > self.MIN_GAP_PCT and close >= today_open:
            direction = "CE"
        elif gap_pct < -self.MIN_GAP_PCT and close <= today_open:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "No sustained gap-and-go"

        return direction, 60.0, ["GAP_AND_GO"], 1, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_gap_and_go_momentum.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/gap_and_go_momentum.py tests/core/strategies/test_gap_and_go_momentum.py
git commit -m "Add Gap-and-Go Momentum strategy"
```

---

## Task 14: Strategy — VIX Spike Fade

**Files:**
- Create: `core/strategies/vix_spike_fade.py`
- Test: `tests/core/strategies/test_vix_spike_fade.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `get_ema_stack_signal` from `core.analysis.technical`, `global_context["vix_extreme"]`.
- Produces: `VIXSpikeFadeStrategy`, `name = "vix_spike_fade"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_vix_spike_fade.py`:

```python
from core.strategies.vix_spike_fade import VIXSpikeFadeStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_fading_a_downtrend_during_vix_extreme():
    strat = VIXSpikeFadeStrategy()
    df = enriched_df(trend=-5.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=-1), make_options_context(),
        make_global_context(vix_extreme=True), make_news_sentiment(),
    )
    assert direction == "CE"


def test_no_signal_when_vix_not_extreme():
    strat = VIXSpikeFadeStrategy()
    df = enriched_df(trend=-5.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=-1), make_options_context(),
        make_global_context(vix_extreme=False), make_news_sentiment(),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_vix_spike_fade.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/vix_spike_fade.py`:

```python
"""Fade the prevailing short-term trend when VIX signals panic/euphoria extremes."""
from core.analysis.technical import get_ema_stack_signal
from core.strategies.base import Strategy


class VIXSpikeFadeStrategy(Strategy):
    name = "vix_spike_fade"
    category = "volatility"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if not global_context.get("vix_extreme"):
            return "NONE", 0.0, [], 0, "VIX not at extreme"

        ema_signal = get_ema_stack_signal(df)
        if ema_signal == -1:
            direction = "CE"
        elif ema_signal == 1:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "No fade direction confirmation from EMA stack"

        return direction, 55.0, ["VIX_EXTREME", "EMA_STACK"], 2, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_vix_spike_fade.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/vix_spike_fade.py tests/core/strategies/test_vix_spike_fade.py
git commit -m "Add VIX Spike Fade strategy"
```

---

## Task 15: Strategy — FII/DII Flow Following

**Files:**
- Create: `core/strategies/fii_dii_flow_following.py`
- Test: `tests/core/strategies/test_fii_dii_flow_following.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `global_context["fii_net_cash_cr"]`.
- Produces: `FIIDIIFlowFollowingStrategy`, `name = "fii_dii_flow_following"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_fii_dii_flow_following.py`:

```python
from core.strategies.fii_dii_flow_following import FIIDIIFlowFollowingStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_on_strong_fii_buying():
    strat = FIIDIIFlowFollowingStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(fii_net_cash_cr=1200.0), make_news_sentiment(),
    )
    assert direction == "CE"


def test_fires_pe_on_strong_fii_selling():
    strat = FIIDIIFlowFollowingStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(fii_net_cash_cr=-1200.0), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_on_insignificant_flow():
    strat = FIIDIIFlowFollowingStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(fii_net_cash_cr=50.0), make_news_sentiment(),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_fii_dii_flow_following.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/fii_dii_flow_following.py`:

```python
"""Follow the direction of significant net FII cash flow."""
from core.strategies.base import Strategy


class FIIDIIFlowFollowingStrategy(Strategy):
    name = "fii_dii_flow_following"
    category = "flow"
    MIN_NET_CR = 500.0

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        fii = global_context.get("fii_net_cash_cr")
        if fii is None:
            return "NONE", 0.0, [], 0, "FII flow data unavailable"

        if fii >= self.MIN_NET_CR:
            direction = "CE"
        elif fii <= -self.MIN_NET_CR:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "FII flow below significance threshold"

        return direction, 57.0, ["FII_FLOW"], 1, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_fii_dii_flow_following.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/fii_dii_flow_following.py tests/core/strategies/test_fii_dii_flow_following.py
git commit -m "Add FII/DII Flow Following strategy"
```

---

## Task 16: Strategy — News Sentiment Momentum

**Files:**
- Create: `core/strategies/news_sentiment_momentum.py`
- Test: `tests/core/strategies/test_news_sentiment_momentum.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `news_sentiment["score"]`.
- Produces: `NewsSentimentMomentumStrategy`, `name = "news_sentiment_momentum"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_news_sentiment_momentum.py`:

```python
from core.strategies.news_sentiment_momentum import NewsSentimentMomentumStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_on_strongly_positive_sentiment():
    strat = NewsSentimentMomentumStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(score=0.7),
    )
    assert direction == "CE"


def test_fires_pe_on_strongly_negative_sentiment():
    strat = NewsSentimentMomentumStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(score=-0.7),
    )
    assert direction == "PE"


def test_no_signal_on_weak_sentiment():
    strat = NewsSentimentMomentumStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(), make_news_sentiment(score=0.1),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_news_sentiment_momentum.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/news_sentiment_momentum.py`:

```python
"""Trade in the direction of strong news/headline sentiment."""
from core.strategies.base import Strategy


class NewsSentimentMomentumStrategy(Strategy):
    name = "news_sentiment_momentum"
    category = "flow"
    MIN_SCORE = 0.4

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        score = news_sentiment.get("score")
        if score is None:
            return "NONE", 0.0, [], 0, "News sentiment unavailable"

        if score >= self.MIN_SCORE:
            direction = "CE"
        elif score <= -self.MIN_SCORE:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "News sentiment not strong enough"

        return direction, 56.0, ["NEWS_SENTIMENT"], 1, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_news_sentiment_momentum.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/news_sentiment_momentum.py tests/core/strategies/test_news_sentiment_momentum.py
git commit -m "Add News Sentiment Momentum strategy"
```

---

## Task 17: Strategy — Global Cues Gap Trading

**Files:**
- Create: `core/strategies/global_cues_gap_trading.py`
- Test: `tests/core/strategies/test_global_cues_gap_trading.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `global_context["global_score"]`.
- Produces: `GlobalCuesGapTradingStrategy`, `name = "global_cues_gap_trading"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_global_cues_gap_trading.py`:

```python
from core.strategies.global_cues_gap_trading import GlobalCuesGapTradingStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_on_strong_positive_global_score():
    strat = GlobalCuesGapTradingStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(global_score=3), make_news_sentiment(),
    )
    assert direction == "CE"


def test_fires_pe_on_strong_negative_global_score():
    strat = GlobalCuesGapTradingStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(global_score=-3), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_on_neutral_global_score():
    strat = GlobalCuesGapTradingStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(),
        make_global_context(global_score=0), make_news_sentiment(),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_global_cues_gap_trading.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/global_cues_gap_trading.py`:

```python
"""Trade in the direction of a strong global overnight cues composite score."""
from core.strategies.base import Strategy


class GlobalCuesGapTradingStrategy(Strategy):
    name = "global_cues_gap_trading"
    category = "flow"
    MIN_SCORE = 2

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        global_score = global_context.get("global_score")
        if global_score is None:
            return "NONE", 0.0, [], 0, "Global score unavailable"

        if global_score >= self.MIN_SCORE:
            direction = "CE"
        elif global_score <= -self.MIN_SCORE:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "Global score within neutral range"

        return direction, 58.0, ["GLOBAL_SCORE"], 1, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_global_cues_gap_trading.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/global_cues_gap_trading.py tests/core/strategies/test_global_cues_gap_trading.py
git commit -m "Add Global Cues Gap Trading strategy"
```

---

## Task 18: Strategy — Stochastic Oscillator Reversal

**Files:**
- Create: `core/strategies/stochastic_reversal.py`
- Test: `tests/core/strategies/test_stochastic_reversal.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `get_stochastic_signal` from `core.analysis.technical` (Task 3).
- Produces: `StochasticReversalStrategy`, `name = "stochastic_reversal"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_stochastic_reversal.py`:

```python
from core.strategies.stochastic_reversal import StochasticReversalStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _with_stoch(df, k, d):
    df = df.copy()
    df.loc[df.index[-len(k):], "stoch_k"] = k
    df.loc[df.index[-len(d):], "stoch_d"] = d
    return df


def test_fires_ce_on_bullish_stochastic_cross():
    strat = StochasticReversalStrategy()
    df = _with_stoch(enriched_df(), [15.0, 25.0], [18.0, 20.0])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_no_signal_without_cross():
    strat = StochasticReversalStrategy()
    df = _with_stoch(enriched_df(), [50.0, 51.0], [50.0, 50.5])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_stochastic_reversal.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/stochastic_reversal.py`:

```python
"""Stochastic %K/%D crossover from oversold/overbought extremes."""
from core.analysis.technical import get_stochastic_signal
from core.strategies.base import Strategy


class StochasticReversalStrategy(Strategy):
    name = "stochastic_reversal"
    category = "mean_reversion"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        stoch_signal = get_stochastic_signal(df)
        if stoch_signal == 0:
            return "NONE", 0.0, [], 0, "No stochastic reversal"

        direction = "CE" if stoch_signal == 1 else "PE"
        return direction, 57.0, ["STOCHASTIC"], 1, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_stochastic_reversal.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/stochastic_reversal.py tests/core/strategies/test_stochastic_reversal.py
git commit -m "Add Stochastic Oscillator Reversal strategy"
```

---

## Task 19: Strategy — ATR Volatility Contraction Breakout

**Files:**
- Create: `core/strategies/atr_volatility_contraction_breakout.py`
- Test: `tests/core/strategies/test_atr_volatility_contraction_breakout.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `get_ema_stack_signal` from `core.analysis.technical`.
- Produces: `ATRVolatilityContractionBreakoutStrategy`, `name = "atr_volatility_contraction_breakout"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_atr_volatility_contraction_breakout.py`:

```python
from core.strategies.atr_volatility_contraction_breakout import ATRVolatilityContractionBreakoutStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _with_atr_squeeze_then_expand(df, ema_direction: int):
    df = df.copy()
    df["atr"] = 10.0  # flat/contracted baseline for the whole lookback window
    df.loc[df.index[-1], "atr"] = 20.0  # sharp expansion on the last bar
    if ema_direction == 1:
        df.loc[df.index[-1], "close"] = df["close"].iloc[-1] + 500  # push EMA stack bullish
    return df


def test_fires_on_atr_expansion_after_contraction_with_ema_bias():
    strat = ATRVolatilityContractionBreakoutStrategy()
    df = _with_atr_squeeze_then_expand(enriched_df(trend=5.0), ema_direction=1)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"
    assert "ATR_EXPANSION" in indicators


def test_no_signal_without_prior_contraction():
    strat = ATRVolatilityContractionBreakoutStrategy()
    df = enriched_df(trend=5.0)
    df["atr"] = 15.0  # already-elevated, non-contracted ATR throughout
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_atr_volatility_contraction_breakout.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/atr_volatility_contraction_breakout.py`:

```python
"""Trade the expansion that follows a period of contracted ATR (a volatility squeeze)."""
from core.analysis.technical import get_ema_stack_signal
from core.strategies.base import Strategy


class ATRVolatilityContractionBreakoutStrategy(Strategy):
    name = "atr_volatility_contraction_breakout"
    category = "volatility"
    CONTRACTION_LOOKBACK = 14

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if "atr" not in df.columns or len(df) < self.CONTRACTION_LOOKBACK + 2:
            return "NONE", 0.0, [], 0, "Insufficient ATR history"

        window = df["atr"].iloc[-(self.CONTRACTION_LOOKBACK + 2):-2]
        if window.empty:
            return "NONE", 0.0, [], 0, "Insufficient ATR history"

        atr_floor = window.min()
        prev_atr = df["atr"].iloc[-2]
        atr_now = df["atr"].iloc[-1]

        was_contracted = prev_atr <= atr_floor * 1.1
        expanding = atr_now > prev_atr * 1.15
        if not (was_contracted and expanding):
            return "NONE", 0.0, [], 0, "No ATR contraction-then-expansion"

        ema_signal = get_ema_stack_signal(df)
        if ema_signal == 1:
            direction = "CE"
        elif ema_signal == -1:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "ATR expanding but no directional bias"

        return direction, 63.0, ["ATR_EXPANSION", "EMA_STACK"], 2, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_atr_volatility_contraction_breakout.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/atr_volatility_contraction_breakout.py tests/core/strategies/test_atr_volatility_contraction_breakout.py
git commit -m "Add ATR Volatility Contraction Breakout strategy"
```

---

## Task 20: Strategy — Regime-Aligned Pullback

**Files:**
- Create: `core/strategies/regime_aligned_pullback.py`
- Test: `tests/core/strategies/test_regime_aligned_pullback.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `get_rsi_signal` from `core.analysis.technical`, `regime.direction`/`regime.confidence`.
- Produces: `RegimeAlignedPullbackStrategy`, `name = "regime_aligned_pullback"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_regime_aligned_pullback.py`:

```python
from core.strategies.regime_aligned_pullback import RegimeAlignedPullbackStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def _with_rsi(df, rsi_values):
    df = df.copy()
    df.loc[df.index[-len(rsi_values):], "rsi"] = rsi_values
    return df


def test_fires_ce_on_bullish_regime_with_rsi_pullback():
    strat = RegimeAlignedPullbackStrategy()
    df = _with_rsi(enriched_df(), [25.0, 28.0])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1, confidence=0.8), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_no_signal_when_regime_low_confidence():
    strat = RegimeAlignedPullbackStrategy()
    df = _with_rsi(enriched_df(), [25.0, 28.0])
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1, confidence=0.2), make_options_context(),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_regime_aligned_pullback.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/regime_aligned_pullback.py`:

```python
"""Enter RSI pullbacks only when the detected regime confidently agrees with direction."""
from core.analysis.technical import get_rsi_signal
from core.strategies.base import Strategy


class RegimeAlignedPullbackStrategy(Strategy):
    name = "regime_aligned_pullback"
    category = "trend"
    MIN_CONFIDENCE = 0.5

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if regime.direction == 0 or regime.confidence < self.MIN_CONFIDENCE:
            return "NONE", 0.0, [], 0, "No confident directional regime"

        rsi = df["rsi"].iloc[-1] if "rsi" in df.columns and not df.empty else 50.0
        prev_rsi = df["rsi"].iloc[-2] if "rsi" in df.columns and len(df) > 1 else rsi

        if regime.direction == 1 and rsi < 35 and rsi > prev_rsi:
            direction = "CE"
        elif regime.direction == -1 and rsi > 65 and rsi < prev_rsi:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "No pullback entry confirmation aligned with regime"

        score = 50.0 + regime.confidence * 30.0
        return direction, score, ["REGIME_ALIGNMENT", "RSI"], 2, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_regime_aligned_pullback.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/regime_aligned_pullback.py tests/core/strategies/test_regime_aligned_pullback.py
git commit -m "Add Regime-Aligned Pullback strategy"
```

---

## Task 21: Strategy — Multi-Timeframe Confluence

**Files:**
- Create: `core/strategies/multi_timeframe_confluence.py`
- Test: `tests/core/strategies/test_multi_timeframe_confluence.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `compute_all`/`get_ema_stack_signal` from `core.analysis.technical`.
- Produces: `MultiTimeframeConfluenceStrategy`, `name = "multi_timeframe_confluence"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_multi_timeframe_confluence.py`:

```python
from core.strategies.multi_timeframe_confluence import MultiTimeframeConfluenceStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_when_both_timeframes_agree_bullish():
    strat = MultiTimeframeConfluenceStrategy()
    df = enriched_df(trend=5.0)  # strong sustained uptrend visible on both 5m and resampled 15m
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=1), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"
    assert confluence == 2


def test_no_signal_on_flat_market():
    strat = MultiTimeframeConfluenceStrategy()
    df = enriched_df(trend=0.0)
    direction, score, indicators, confluence, reason = strat.evaluate(
        df, make_regime_result(direction=0), make_options_context(), make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_multi_timeframe_confluence.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/multi_timeframe_confluence.py`:

```python
"""Require the 5m EMA stack signal to agree with a resampled 15m EMA stack signal."""
import pandas as pd

from core.analysis.technical import compute_all, get_ema_stack_signal
from core.strategies.base import Strategy


class MultiTimeframeConfluenceStrategy(Strategy):
    name = "multi_timeframe_confluence"
    category = "trend"
    HIGHER_TF = "15min"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 10:
            return "NONE", 0.0, [], 0, "Insufficient data for multi-timeframe resample"

        higher_df = df.resample(self.HIGHER_TF).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        if len(higher_df) < 5:
            return "NONE", 0.0, [], 0, "Insufficient higher-timeframe bars"

        higher_df = compute_all(higher_df)
        higher_signal = get_ema_stack_signal(higher_df)
        lower_signal = get_ema_stack_signal(df)

        if higher_signal == 0 or higher_signal != lower_signal:
            return "NONE", 0.0, [], 0, "Timeframes not in agreement"

        direction = "CE" if higher_signal == 1 else "PE"
        return direction, 66.0, ["EMA_STACK_5M", "EMA_STACK_15M"], 2, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_multi_timeframe_confluence.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/multi_timeframe_confluence.py tests/core/strategies/test_multi_timeframe_confluence.py
git commit -m "Add Multi-Timeframe Confluence strategy"
```

---

## Task 22: Strategy — End-of-Day Momentum

**Files:**
- Create: `core/strategies/end_of_day_momentum.py`
- Test: `tests/core/strategies/test_end_of_day_momentum.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `get_macd_signal`/`get_ema_stack_signal` from `core.analysis.technical`.
- Produces: `EndOfDayMomentumStrategy`, `name = "end_of_day_momentum"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_end_of_day_momentum.py`:

```python
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
    idx = pd.date_range("2026-01-05 10:00", periods=260, freq="1min", tz=IST)
    closes = [22000.0 + i * 0.5 for i in range(260)]
    df = pd.DataFrame({
        "open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
        "close": closes, "volume": [1000] * 260,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_end_of_day_momentum.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/end_of_day_momentum.py`:

```python
"""Trade last-hour trend continuation, confirmed by both MACD and EMA stack."""
from datetime import time

import pandas as pd

from core.analysis.technical import get_macd_signal, get_ema_stack_signal
from core.strategies.base import Strategy


class EndOfDayMomentumStrategy(Strategy):
    name = "end_of_day_momentum"
    category = "trend"
    WINDOW_START = time(13, 30)

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return "NONE", 0.0, [], 0, "DataFrame index is not datetime"

        current_time = df.index[-1].time()
        if current_time < self.WINDOW_START:
            return "NONE", 0.0, [], 0, "Outside end-of-day window"

        macd_signal = get_macd_signal(df)
        ema_signal = get_ema_stack_signal(df)

        if macd_signal == 1 and ema_signal == 1:
            direction = "CE"
        elif macd_signal == -1 and ema_signal == -1:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "No confirmed end-of-day continuation"

        return direction, 59.0, ["MACD", "EMA_STACK"], 2, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_end_of_day_momentum.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/end_of_day_momentum.py tests/core/strategies/test_end_of_day_momentum.py
git commit -m "Add End-of-Day Momentum strategy"
```

---

## Task 23: Strategy — Options OI Buildup Direction

**Files:**
- Create: `core/strategies/options_oi_buildup_direction.py`
- Test: `tests/core/strategies/test_options_oi_buildup_direction.py`

**Interfaces:**
- Consumes: `Strategy` base class (Task 2), `options_context["oi_signal"]`.
- Produces: `OptionsOIBuildupDirectionStrategy`, `name = "options_oi_buildup_direction"`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_options_oi_buildup_direction.py`:

```python
from core.strategies.options_oi_buildup_direction import OptionsOIBuildupDirectionStrategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)


def test_fires_ce_on_bullish_oi_buildup():
    strat = OptionsOIBuildupDirectionStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(oi_signal=1),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "CE"


def test_fires_pe_on_bearish_oi_buildup():
    strat = OptionsOIBuildupDirectionStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(oi_signal=-1),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "PE"


def test_no_signal_without_oi_buildup():
    strat = OptionsOIBuildupDirectionStrategy()
    direction, score, indicators, confluence, reason = strat.evaluate(
        enriched_df(), make_regime_result(), make_options_context(oi_signal=0),
        make_global_context(), make_news_sentiment(),
    )
    assert direction == "NONE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_options_oi_buildup_direction.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`core/strategies/options_oi_buildup_direction.py`:

```python
"""Follow the option chain's OI buildup signal (long buildup vs short covering)."""
from core.strategies.base import Strategy


class OptionsOIBuildupDirectionStrategy(Strategy):
    name = "options_oi_buildup_direction"
    category = "flow"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        oi_signal = options_context.get("oi_signal")
        if not oi_signal:
            return "NONE", 0.0, [], 0, "No OI buildup signal"

        direction = "CE" if oi_signal == 1 else "PE"
        return direction, 60.0, ["OI_BUILDUP"], 1, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_options_oi_buildup_direction.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/strategies/options_oi_buildup_direction.py tests/core/strategies/test_options_oi_buildup_direction.py
git commit -m "Add Options OI Buildup Direction strategy"
```

---

## Task 24: `core/strategies/registry.py` — auto-discovery

**Files:**
- Create: `core/strategies/registry.py`
- Test: `tests/core/strategies/test_registry.py`

**Interfaces:**
- Consumes: all 20 strategy modules from Tasks 4–23, `Strategy` base class from Task 2.
- Produces: `get_all_strategies() -> list[Strategy]`, `get_strategy_by_name(name: str) -> Optional[Strategy]`. Plan 2 (tournament) and Plan 4 (live wiring) both consume `get_all_strategies()`.

- [ ] **Step 1: Write the failing test**

`tests/core/strategies/test_registry.py`:

```python
from core.strategies.base import Strategy
from core.strategies.registry import get_all_strategies, get_strategy_by_name


def test_discovers_exactly_twenty_strategies():
    strategies = get_all_strategies()
    assert len(strategies) == 20


def test_all_discovered_objects_are_strategy_instances():
    for strat in get_all_strategies():
        assert isinstance(strat, Strategy)


def test_all_strategy_names_are_unique():
    names = [s.name for s in get_all_strategies()]
    assert len(names) == len(set(names))


def test_get_strategy_by_name_returns_correct_strategy():
    strat = get_strategy_by_name("ema_trend_following")
    assert strat is not None
    assert strat.name == "ema_trend_following"


def test_get_strategy_by_name_returns_none_for_unknown():
    assert get_strategy_by_name("does_not_exist") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/strategies/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.strategies.registry'`

- [ ] **Step 3: Write the implementation**

`core/strategies/registry.py`:

```python
"""Auto-discovers all Strategy subclasses defined in core/strategies/ modules."""
import importlib
import inspect
import pkgutil
from typing import Optional

import core.strategies as strategies_package
from core.strategies.base import Strategy

_EXCLUDED_MODULES = {"base", "registry"}


def _discover_strategy_classes() -> list[type]:
    classes = []
    for _, module_name, is_pkg in pkgutil.iter_modules(strategies_package.__path__):
        if is_pkg or module_name in _EXCLUDED_MODULES:
            continue
        module = importlib.import_module(f"core.strategies.{module_name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Strategy) and obj is not Strategy and obj.__module__ == module.__name__:
                classes.append(obj)
    return classes


def get_all_strategies() -> list[Strategy]:
    """Return one instance of every registered Strategy subclass."""
    return [cls() for cls in _discover_strategy_classes()]


def get_strategy_by_name(name: str) -> Optional[Strategy]:
    for strat in get_all_strategies():
        if strat.name == name:
            return strat
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/strategies/test_registry.py -v`
Expected: PASS (5 passed). If the count assertion fails, check for a strategy file with a class that doesn't match the module-name-equals-class-module check (e.g. a typo'd import), or a missing/extra file from Tasks 4–23.

- [ ] **Step 5: Commit**

```bash
git add core/strategies/registry.py tests/core/strategies/test_registry.py
git commit -m "Add strategy registry with auto-discovery"
```

---

## Task 25: Full suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire new test suite**

Run: `pytest tests/ -v`
Expected: all tests across `tests/core/strategies/` and `tests/core/analysis/` PASS (should be around 60+ passed, 0 failed).

- [ ] **Step 2: Confirm registry count matches file count**

Run: `python -c "from core.strategies.registry import get_all_strategies; names = sorted(s.name for s in get_all_strategies()); print(len(names)); [print(n) for n in names]"`
Expected: `20` followed by 20 unique strategy names.

- [ ] **Step 3: Push to GitHub**

```bash
git push origin main
```

---

## Self-Review Notes

- **Spec coverage:** every item in the amended strategy list (Task 0) has a corresponding task (Tasks 4–23). Base class, registry, and the two missing indicator helpers are covered (Tasks 2, 3, 24). Multi-leg out-of-scope note carried through from spec. DB tables, tournament, scheduler, dashboard are explicitly deferred to Plans 2–4, not silently dropped.
- **Placeholder scan:** no TBD/TODO; every step has complete, runnable code.
- **Type consistency:** all 20 strategies return the same 5-tuple shape from `evaluate()`; all consume the same 5-positional-arg signature; `Strategy.generate_signal` and `TradeSignal` field names match `core/signals/signal_engine.py` exactly as verified against the real file.
- **Known research gaps flagged inline for implementers:** Task 8 (Supertrend column contract not fully verified — flagged with a fallback instruction), Task 4/Task 1 (EMA-200 warmup row count — flagged with a fallback instruction). These are the only two points where the plan couldn't fully verify behavior against the real function bodies and says so explicitly, rather than guessing silently.



# AI Brain: Strategy Tournament + Mistake Learning

Status: Approved by user 2026-07-03. Ready for implementation planning.

## Why

The system currently trades one hardcoded signal pipeline (`signal_engine.py` composite
score → `strategy_selector.py`). The user wants the system to:

1. Learn from its own mistakes — not just losing trades (partially covered today by
   `core/learning/adaptive_weights.py` and `ml_scorer.py`), but also **missed
   opportunities**: profitable moves the system never signaled on at all.
2. Compete against ~20 well-documented, profitable strategy archetypes (the kind of
   approaches associated with successful traders — trend-following, mean-reversion,
   breakout, options-selling, etc.), adapted to Nifty/BankNifty weekly options, and
   automatically favor whichever are actually working.

## Scope boundary (important)

The existing execution engine (`core/execution/order_manager.py`,
`position_manager.py`) supports **one directional single-leg option position at a
time**, with partial exits across TP1/TP2/TP3 (see `order_manager.py:138` — "only the
final leg marks it CLOSED"). It does **not** support true multi-leg spreads (iron
condor, strangle/straddle selling with simultaneous multiple legs).

Therefore all 20 strategy archetypes in this phase are **directional signal-generation
methods** that each produce a single CE/PE `TradeSignal` (matching the existing
dataclass in `signal_engine.py`), not literal multi-leg spread execution. Multi-leg
strategies are explicitly out of scope and would require a separate future phase to
extend the execution engine. This keeps every strategy compatible with the existing
`order_manager`/`position_manager`/`risk_manager` without changes.

## Architecture

### 1. `core/strategies/` (new package)

- `core/strategies/base.py` — abstract `Strategy` base class:
  - `name: str`, `category: str` (trend / mean_reversion / breakout / volatility / flow)
  - `generate_signal(market_ctx: MarketContext) -> TradeSignal` — reuses the existing
    `TradeSignal` dataclass from `signal_engine.py`, with `strategy` field set to the
    archetype's name (that field already exists and is currently just informational).
  - Strategies reuse existing indicator/data functions from `core/analysis/` — they
    recombine and re-threshold existing signals (EMA, RSI, MACD, VWAP, Supertrend, OI,
    PCR, global market, news sentiment) rather than each requiring net-new data
    pipelines.
- `core/strategies/registry.py` — auto-discovers all `Strategy` subclasses in the
  package at import time, exposes `get_all_strategies()` and `get_active_strategies()`
  (active = currently promoted, see below).
- 20 concrete strategy modules, one file each:
  1. EMA Trend Following (Turtle-style breakout on EMA stack)
  2. RSI Mean-Reversion
  3. MACD Momentum Crossover
  4. VWAP Reversion
  5. Supertrend Trend-Following
  6. Opening Range Breakout (ORB)
  7. Bollinger Band Squeeze Breakout
  8. PCR Contrarian (put-call ratio extremes)
  9. Max Pain Gravitational Pull
  10. Gap-and-Go Momentum
  11. VIX Spike Fade (volatility mean reversion)
  12. FII/DII Flow Following
  13. News Sentiment Momentum
  14. Global Cues Gap Trading (SGX/GIFT Nifty correlation)
  15. Support/Resistance Breakout
  16. Fibonacci Retracement Bounce
  17. Volume Spike Confirmation
  18. Multi-Timeframe Confluence (higher-TF trend + lower-TF entry)
  19. End-of-Day Momentum (last-hour continuation, respecting existing square-off rules)
  20. Options OI Buildup Direction (long buildup / short-covering signals)

### 2. `backtesting/tournament.py` (new)

- Runs the existing `backtest_engine.py` once per registered strategy, over a
  configurable rolling lookback window (default 180 days, config-driven).
- Computes the same metrics `performance_tracker.py` already computes for the live
  escalation check (win rate, profit factor, Sharpe approx, max drawdown) — the
  metric calculation is extracted from `performance_tracker` into a shared
  `core/learning/metrics.py` helper so both the live-escalation check and the
  tournament use one implementation.
- Ranks strategies by a composite tournament score (config-weighted blend of the four
  metrics — default weights: profit_factor 0.35, sharpe 0.25, win_rate 0.20,
  max_drawdown 0.20 (inverted, lower is better)).
- Writes results to a new `StrategyRanking` table (see schema below).
- Strategies that fail during backtest (exception) or have fewer than
  `min_backtest_trades_for_ranking` (config, default 15) trades in the window are
  marked `insufficient_data` / `errored` and excluded from ranking that cycle — they
  do not crash the tournament run.

### 3. `core/learning/mistake_analyzer.py` (new)

Two responsibilities:

- **Losing-trade learning (extends existing pattern):** per-strategy win/loss outcomes
  (from both backtest and live paper trades) adjust a per-strategy score in
  `StrategyRanking`, following the same clamp-and-normalize pattern already used in
  `adaptive_weights.py`. This is in addition to — not a replacement for — the existing
  global indicator-weight learning, which continues to run unchanged.
- **Missed-opportunity detection (new):** a daily scheduled job that, after market
  close, replays the day's actual historical price bars through every registered
  strategy (via the same `MarketContext` structure used for live signals). For any
  case where:
  - the underlying moved beyond what would have hit TP1 for a plausible CE/PE entry,
    **and**
  - no *active/promoted* strategy actually produced a valid signal at the relevant
    time, but a *non-promoted* strategy would have —

  it logs a row to the new `MissedOpportunity` table, including which strategies
  would have caught it. This feeds into the next tournament cycle as an extra scoring
  dimension ("opportunity capture rate") added to the composite score.

### 4. Scheduler integration

- New weekly APScheduler job (default: Sunday, configurable) runs
  `backtesting/tournament.py`, updates `StrategyRanking`, and sets `promoted = true` on
  the top `promote_top_n` (config, default 3) strategies, `promoted = false` on the
  rest.
- New daily job (after EOD square-off) runs the missed-opportunity scan.
- `strategy_selector.py` is extended: instead of always using the single hardcoded
  pipeline, it asks `registry.get_active_strategies()` for the promoted strategies,
  runs each against the current bar, and selects the highest-`composite_score` valid
  signal among them (ties broken by tournament rank). If zero strategies are promoted
  yet (e.g. first run before any tournament has completed), it falls back to the
  existing single hardcoded pipeline unchanged — so the system is never left without a
  working strategy.

### 5. Dashboard

- New "Strategy Leaderboard" page: current rankings, per-strategy metrics, promoted
  status, promotion/demotion history, and a feed of recent missed opportunities.

### 6. Database additions (`database/models.py`)

- `StrategyRanking`: `id, strategy_name, category, period_start, period_end,
  win_rate, profit_factor, sharpe_approx, max_drawdown, opportunity_capture_rate,
  composite_score, rank, promoted, status (ranked|insufficient_data|errored),
  created_at`
- `MissedOpportunity`: `id, date, underlying, move_pct, direction, would_have_matched
  (JSON list of strategy names), reason (e.g. "no_promoted_strategy_signaled"),
  created_at`

Both follow the existing SQLAlchemy model conventions in `database/models.py`.
Schema changes are applied via the same mechanism currently used for existing tables
(no separate migration framework exists yet, so this stays consistent with current
practice).

### 7. Config additions (`config/settings.yaml`)

```yaml
strategy_tournament:
  enabled: true
  run_day: "sunday"
  backtest_lookback_days: 180
  min_backtest_trades_for_ranking: 15
  promote_top_n: 3
  score_weights:
    profit_factor: 0.35
    sharpe: 0.25
    win_rate: 0.20
    max_drawdown: 0.20
  missed_opportunity:
    enabled: true
    min_move_pct_threshold: 0.5
```

## Data flow summary

```
Weekly: tournament.py runs all 20 strategies through backtest_engine
      -> metrics.py computes win_rate/profit_factor/sharpe/drawdown per strategy
      -> composite score computed (incl. opportunity_capture_rate from prior week)
      -> StrategyRanking rows written, top N marked promoted

Live/paper (intraday): strategy_selector asks registry for promoted strategies
      -> each generates a TradeSignal for the current bar
      -> highest-scoring valid signal is traded (existing order_manager/risk_manager
         unchanged)
      -> on trade close: existing global adaptive_weights update (unchanged) AND
         new per-strategy score update in mistake_analyzer

Daily (after square-off): mistake_analyzer replays the day's bars through all 20
      strategies (not just promoted) -> logs MissedOpportunity rows -> feeds into
      next week's tournament composite score
```

## Error handling

- Strategy-level isolation: any exception inside a single strategy's
  `generate_signal()` (live, backtest, or missed-opportunity replay) is caught,
  logged with the strategy name, and that strategy is skipped for that cycle/bar. It
  never aborts the tournament run or a live scan.
- Missing/insufficient historical data for a strategy's backtest → marked
  `insufficient_data`, excluded from ranking, not promoted, no crash.
- If the tournament job itself fails entirely (e.g. DB unavailable), the previous
  week's `promoted` flags remain in effect — `strategy_selector` always has a last-known
  promoted set (or falls back to the original hardcoded pipeline if none exists yet).

## Testing

- Per-strategy unit tests: feed each of the 20 strategies a synthetic price series
  engineered to trigger it, assert it produces the expected directional signal; feed a
  flat/neutral series, assert `is_valid == False`.
- `tournament.py` integration test: run against a small fixture historical dataset,
  verify `StrategyRanking` rows are written with sane values and exactly
  `promote_top_n` strategies are marked promoted.
- `mistake_analyzer` missed-opportunity test: construct a fixture day with a known
  large move and zero active signals, assert a `MissedOpportunity` row is created
  naming the strategies that would have caught it.
- Failure-isolation test: one strategy engineered to raise an exception, assert the
  tournament still completes and ranks the other 19.

## Out of scope (this phase)

- Multi-leg spread strategies (iron condor, strangle/straddle selling) — requires
  execution-engine changes, deferred to a future phase.
- Per-strategy live-escalation criteria — the existing paper→live escalation check in
  `performance_tracker.py` remains global (all promoted strategies combined), not
  per-strategy. Per-strategy live promotion could be a future enhancement.
- Real-money capital allocation across strategies (Approach C from the design
  discussion — full parallel live competition with virtual capital allocation) — not
  built now; this phase only promotes the top strategies into the existing single
  paper-trading pipeline.

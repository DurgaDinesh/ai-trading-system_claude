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

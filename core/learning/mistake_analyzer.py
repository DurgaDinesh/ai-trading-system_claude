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

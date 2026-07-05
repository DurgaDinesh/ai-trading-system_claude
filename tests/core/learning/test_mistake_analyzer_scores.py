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

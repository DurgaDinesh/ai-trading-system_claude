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

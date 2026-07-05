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

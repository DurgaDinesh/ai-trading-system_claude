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

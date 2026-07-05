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

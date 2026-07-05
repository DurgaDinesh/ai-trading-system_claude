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

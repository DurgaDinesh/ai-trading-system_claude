"""StrategyRanking table: schema round-trip on an in-memory SQLite DB."""
from datetime import datetime

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, StrategyRanking


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_strategy_ranking_round_trip():
    db = _session()
    row = StrategyRanking(
        strategy_name="ema_trend_following", category="trend",
        period_start=datetime(2026, 1, 1), period_end=datetime(2026, 6, 30),
        win_rate=0.61, profit_factor=1.8, sharpe_approx=1.2, max_drawdown=12000.0,
        opportunity_capture_rate=0.0, composite_score=0.74, rank=1,
        promoted=True, status="ranked",
    )
    db.add(row)
    db.commit()
    got = db.query(StrategyRanking).one()
    assert got.strategy_name == "ema_trend_following"
    assert got.promoted is True
    assert got.status == "ranked"
    assert got.created_at is not None


def test_tournament_config_block_present():
    cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))
    st = cfg["strategy_tournament"]
    assert st["enabled"] is True
    assert st["backtest_lookback_days"] == 180
    assert st["min_backtest_trades_for_ranking"] == 15
    assert st["promote_top_n"] == 3
    w = st["score_weights"]
    assert abs(w["profit_factor"] + w["sharpe"] + w["win_rate"] + w["max_drawdown"] - 1.0) < 1e-9

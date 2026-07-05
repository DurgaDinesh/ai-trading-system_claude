"""Read-model for the dashboard Strategy Leaderboard page (no FastAPI imports
so it stays unit-testable with an in-memory DB)."""
from datetime import datetime

import structlog
from sqlalchemy import func

from database.models import MissedOpportunity, StrategyRanking

logger = structlog.get_logger(__name__)


def get_leaderboard_data(
    session_factory=None,
    history_runs: int = 8,
    missed_limit: int = 20,
) -> dict:
    if session_factory is None:
        from database.trade_journal import SessionLocal, init_db
        init_db()
        session_factory = SessionLocal

    db = session_factory()
    try:
        latest_period_end = db.query(func.max(StrategyRanking.period_end)).scalar()
        rankings = []
        if latest_period_end is not None:
            rows = (
                db.query(StrategyRanking)
                .filter(StrategyRanking.period_end == latest_period_end)
                .order_by(StrategyRanking.rank.is_(None), StrategyRanking.rank)
                .all()
            )
            rankings = [{
                "strategy_name": r.strategy_name,
                "category": r.category,
                "rank": r.rank,
                "promoted": bool(r.promoted),
                "status": r.status,
                "composite_score": r.composite_score,
                "win_rate": r.win_rate,
                "profit_factor": r.profit_factor,
                "sharpe_approx": r.sharpe_approx,
                "max_drawdown": r.max_drawdown,
                "opportunity_capture_rate": r.opportunity_capture_rate,
            } for r in rows]

        period_ends = [
            p[0] for p in (
                db.query(StrategyRanking.period_end)
                .filter(StrategyRanking.period_end.isnot(None))
                .distinct()
                .order_by(StrategyRanking.period_end.desc())
                .limit(history_runs)
                .all()
            )
        ]
        history = []
        for pe in period_ends:
            promoted_rows = (
                db.query(StrategyRanking)
                .filter(
                    StrategyRanking.period_end == pe,
                    StrategyRanking.promoted.is_(True),
                )
                .order_by(StrategyRanking.rank)
                .all()
            )
            history.append({
                "period_end": pe.isoformat(),
                "promoted": [r.strategy_name for r in promoted_rows],
            })

        missed_rows = (
            db.query(MissedOpportunity)
            .order_by(MissedOpportunity.date.desc(), MissedOpportunity.id.desc())
            .limit(missed_limit)
            .all()
        )
        missed = [{
            "date": m.date.isoformat() if m.date else None,
            "underlying": m.underlying,
            "direction": m.direction,
            "move_pct": m.move_pct,
            "would_have_matched": m.would_have_matched or [],
            "reason": m.reason,
        } for m in missed_rows]

        return {
            "rankings": rankings,
            "history": history,
            "missed_opportunities": missed,
            "generated_at": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
        }
    finally:
        db.close()

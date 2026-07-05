"""Auto-discovers all Strategy subclasses defined in core/strategies/ modules."""
import importlib
import inspect
import pkgutil
from typing import Optional

import core.strategies as strategies_package
from core.strategies.base import Strategy

_EXCLUDED_MODULES = {"base", "registry"}


def _discover_strategy_classes() -> list[type]:
    classes = []
    for _, module_name, is_pkg in pkgutil.iter_modules(strategies_package.__path__):
        if is_pkg or module_name in _EXCLUDED_MODULES:
            continue
        module = importlib.import_module(f"core.strategies.{module_name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Strategy) and obj is not Strategy and obj.__module__ == module.__name__:
                classes.append(obj)
    return classes


def get_all_strategies() -> list[Strategy]:
    """Return one instance of every registered Strategy subclass."""
    return [cls() for cls in _discover_strategy_classes()]


def get_strategy_by_name(name: str) -> Optional[Strategy]:
    for strat in get_all_strategies():
        if strat.name == name:
            return strat
    return None


def get_promoted_strategy_names(session_factory=None) -> list[str]:
    """Promoted strategy names from the latest tournament run, ordered by rank.

    Returns [] if no tournament has run yet or the DB is unavailable — callers
    treat an empty list as "fall back to the hardcoded pipeline" (spec §4).
    """
    try:
        from sqlalchemy import func

        from database.models import StrategyRanking

        if session_factory is None:
            from database.trade_journal import SessionLocal
            session_factory = SessionLocal
        db = session_factory()
        try:
            latest_period_end = db.query(func.max(StrategyRanking.period_end)).scalar()
            if latest_period_end is None:
                return []
            rows = (
                db.query(StrategyRanking)
                .filter(
                    StrategyRanking.period_end == latest_period_end,
                    StrategyRanking.promoted.is_(True),
                )
                .order_by(StrategyRanking.rank)
                .all()
            )
            return [r.strategy_name for r in rows]
        finally:
            db.close()
    except Exception:
        return []


def get_active_strategies(session_factory=None) -> list[Strategy]:
    """Instances of the currently promoted strategies, in tournament-rank order."""
    names = get_promoted_strategy_names(session_factory)
    by_name = {s.name: s for s in get_all_strategies()}
    return [by_name[n] for n in names if n in by_name]

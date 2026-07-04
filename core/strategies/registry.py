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

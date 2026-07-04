from core.strategies.base import Strategy
from core.strategies.registry import get_all_strategies, get_strategy_by_name


def test_discovers_exactly_twenty_strategies():
    strategies = get_all_strategies()
    assert len(strategies) == 20


def test_all_discovered_objects_are_strategy_instances():
    for strat in get_all_strategies():
        assert isinstance(strat, Strategy)


def test_all_strategy_names_are_unique():
    names = [s.name for s in get_all_strategies()]
    assert len(names) == len(set(names))


def test_get_strategy_by_name_returns_correct_strategy():
    strat = get_strategy_by_name("ema_trend_following")
    assert strat is not None
    assert strat.name == "ema_trend_following"


def test_get_strategy_by_name_returns_none_for_unknown():
    assert get_strategy_by_name("does_not_exist") is None

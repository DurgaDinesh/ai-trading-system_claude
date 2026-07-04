"""Tests for the Strategy base class template method."""
from datetime import datetime, time
import pytz

from core.strategies.base import Strategy
from tests.core.strategies.conftest import (
    enriched_df, make_regime_result, make_options_context,
    make_global_context, make_news_sentiment,
)

IST = pytz.timezone("Asia/Kolkata")


class _AlwaysCEStrategy(Strategy):
    name = "always_ce_test_strategy"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "CE", 77.0, ["TEST_INDICATOR"], 1, ""


class _AlwaysNoneStrategy(Strategy):
    name = "always_none_test_strategy"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "NONE", 0.0, [], 0, "test never fires"


def _mid_session_time():
    return IST.localize(datetime(2026, 1, 5, 11, 0))


def _pre_market_time():
    return IST.localize(datetime(2026, 1, 5, 9, 15))


def test_generate_signal_returns_valid_ce_signal_during_session():
    strat = _AlwaysCEStrategy()
    df = enriched_df()
    signal = strat.generate_signal(
        df, make_regime_result(), make_options_context(), make_global_context(),
        make_news_sentiment(), now=_mid_session_time(),
    )
    assert signal.is_valid is True
    assert signal.direction == "CE"
    assert signal.composite_score == 77.0
    assert signal.indicators_triggered == ["TEST_INDICATOR"]
    assert signal.strategy == "always_ce_test_strategy"
    assert signal.stop_loss == 0.0
    assert signal.tp1 == 0.0
    assert signal.quantity == 0


def test_generate_signal_invalid_when_evaluate_returns_none():
    strat = _AlwaysNoneStrategy()
    df = enriched_df()
    signal = strat.generate_signal(
        df, make_regime_result(), make_options_context(), make_global_context(),
        make_news_sentiment(), now=_mid_session_time(),
    )
    assert signal.is_valid is False
    assert signal.direction == "NONE"
    assert signal.invalidation_reason == "test never fires"


def test_generate_signal_invalid_outside_session_window():
    strat = _AlwaysCEStrategy()
    df = enriched_df()
    signal = strat.generate_signal(
        df, make_regime_result(), make_options_context(), make_global_context(),
        make_news_sentiment(), now=_pre_market_time(),
    )
    assert signal.is_valid is False
    assert "window" in signal.invalidation_reason.lower()

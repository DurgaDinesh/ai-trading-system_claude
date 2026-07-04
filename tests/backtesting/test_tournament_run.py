"""End-to-end tournament: ranking, statuses, failure isolation, promotion."""
import pandas as pd
import pytz
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backtesting.tournament import compute_composite_scores, run_tournament
from core.analysis.technical import compute_all
from core.strategies.base import Strategy
from database.models import Base, StrategyRanking

IST = pytz.timezone("Asia/Kolkata")


class _GoodStrategy(Strategy):
    """Fires CE every 4th bar in an uptrend -> plenty of winning trades.

    generate_strategy_signals() replays with a fixed-size rolling window
    (WARMUP_BARS + 1 rows), so len(df) inside evaluate() is constant and
    can't be used to detect "every Nth bar". Key off the window's last
    close instead, which increments by 5 per bar in _uptrend_df() and is
    therefore an exact, deterministic proxy for the bar index. Firing every
    4th bar (rather than every 10th) comfortably clears
    min_backtest_trades_for_ranking after accounting for signals dropped
    while a trade is already open.
    """
    name = "good_strategy"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if round(df["close"].iloc[-1]) % 20 == 0:
            return "CE", 70.0, ["TEST"], 1, ""
        return "NONE", 0.0, [], 0, "off-cycle"


# NOTE (fixture fix): rows=420 leaves only 220 post-warmup bars, and TP2
# (4x ATR) at this synthetic uptrend's velocity takes ~14 bars per trade to
# close -- durations, not signal frequency, cap throughput at ~15 trades
# regardless of how often the strategy fires. That sits exactly on the
# min_backtest_trades_for_ranking threshold. Using more rows gives enough
# post-warmup bars for "every 4th bar" firing to close comfortably more
# than 15 trades.


class _QuietStrategy(Strategy):
    """Never fires -> insufficient_data."""
    name = "quiet_strategy"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        return "NONE", 0.0, [], 0, "never"


class _BrokenStrategy(Strategy):
    """Raises -> errored, must not kill the run."""
    name = "broken_strategy"
    category = "test"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        raise RuntimeError("boom")


def _uptrend_df(rows=600):
    idx = pd.date_range("2026-01-05 09:15", periods=rows, freq="1h", tz=IST)
    closes = [22000.0 + 5 * i for i in range(rows)]
    df = pd.DataFrame({
        "open": closes, "high": [c + 8 for c in closes], "low": [c - 8 for c in closes],
        "close": closes, "volume": [1000] * rows,
    }, index=idx)
    return compute_all(df)


def _mem_sessions():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_run_tournament_ranks_and_isolates_failures():
    sessions = _mem_sessions()
    results = run_tournament(
        strategies=[_GoodStrategy(), _QuietStrategy(), _BrokenStrategy()],
        df=_uptrend_df(),
        session_factory=sessions,
    )
    by_name = {r["strategy_name"]: r for r in results}
    assert by_name["broken_strategy"]["status"] == "errored"
    assert by_name["quiet_strategy"]["status"] == "insufficient_data"
    assert by_name["good_strategy"]["status"] == "ranked"
    assert by_name["good_strategy"]["rank"] == 1
    assert by_name["good_strategy"]["promoted"] is True
    # non-ranked strategies are never promoted
    assert by_name["quiet_strategy"]["promoted"] is False
    assert by_name["broken_strategy"]["promoted"] is False

    db = sessions()
    rows = db.query(StrategyRanking).all()
    assert len(rows) == 3
    promoted = [r for r in rows if r.promoted]
    assert len(promoted) == 1  # only 1 ranked strategy, promote_top_n=3 caps at available


def test_compute_composite_scores_weighting_and_inversion():
    results = [
        {"strategy_name": "a", "status": "ranked", "win_rate": 0.6,
         "profit_factor": 2.0, "sharpe_approx": 1.5, "max_drawdown": 1000.0},
        {"strategy_name": "b", "status": "ranked", "win_rate": 0.4,
         "profit_factor": 1.0, "sharpe_approx": 0.5, "max_drawdown": 5000.0},
    ]
    weights = {"profit_factor": 0.35, "sharpe": 0.25, "win_rate": 0.20, "max_drawdown": 0.20}
    scored = compute_composite_scores(results, weights, pf_cap=10.0)
    a = next(r for r in scored if r["strategy_name"] == "a")
    b = next(r for r in scored if r["strategy_name"] == "b")
    # a is better on every dimension (incl. lower drawdown) -> normalized 1.0 vs 0.0
    assert abs(a["composite_score"] - 1.0) < 1e-9
    assert abs(b["composite_score"] - 0.0) < 1e-9


def test_compute_composite_scores_caps_infinite_profit_factor():
    results = [
        {"strategy_name": "a", "status": "ranked", "win_rate": 0.5,
         "profit_factor": float("inf"), "sharpe_approx": 1.0, "max_drawdown": 100.0},
        {"strategy_name": "b", "status": "ranked", "win_rate": 0.5,
         "profit_factor": 1.0, "sharpe_approx": 1.0, "max_drawdown": 100.0},
    ]
    weights = {"profit_factor": 1.0, "sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0}
    scored = compute_composite_scores(results, weights, pf_cap=10.0)
    a = next(r for r in scored if r["strategy_name"] == "a")
    assert a["composite_score"] == 1.0  # capped, not inf/nan

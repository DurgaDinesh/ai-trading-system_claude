"""
Walk-forward optimization.
Splits the historical period into rolling train/test windows
to avoid in-sample overfitting and validate out-of-sample performance.
"""

from datetime import date, timedelta
from typing import Optional
import structlog
import yaml

from backtesting.backtest_engine import run_backtest

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))


def run_walk_forward(
    symbol: str = "NIFTY",
    start: Optional[date] = None,
    end: Optional[date] = None,
    train_days: int = None,
    test_days: int = None,
    interval: str = "1h",
) -> dict:
    """
    Walk-forward validation.

    Each window: train on N days → test on M days → slide forward M days.
    Reports aggregated out-of-sample performance.
    """
    bt_cfg = _cfg["backtesting"]
    start = start or date.fromisoformat(bt_cfg["default_start_date"])
    end = end or date.fromisoformat(bt_cfg["default_end_date"])
    train_days = train_days or bt_cfg["walk_forward_window_days"]
    test_days = test_days or bt_cfg["walk_forward_test_days"]

    windows = []
    current = start

    logger.info(
        "walk_forward_starting",
        symbol=symbol, start=str(start), end=str(end),
        train_days=train_days, test_days=test_days,
    )

    fold_num = 0
    while current + timedelta(days=train_days + test_days) <= end:
        train_end = current + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        fold_num += 1

        logger.info(
            "walk_forward_fold",
            fold=fold_num,
            train=f"{current}→{train_end}",
            test=f"{train_end}→{test_end}",
        )

        # Train fold (in-sample)
        train_result = run_backtest(
            symbol=symbol, start=current, end=train_end, interval=interval
        )

        # Test fold (out-of-sample — this is what matters)
        test_result = run_backtest(
            symbol=symbol, start=train_end, end=test_end, interval=interval
        )

        windows.append({
            "fold": fold_num,
            "train_start": str(current),
            "train_end": str(train_end),
            "test_start": str(train_end),
            "test_end": str(test_end),
            "in_sample": {
                "total_trades": train_result.get("total_trades", 0),
                "win_rate": train_result.get("win_rate", 0),
                "profit_factor": train_result.get("profit_factor", 0),
                "sharpe": train_result.get("sharpe_ratio", 0),
                "net_pnl": train_result.get("total_net_pnl", 0),
            },
            "out_of_sample": {
                "total_trades": test_result.get("total_trades", 0),
                "win_rate": test_result.get("win_rate", 0),
                "profit_factor": test_result.get("profit_factor", 0),
                "sharpe": test_result.get("sharpe_ratio", 0),
                "net_pnl": test_result.get("total_net_pnl", 0),
            },
        })

        current += timedelta(days=test_days)

    if not windows:
        return {"error": "Not enough data for walk-forward analysis"}

    # Aggregate OOS performance
    oos_results = [w["out_of_sample"] for w in windows if w["out_of_sample"]["total_trades"] > 0]
    if oos_results:
        avg_win_rate = sum(r["win_rate"] for r in oos_results) / len(oos_results)
        avg_pf = sum(r["profit_factor"] for r in oos_results) / len(oos_results)
        avg_sharpe = sum(r["sharpe"] for r in oos_results) / len(oos_results)
        total_oos_pnl = sum(r["net_pnl"] for r in oos_results)
        total_oos_trades = sum(r["total_trades"] for r in oos_results)
    else:
        avg_win_rate = avg_pf = avg_sharpe = total_oos_pnl = total_oos_trades = 0

    result = {
        "symbol": symbol,
        "full_period": f"{start} → {end}",
        "n_folds": len(windows),
        "train_days": train_days,
        "test_days": test_days,
        "aggregated_out_of_sample": {
            "total_trades": total_oos_trades,
            "avg_win_rate": round(avg_win_rate, 3),
            "avg_profit_factor": round(avg_pf, 3),
            "avg_sharpe": round(avg_sharpe, 3),
            "total_net_pnl": round(total_oos_pnl, 2),
            "consistent_positive_folds": sum(1 for r in oos_results if r["net_pnl"] > 0),
        },
        "folds": windows,
    }

    logger.info(
        "walk_forward_complete",
        folds=len(windows),
        oos_win_rate=avg_win_rate,
        oos_sharpe=avg_sharpe,
        oos_pnl=total_oos_pnl,
    )
    return result

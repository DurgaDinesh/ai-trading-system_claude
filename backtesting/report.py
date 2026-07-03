"""Backtesting report generator: prints and exports full performance analysis."""

import json
from pathlib import Path
from datetime import date
from typing import Optional
import structlog

from backtesting.backtest_engine import run_backtest
from backtesting.walk_forward import run_walk_forward

logger = structlog.get_logger(__name__)
REPORTS_DIR = Path("reports/backtests")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def print_backtest_report(result: dict):
    """Pretty-print a backtest result to console."""
    if "error" in result:
        print(f"\n❌ Backtest Error: {result['error']}")
        return

    print("\n" + "═" * 60)
    print(f"  BACKTEST REPORT — {result.get('symbol', 'N/A')}")
    print(f"  Period: {result.get('period', 'N/A')}")
    print(f"  Interval: {result.get('interval', 'N/A')}")
    print("═" * 60)
    print(f"  Initial Capital:     ₹{result.get('initial_capital', 0):>12,.2f}")
    print(f"  Final Capital:       ₹{result.get('final_capital', 0):>12,.2f}")
    print(f"  Total Return:        {result.get('total_return_pct', 0):>+11.2f}%")
    print(f"  Net P&L:             ₹{result.get('total_net_pnl', 0):>+11,.2f}")
    print("─" * 60)
    print(f"  Total Trades:        {result.get('total_trades', 0):>12}")
    print(f"  Winning Trades:      {result.get('winning_trades', 0):>12}")
    print(f"  Losing Trades:       {result.get('losing_trades', 0):>12}")
    print(f"  Win Rate:            {result.get('win_rate', 0)*100:>11.1f}%")
    print(f"  Profit Factor:       {result.get('profit_factor', 0):>12.3f}")
    print(f"  Avg P&L / Trade:     ₹{result.get('avg_pnl_per_trade', 0):>+11,.2f}")
    print(f"  Best Trade:          ₹{result.get('max_win', 0):>+11,.2f}")
    print(f"  Worst Trade:         ₹{result.get('max_loss', 0):>+11,.2f}")
    print("─" * 60)
    print(f"  Max Drawdown:        ₹{result.get('max_drawdown', 0):>12,.2f}  ({result.get('max_drawdown_pct', 0):.2f}%)")
    print(f"  Sharpe Ratio:        {result.get('sharpe_ratio', 0):>12.3f}")
    print(f"  Sortino Ratio:       {result.get('sortino_ratio', 0):>12.3f}")
    print("═" * 60)

    # Assessment
    win_rate = result.get("win_rate", 0)
    pf = result.get("profit_factor", 0)
    sharpe = result.get("sharpe_ratio", 0)
    drawdown_pct = result.get("max_drawdown_pct", 0)

    print("\n  📊 ASSESSMENT:")
    print(f"  {'✅' if win_rate >= 0.60 else '❌'} Win Rate {win_rate*100:.1f}% (Target: ≥60%)")
    print(f"  {'✅' if pf >= 1.5 else '❌'} Profit Factor {pf:.2f} (Target: ≥1.5)")
    print(f"  {'✅' if sharpe >= 1.0 else '❌'} Sharpe Ratio {sharpe:.3f} (Target: ≥1.0)")
    print(f"  {'✅' if drawdown_pct <= 15 else '❌'} Max Drawdown {drawdown_pct:.2f}% (Target: ≤15%)")

    all_pass = win_rate >= 0.60 and pf >= 1.5 and sharpe >= 1.0 and drawdown_pct <= 15
    verdict = "✅ STRATEGY APPROVED — Ready for paper trading" if all_pass else "❌ STRATEGY NEEDS IMPROVEMENT"
    print(f"\n  VERDICT: {verdict}")
    print("═" * 60 + "\n")


def save_backtest_report(result: dict, filename: Optional[str] = None) -> Path:
    """Save backtest result as JSON for dashboard consumption."""
    if not filename:
        today = date.today().strftime("%Y%m%d")
        symbol = result.get("symbol", "NIFTY")
        filename = f"backtest_{symbol}_{today}.json"
    path = REPORTS_DIR / filename
    # Remove large pnl_curve and trades list for JSON to keep it small
    export = {k: v for k, v in result.items() if k not in ("pnl_curve", "trades")}
    path.write_text(json.dumps(export, indent=2, default=str))
    logger.info("backtest_report_saved", path=str(path))
    return path


def run_full_pre_live_validation(symbol: str = "NIFTY") -> bool:
    """
    Run full backtest + walk-forward and return True if strategy is approved.
    Must be run before switching to live mode.
    """
    print("\n🔄 Running pre-live backtest validation...\n")

    # Full backtest
    result = run_backtest(symbol=symbol, interval="1h")
    print_backtest_report(result)
    save_backtest_report(result)

    # Walk-forward
    print("\n🔄 Running walk-forward validation...\n")
    wf_result = run_walk_forward(symbol=symbol)
    oos = wf_result.get("aggregated_out_of_sample", {})
    print(f"Walk-Forward OOS Results ({wf_result.get('n_folds', 0)} folds):")
    print(f"  Avg Win Rate:    {oos.get('avg_win_rate', 0)*100:.1f}%")
    print(f"  Avg Sharpe:      {oos.get('avg_sharpe', 0):.3f}")
    print(f"  Total OOS P&L:   ₹{oos.get('total_net_pnl', 0):+,.2f}")
    print(f"  Positive Folds:  {oos.get('consistent_positive_folds', 0)}/{wf_result.get('n_folds', 0)}")

    wf_path = REPORTS_DIR / f"walkforward_{symbol}_{date.today().strftime('%Y%m%d')}.json"
    wf_path.write_text(json.dumps(wf_result, indent=2, default=str))

    approved = (
        result.get("win_rate", 0) >= 0.55 and
        result.get("profit_factor", 0) >= 1.3 and
        result.get("sharpe_ratio", 0) >= 0.8 and
        result.get("max_drawdown_pct", 100) <= 20 and
        oos.get("avg_win_rate", 0) >= 0.50   # OOS bar slightly lower
    )
    print(f"\n{'✅ PRE-LIVE VALIDATION PASSED' if approved else '❌ PRE-LIVE VALIDATION FAILED'}")
    return approved

"""
Backtesting engine using vectorbt for fast vectorized simulation.
Supports full strategy replay on historical Nifty data with realistic costs.
"""

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog
import yaml

from core.market_data.historical import fetch_historical_yfinance
from core.analysis.technical import compute_all, get_ema_stack_signal, get_rsi_signal, get_macd_signal, get_vwap_signal

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))
RESULTS_DIR = Path("reports/backtests")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    Generate entry signals using the same confluence logic as the live engine.
    Returns a Series: +1 (long/CE), -1 (short/PE), 0 (no signal).
    """
    df = compute_all(df)
    signals = pd.Series(0, index=df.index)

    for i in range(200, len(df)):
        window = df.iloc[max(0, i-200):i+1]
        votes = []

        ema = get_ema_stack_signal(window)
        rsi = get_rsi_signal(window)
        macd = get_macd_signal(window)
        vwap = get_vwap_signal(window)

        if ema != 0: votes.append(ema)
        if rsi != 0: votes.append(rsi)
        if macd != 0: votes.append(macd)
        if vwap != 0: votes.append(vwap)

        min_confluence = _cfg["signals"]["min_confluence"]
        if not votes or len(votes) < min_confluence:
            continue

        net = sum(votes)
        if abs(net) >= min_confluence:
            signals.iloc[i] = 1 if net > 0 else -1

    return signals


def _apply_atr_levels(df: pd.DataFrame, signals: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute SL and TP1 series for each signal bar."""
    sl_series = pd.Series(np.nan, index=df.index)
    tp1_series = pd.Series(np.nan, index=df.index)
    tp2_series = pd.Series(np.nan, index=df.index)

    sl_mult = _cfg["backtesting"]["sl_atr"]
    tp1_mult = _cfg["backtesting"]["tp1_atr"]
    tp2_mult = _cfg["backtesting"]["tp2_atr"]

    for i in df.index:
        sig = signals.loc[i]
        if sig == 0:
            continue
        price = df.loc[i, "close"]
        atr = df.loc[i, "atr"] if "atr" in df.columns and not pd.isna(df.loc[i, "atr"]) else price * 0.01
        if sig == 1:  # Long
            sl_series.loc[i] = price - sl_mult * atr
            tp1_series.loc[i] = price + tp1_mult * atr
            tp2_series.loc[i] = price + tp2_mult * atr
        else:  # Short
            sl_series.loc[i] = price + sl_mult * atr
            tp1_series.loc[i] = price - tp1_mult * atr
            tp2_series.loc[i] = price - tp2_mult * atr

    return sl_series, tp1_series, tp2_series


def run_backtest(
    symbol: str = "NIFTY",
    start: Optional[date] = None,
    end: Optional[date] = None,
    interval: str = "1h",
    initial_capital: float = None,
) -> dict:
    """
    Run the full strategy backtest on historical data.
    Returns a comprehensive performance report dict.
    """
    bt_cfg = _cfg["backtesting"]
    start = start or date.fromisoformat(bt_cfg["default_start_date"])
    end = end or date.fromisoformat(bt_cfg["default_end_date"])
    initial_capital = initial_capital or bt_cfg["initial_capital"]
    commission = bt_cfg["commission_pct"]
    slippage = bt_cfg["slippage_pct"]

    logger.info("backtest_starting", symbol=symbol, start=str(start), end=str(end), interval=interval)

    df = fetch_historical_yfinance(symbol, start=start, end=end, interval=interval)
    if df.empty:
        logger.error("backtest_no_data", symbol=symbol)
        return {"error": "No data available for the specified period"}

    df = compute_all(df)
    signals = _generate_signals(df)
    sl_series, tp1_series, tp2_series = _apply_atr_levels(df, signals)

    # Simple simulation loop (vectorbt alternative for portability)
    tp1_alloc = _cfg["signals"]["tp_allocation"]["tp1"]
    capital = initial_capital
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_idx = None
    entry_sl = 0.0
    entry_tp1 = 0.0
    entry_tp2 = 0.0
    direction = 0
    tp1_booked = False
    remaining_fraction = 1.0

    # A signal on bar i can only be filled at bar i+1's open — the earliest
    # realistic execution point — never at the same bar's own close.
    pending_direction = None
    pending_sl = pending_tp1 = pending_tp2 = 0.0

    for i in range(len(df)):
        row = df.iloc[i]
        close = row["close"]
        idx = df.index[i]

        if pending_direction is not None and not in_trade:
            direction = pending_direction
            entry_price = row["open"] * (1 + slippage if direction == 1 else 1 - slippage)
            entry_sl = pending_sl
            entry_tp1 = pending_tp1
            entry_tp2 = pending_tp2
            entry_idx = idx
            in_trade = True
            tp1_booked = False
            remaining_fraction = 1.0
            pending_direction = None

        if in_trade:
            # Check exit conditions
            tp1_hit = close >= entry_tp1 if direction == 1 else close <= entry_tp1
            tp2_hit = close >= entry_tp2 if direction == 1 else close <= entry_tp2
            sl_hit = close <= entry_sl if direction == 1 else close >= entry_sl
            base_qty = capital * _cfg["capital"]["max_per_trade_pct"] / entry_price

            if sl_hit:
                fill = entry_sl * (1 - slippage if direction == 1 else 1 + slippage)
                pnl = (fill - entry_price) * direction
                gross = pnl * remaining_fraction * base_qty
                net = gross - gross * commission * 2
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": idx,
                    "direction": direction, "entry": entry_price, "exit": fill,
                    "pnl": net, "exit_reason": "SL",
                })
                capital += net
                in_trade = False

            elif tp2_hit:
                fill = entry_tp2 * (1 - slippage if direction == 1 else 1 + slippage)
                pnl = (fill - entry_price) * direction
                gross = pnl * remaining_fraction * base_qty
                net = gross - gross * commission * 2
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": idx,
                    "direction": direction, "entry": entry_price, "exit": fill,
                    "pnl": net, "exit_reason": "TP2",
                })
                capital += net
                in_trade = False

            elif tp1_hit and not tp1_booked:
                # Book the configured TP1 fraction once, move SL to breakeven,
                # and leave the rest of the position open.
                tp1_booked = True
                entry_sl = entry_price   # Trail SL to BE
                fill = entry_tp1
                pnl = (fill - entry_price) * direction
                gross = pnl * tp1_alloc * base_qty
                net = gross - gross * commission
                capital += net
                remaining_fraction -= tp1_alloc

        elif signals.iloc[i] != 0 and not pd.isna(sl_series.iloc[i]) and pending_direction is None:
            # Queue entry for the next bar's open (no same-bar lookahead fill)
            pending_direction = int(signals.iloc[i])
            pending_sl = sl_series.iloc[i]
            pending_tp1 = tp1_series.iloc[i]
            pending_tp2 = tp2_series.iloc[i]

    # Compute performance metrics
    if not trades:
        return {
            "symbol": symbol, "period": f"{start} → {end}", "total_trades": 0,
            "error": "No trades generated — check indicator parameters",
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_wins = sum(wins) if wins else 0
    gross_losses = abs(sum(losses))
    if gross_losses > 0:
        profit_factor = gross_wins / gross_losses
    else:
        profit_factor = float("inf") if gross_wins > 0 else 0.0

    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) else 0

    # Annualize using this backtest's actual trade frequency, not a fixed
    # 252-trading-day assumption — an intraday options system can take
    # several trades per day, which sqrt(252) would badly understate.
    period_days = max((end - start).days, 1)
    trades_per_year = len(pnls) / period_days * 365.25
    annualization = trades_per_year ** 0.5

    if len(pnls) > 1:
        import statistics
        mean_pnl = statistics.mean(pnls)
        std_pnl = statistics.stdev(pnls)
        sharpe = (mean_pnl / std_pnl) * annualization if std_pnl else 0
    else:
        sharpe = 0

    sortino_losses = [p for p in pnls if p < 0]
    if sortino_losses and len(sortino_losses) > 1:
        import statistics
        sortino_std = statistics.stdev(sortino_losses)
        sortino = (statistics.mean(pnls) / sortino_std) * annualization if sortino_std else 0
    else:
        sortino = 0

    result = {
        "symbol": symbol,
        "period": f"{start} → {end}",
        "interval": interval,
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100, 2),
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(len(wins) / len(trades), 3),
        "profit_factor": round(profit_factor, 3),
        "total_net_pnl": round(sum(pnls), 2),
        "avg_pnl_per_trade": round(sum(pnls) / len(pnls), 2),
        "max_win": round(max(pnls), 2),
        "max_loss": round(min(pnls), 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown / initial_capital * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "trades": trades,
        "pnl_curve": cumulative.tolist(),
    }

    logger.info(
        "backtest_complete",
        trades=len(trades), win_rate=result["win_rate"],
        sharpe=result["sharpe_ratio"], pnl=result["total_net_pnl"],
    )
    return result

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

from backtesting.simulation import simulate_signals
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

    period_days = max((end - start).days, 1)
    sim = simulate_signals(df, signals, sl_series, tp1_series, tp2_series,
                           initial_capital, period_days)
    if sim["total_trades"] == 0:
        return {
            "symbol": symbol, "period": f"{start} → {end}", "total_trades": 0,
            "error": "No trades generated — check indicator parameters",
        }

    result = {
        "symbol": symbol,
        "period": f"{start} → {end}",
        "interval": interval,
        "initial_capital": initial_capital,
        **sim,
    }

    logger.info(
        "backtest_complete",
        trades=result["total_trades"], win_rate=result["win_rate"],
        sharpe=result["sharpe_ratio"], pnl=result["total_net_pnl"],
    )
    return result

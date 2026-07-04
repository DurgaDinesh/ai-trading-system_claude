"""
Technical indicator calculations using the `ta` library (pure Python, Python 3.14 compatible).
All functions accept a pd.DataFrame with OHLCV columns and return an enriched DataFrame.
"""

import pandas as pd
import numpy as np
import ta
import structlog
import yaml

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))["indicators"]


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure lowercase columns
    df.columns = [c.lower() for c in df.columns]
    df = _add_ema(df)
    df = _add_rsi(df)
    df = _add_macd(df)
    df = _add_atr(df)
    df = _add_supertrend(df)
    df = _add_vwap(df)
    df = _add_bollinger(df)
    df = _add_stochastic(df)
    return df


def _add_ema(df: pd.DataFrame) -> pd.DataFrame:
    for period in _cfg["ema"]["periods"]:
        df[f"ema_{period}"] = ta.trend.ema_indicator(df["close"], window=period)
    return df


def _add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi"] = ta.momentum.rsi(df["close"], window=_cfg["rsi"]["period"])
    return df


def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
    c = _cfg["macd"]
    macd_obj = ta.trend.MACD(df["close"], window_fast=c["fast"], window_slow=c["slow"], window_sign=c["signal"])
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"] = macd_obj.macd_diff()
    return df


def _add_atr(df: pd.DataFrame) -> pd.DataFrame:
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=_cfg["atr"]["period"])
    return df


def _add_supertrend(df: pd.DataFrame) -> pd.DataFrame:
    """Manual Supertrend implementation (not in ta library)."""
    c = _cfg["supertrend"]
    period = c["period"]
    multiplier = c["multiplier"]

    atr = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=period)
    hl2 = (df["high"] + df["low"]) / 2

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    for i in range(1, len(df)):
        # Upper band
        if upper_band.iloc[i] < upper_band.iloc[i - 1] or df["close"].iloc[i - 1] > upper_band.iloc[i - 1]:
            upper_band.iloc[i] = upper_band.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        # Lower band
        if lower_band.iloc[i] > lower_band.iloc[i - 1] or df["close"].iloc[i - 1] < lower_band.iloc[i - 1]:
            lower_band.iloc[i] = lower_band.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i - 1]

        # Direction
        if i == 1:
            direction.iloc[i] = 1
        elif supertrend.iloc[i - 1] == upper_band.iloc[i - 1]:
            direction.iloc[i] = -1 if df["close"].iloc[i] > upper_band.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if df["close"].iloc[i] < lower_band.iloc[i] else -1

        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == -1 else upper_band.iloc[i]

    df["supertrend"] = supertrend
    df["supertrend_dir"] = direction   # -1 = bullish (price above ST line), 1 = bearish
    return df


def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Session-anchored VWAP: cumulative sum reset at the start of each trading day."""
    session = df.index.date
    typical = (df["high"] + df["low"] + df["close"]) / 3
    if "volume" in df.columns and df["volume"].sum() > 0:
        pv_cum = (typical * df["volume"]).groupby(session).cumsum()
        vol_cum = df["volume"].groupby(session).cumsum()
        df["vwap"] = pv_cum / vol_cum.replace(0, np.nan)
    else:
        # Index data has no volume — cumulative session-average typical price as proxy
        count = typical.groupby(session).cumcount() + 1
        df["vwap"] = typical.groupby(session).cumsum() / count
    return df


def _add_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    c = _cfg["bb"]
    bb = ta.volatility.BollingerBands(df["close"], window=c["period"], window_dev=c["std_dev"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    return df


def _add_stochastic(df: pd.DataFrame) -> pd.DataFrame:
    c = _cfg["stochastic"]
    stoch = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"],
        window=c["k_period"], smooth_window=c["smooth"]
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    return df


def get_ema_stack_signal(df: pd.DataFrame) -> int:
    """
    +1 if EMA9 > EMA21 > EMA50 > EMA200 (bullish stack).
    -1 if EMA9 < EMA21 < EMA50 < EMA200 (bearish stack).
    """
    row = df.iloc[-1]
    try:
        e9, e21, e50 = row.get("ema_9"), row.get("ema_21"), row.get("ema_50")
        e200 = row.get("ema_200")
        if any(pd.isna(v) for v in [e9, e21, e50]):
            return 0
        if pd.isna(e200):
            if e9 > e21 > e50:
                return 1
            if e9 < e21 < e50:
                return -1
            return 0
        if e9 > e21 > e50 > e200:
            return 1
        if e9 < e21 < e50 < e200:
            return -1
    except Exception:
        pass
    return 0


def get_rsi_signal(df: pd.DataFrame) -> int:
    """
    +1 if RSI crossed above 60 from below.
    -1 if RSI crossed below 40 from above.
    """
    if len(df) < 2 or "rsi" not in df.columns:
        return 0
    bullish_lvl = _cfg["rsi"]["bullish_crossover"]
    bearish_lvl = _cfg["rsi"]["bearish_crossover"]
    prev_rsi = df["rsi"].iloc[-2]
    curr_rsi = df["rsi"].iloc[-1]
    if pd.isna(prev_rsi) or pd.isna(curr_rsi):
        return 0
    if prev_rsi < bullish_lvl <= curr_rsi:
        return 1
    if prev_rsi > bearish_lvl >= curr_rsi:
        return -1
    return 0


def get_macd_signal(df: pd.DataFrame) -> int:
    """
    +1 on MACD bullish crossover (macd crosses above signal).
    -1 on bearish crossover.
    """
    if len(df) < 2 or "macd" not in df.columns:
        return 0
    prev, curr = df.iloc[-2], df.iloc[-1]
    if any(pd.isna(curr.get(c)) for c in ["macd", "macd_signal"]):
        return 0
    if prev["macd"] < prev["macd_signal"] and curr["macd"] >= curr["macd_signal"]:
        return 1
    if prev["macd"] > prev["macd_signal"] and curr["macd"] <= curr["macd_signal"]:
        return -1
    return 0


def get_vwap_signal(df: pd.DataFrame) -> int:
    """+1 if price above VWAP, -1 if below."""
    if "vwap" not in df.columns:
        return 0
    row = df.iloc[-1]
    if pd.isna(row.get("vwap")):
        return 0
    return 1 if row["close"] > row["vwap"] else -1


def get_supertrend_signal(df: pd.DataFrame) -> int:
    """+1 bullish (price above supertrend), -1 bearish."""
    if "supertrend_dir" not in df.columns:
        return 0
    val = df["supertrend_dir"].iloc[-1]
    if pd.isna(val):
        return 0
    return 1 if val == -1 else -1   # dir=-1 means price above ST (bullish)


def get_bollinger_signal(df: pd.DataFrame, squeeze_lookback: int = 20) -> int:
    """
    1 if close breaks above bb_upper following a bandwidth squeeze,
    -1 if it breaks below bb_lower following a squeeze, else 0.
    Reads columns already produced by compute_all() — no new indicator math.
    """
    required = {"close", "bb_upper", "bb_lower", "bb_width"}
    if not required.issubset(df.columns) or len(df) < squeeze_lookback + 1:
        return 0
    close = df["close"].iloc[-1]
    bb_upper = df["bb_upper"].iloc[-1]
    bb_lower = df["bb_lower"].iloc[-1]
    width_now = df["bb_width"].iloc[-1]
    width_recent_min = df["bb_width"].iloc[-(squeeze_lookback + 1):-1].min()
    was_squeezed = width_now <= width_recent_min * 1.1
    if not was_squeezed:
        return 0
    if close > bb_upper:
        return 1
    if close < bb_lower:
        return -1
    return 0


def get_stochastic_signal(df: pd.DataFrame) -> int:
    """
    1 if %K crosses above %D coming from oversold (prev %K < 20), -1 if %K
    crosses below %D coming from overbought (prev %K > 80), else 0.
    """
    required = {"stoch_k", "stoch_d"}
    if not required.issubset(df.columns) or len(df) < 2:
        return 0
    k, d = df["stoch_k"].iloc[-1], df["stoch_d"].iloc[-1]
    prev_k, prev_d = df["stoch_k"].iloc[-2], df["stoch_d"].iloc[-2]
    crossed_up = prev_k <= prev_d and k > d
    crossed_down = prev_k >= prev_d and k < d
    if crossed_up and prev_k < 20:
        return 1
    if crossed_down and prev_k > 80:
        return -1
    return 0


def compute_premium_levels(entry_premium: float, cfg: dict) -> dict:
    """
    Calculate SL/TP levels as a percentage move of the option premium itself.
    Both CE and PE positions are long premium (we only ever buy options), so
    stop-loss is always below entry and targets are always above entry.
    """
    sl_pct = cfg["signals"]["sl_premium_pct"]
    tp_pct = cfg["signals"]["tp_premium_pct"]
    return {
        "stop_loss": round(entry_premium * (1 - sl_pct), 2),
        "tp1": round(entry_premium * (1 + tp_pct["tp1"]), 2),
        "tp2": round(entry_premium * (1 + tp_pct["tp2"]), 2),
        "tp3": round(entry_premium * (1 + tp_pct["tp3"]), 2),
    }

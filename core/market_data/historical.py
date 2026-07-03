"""Historical OHLCV data fetcher — Kite Connect primary, yfinance fallback."""

from datetime import datetime, timedelta, date
from typing import Optional

import pandas as pd
import yfinance as yf
import structlog

from core.market_data.kite_client import kite_client

logger = structlog.get_logger(__name__)

# Kite interval strings
INTERVAL_MAP = {
    "1m": "minute",
    "3m": "3minute",
    "5m": "5minute",
    "10m": "10minute",
    "15m": "15minute",
    "30m": "30minute",
    "1h": "60minute",
    "1d": "day",
}

# yfinance ticker map for fallback / global indices
YFINANCE_MAP = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "CRUDE": "CL=F",
    "DXY": "DX-Y.NYB",
    "SPX": "^GSPC",
    "DOW": "^DJI",
    "NASDAQ": "^IXIC",
    "VIX": "^VIX",
}


def fetch_historical_kite(
    instrument_token: int,
    from_dt: datetime,
    to_dt: datetime,
    interval: str = "5m",
    continuous: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV data from Kite Connect."""
    kite_interval = INTERVAL_MAP.get(interval, "5minute")
    records = kite_client.kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_dt,
        to_date=to_dt,
        interval=kite_interval,
        continuous=continuous,
        oi=True,
    )
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


def fetch_historical_yfinance(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """Fetch OHLCV from yfinance — used for global indices and backtesting."""
    ticker = YFINANCE_MAP.get(symbol.upper(), symbol)
    yf_kwargs = {"period": period, "interval": interval, "auto_adjust": True}
    if start and end:
        yf_kwargs = {"start": start, "end": end, "interval": interval, "auto_adjust": True}

    df = yf.download(ticker, **yf_kwargs, progress=False)
    if df.empty:
        logger.warning("yfinance_no_data", symbol=ticker)
        return df

    # yfinance 1.4+ returns MultiIndex columns: ('Close', '^NSEI') — flatten to 'close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    return df


def fetch_intraday_nifty(days_back: int = 30, interval: str = "5m") -> pd.DataFrame:
    """Convenience: fetch recent Nifty 50 intraday data via yfinance."""
    end = datetime.today()
    start = end - timedelta(days=days_back)
    return fetch_historical_yfinance("NIFTY", start=start.date(), end=end.date(), interval=interval)


def fetch_global_snapshot() -> dict[str, float]:
    """
    Single-shot fetch of current levels for all global market indicators.
    Returns a flat dict for use in pre-market analysis.
    """
    symbols = {
        "gift_nifty": "^NSEI",       # Closest proxy (real Gift Nifty via Kite)
        "dow": "^DJI",
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "dxy": "DX-Y.NYB",
        "crude": "CL=F",
        "gold": "GC=F",
        "silver": "SI=F",
        "vix_global": "^VIX",
    }
    result = {}
    for key, ticker in symbols.items():
        try:
            data = yf.download(ticker, period="2d", interval="1d", auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            if not data.empty and len(data) >= 2:
                prev_close = float(data["Close"].iloc[-2])
                last_close = float(data["Close"].iloc[-1])
                result[key] = last_close
                result[f"{key}_chg_pct"] = round((last_close - prev_close) / prev_close * 100, 3)
            elif not data.empty:
                result[key] = float(data["Close"].iloc[-1])
                result[f"{key}_chg_pct"] = 0.0
        except Exception as e:
            logger.warning("global_snapshot_fetch_failed", key=key, error=str(e))
            result[key] = None
            result[f"{key}_chg_pct"] = None

    return result

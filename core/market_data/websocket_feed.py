"""Kite Connect WebSocket tick feed with in-memory OHLCV aggregation."""

import threading
from datetime import datetime
from typing import Callable, Optional
from collections import defaultdict

import pandas as pd
import structlog
from kiteconnect import KiteTicker

from core.market_data.kite_client import kite_client, vault

logger = structlog.get_logger(__name__)


class TickAggregator:
    """Aggregates raw ticks into 1m / 5m OHLCV candles in memory."""

    def __init__(self, interval_minutes: int = 5):
        self.interval = interval_minutes
        self._ticks: dict[int, list] = defaultdict(list)
        self._candles: dict[int, pd.DataFrame] = {}
        self._lock = threading.Lock()

    def add_tick(self, token: int, price: float, ts: datetime, volume: int = 0, oi: int = 0):
        with self._lock:
            self._ticks[token].append({"ts": ts, "price": price, "volume": volume, "oi": oi})

    def get_latest_candle(self, token: int) -> Optional[dict]:
        """Build the most recent complete candle on demand."""
        with self._lock:
            ticks = self._ticks.get(token, [])
        if not ticks:
            return None
        df = pd.DataFrame(ticks).set_index("ts")
        df.index = pd.to_datetime(df.index)
        ohlcv = df["price"].resample(f"{self.interval}min").ohlc()
        ohlcv["volume"] = df["volume"].resample(f"{self.interval}min").sum()
        ohlcv["oi"] = df["oi"].resample(f"{self.interval}min").last()
        ohlcv.dropna(inplace=True)
        if len(ohlcv) < 2:
            return None
        # Return second-to-last (last fully completed candle)
        row = ohlcv.iloc[-2]
        return {
            "open": row["open"], "high": row["high"],
            "low": row["low"], "close": row["close"],
            "volume": row["volume"], "oi": row.get("oi"),
            "timestamp": ohlcv.index[-2],
        }

    def get_all_candles(self, token: int) -> pd.DataFrame:
        with self._lock:
            ticks = list(self._ticks.get(token, []))
        if not ticks:
            return pd.DataFrame()
        df = pd.DataFrame(ticks).set_index("ts")
        df.index = pd.to_datetime(df.index)
        ohlcv = df["price"].resample(f"{self.interval}min").ohlc()
        ohlcv["volume"] = df["volume"].resample(f"{self.interval}min").sum()
        return ohlcv.dropna()


class LiveFeed:
    """
    Manages the KiteTicker WebSocket connection.
    Subscribers register callbacks that receive ticks in real time.
    """

    def __init__(self, interval_minutes: int = 5):
        self._ticker: Optional[KiteTicker] = None
        self._subscribed_tokens: set[int] = set()
        self._callbacks: list[Callable] = []
        self.aggregator = TickAggregator(interval_minutes)
        self._running = False

    def subscribe(self, tokens: list[int]):
        self._subscribed_tokens.update(tokens)
        if self._ticker and self._running:
            self._ticker.subscribe(list(tokens))
            self._ticker.set_mode(self._ticker.MODE_FULL, list(tokens))

    def add_callback(self, fn: Callable):
        """Register a function(ticks: list[dict]) → None called on each tick batch."""
        self._callbacks.append(fn)

    def start(self):
        self._ticker = KiteTicker(
            api_key=vault.kite_api_key,
            access_token=kite_client._access_token,
        )

        def on_ticks(ws, ticks):
            for tick in ticks:
                token = tick["instrument_token"]
                price = tick.get("last_price", 0)
                ts = tick.get("exchange_timestamp", datetime.now())
                volume = tick.get("volume", 0)
                oi = tick.get("oi", 0)
                self.aggregator.add_tick(token, price, ts, volume, oi)
            for cb in self._callbacks:
                try:
                    cb(ticks)
                except Exception as e:
                    logger.error("tick_callback_error", error=str(e))

        def on_connect(ws, response):
            logger.info("websocket_connected")
            self._running = True
            if self._subscribed_tokens:
                tokens = list(self._subscribed_tokens)
                ws.subscribe(tokens)
                ws.set_mode(ws.MODE_FULL, tokens)

        def on_close(ws, code, reason):
            logger.warning("websocket_closed", code=code, reason=reason)
            self._running = False

        def on_error(ws, code, reason):
            logger.error("websocket_error", code=code, reason=reason)

        def on_reconnect(ws, attempts_count):
            logger.info("websocket_reconnecting", attempt=attempts_count)

        self._ticker.on_ticks = on_ticks
        self._ticker.on_connect = on_connect
        self._ticker.on_close = on_close
        self._ticker.on_error = on_error
        self._ticker.on_reconnect = on_reconnect
        self._ticker.connect(threaded=True)
        logger.info("websocket_feed_started")

    def stop(self):
        if self._ticker:
            self._ticker.close()
        self._running = False
        logger.info("websocket_feed_stopped")

    @property
    def is_running(self) -> bool:
        return self._running


live_feed = LiveFeed(interval_minutes=5)

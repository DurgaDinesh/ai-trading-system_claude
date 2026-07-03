"""Zerodha Kite Connect session manager with auto-reconnect and token caching."""

import os
import json
import time
from pathlib import Path
from datetime import datetime, date
from typing import Optional

import yaml
import structlog
from kiteconnect import KiteConnect

from core.security.vault import vault

logger = structlog.get_logger(__name__)
_TOKEN_CACHE = Path("data/.kite_token_cache.json")


class KiteClient:
    """
    Singleton wrapper around KiteConnect.
    Handles login token persistence so we don't need to re-authenticate
    on every restart within the same trading day.
    """

    def __init__(self):
        self._kite: Optional[KiteConnect] = None
        self._logged_in = False
        self._access_token: Optional[str] = None

    def _load_cached_token(self) -> Optional[str]:
        if not _TOKEN_CACHE.exists():
            return None
        try:
            data = json.loads(_TOKEN_CACHE.read_text())
            if data.get("date") == str(date.today()):
                return data.get("access_token")
        except Exception:
            pass
        return None

    def _save_token(self, token: str):
        _TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _TOKEN_CACHE.write_text(json.dumps({"date": str(date.today()), "access_token": token}))

    def connect(self, request_token: Optional[str] = None) -> KiteConnect:
        """
        Initialize the Kite session.
        - If a cached access token exists for today, reuse it.
        - Otherwise, exchange the request_token (from login URL) for an access token.

        Usage (first run each day):
          1. Open the login URL printed in logs
          2. After login, copy the request_token from the redirect URL
          3. Pass it here OR set env var KITE_REQUEST_TOKEN
        """
        self._kite = KiteConnect(api_key=vault.kite_api_key)

        cached = self._load_cached_token()
        if cached:
            self._kite.set_access_token(cached)
            self._access_token = cached
            self._logged_in = True
            logger.info("kite_connected_from_cache", date=str(date.today()))
            return self._kite

        req_token = request_token or os.environ.get("KITE_REQUEST_TOKEN")
        if not req_token:
            login_url = self._kite.login_url()
            logger.info("kite_login_required", login_url=login_url)
            print(f"\n[ACTION REQUIRED] Open this URL to login:\n{login_url}\n")
            req_token = input("Paste request_token from redirect URL: ").strip()

        data = self._kite.generate_session(req_token, api_secret=vault.kite_api_secret)
        self._access_token = data["access_token"]
        self._kite.set_access_token(self._access_token)
        self._save_token(self._access_token)
        self._logged_in = True
        logger.info("kite_session_created", user_id=vault.kite_user_id)
        return self._kite

    def connect_if_possible(self) -> bool:
        """
        Non-interactive connect attempt — never prompts for a request_token.
        For paper mode: use a live Kite session if one's already available
        (cached token from today, or KITE_REQUEST_TOKEN env var), otherwise
        skip silently so callers can fall back to non-Kite data sources
        (yfinance spot + Black-Scholes estimated premiums).
        """
        self._kite = KiteConnect(api_key=vault.kite_api_key)

        cached = self._load_cached_token()
        if cached:
            self._kite.set_access_token(cached)
            self._access_token = cached
            self._logged_in = True
            logger.info("kite_connected_from_cache", date=str(date.today()))
            return True

        req_token = os.environ.get("KITE_REQUEST_TOKEN")
        if not req_token:
            self._kite = None
            return False

        try:
            data = self._kite.generate_session(req_token, api_secret=vault.kite_api_secret)
            self._access_token = data["access_token"]
            self._kite.set_access_token(self._access_token)
            self._save_token(self._access_token)
            self._logged_in = True
            logger.info("kite_session_created", user_id=vault.kite_user_id)
            return True
        except Exception as e:
            logger.warning("kite_connect_if_possible_failed", error=str(e))
            self._kite = None
            return False

    @property
    def is_connected(self) -> bool:
        return self._logged_in and self._kite is not None

    @property
    def kite(self) -> KiteConnect:
        if not self._kite or not self._logged_in:
            raise RuntimeError("Kite session not initialized. Call connect() first.")
        return self._kite

    def get_instrument_token(self, exchange: str, symbol: str) -> Optional[int]:
        instruments = self.kite.instruments(exchange)
        for i in instruments:
            if i["tradingsymbol"] == symbol:
                return i["instrument_token"]
        return None

    def get_option_chain(self, underlying: str, expiry: date) -> list[dict]:
        """Return all option contracts for a given underlying and expiry."""
        instruments = self.kite.instruments("NFO")
        chain = [
            i for i in instruments
            if i["name"] == underlying
            and isinstance(i.get("expiry"), date)
            and i["expiry"] == expiry
            and i["instrument_type"] in ("CE", "PE")
        ]
        return sorted(chain, key=lambda x: x["strike"])

    def get_ltp(self, instruments: list[str]) -> dict[str, float]:
        """Get last traded price for a list of instruments."""
        quotes = self.kite.ltp(instruments)
        return {k: v["last_price"] for k, v in quotes.items()}

    def get_quote(self, instruments: list[str]) -> dict:
        return self.kite.quote(instruments)

    def place_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,   # kite.TRANSACTION_TYPE_BUY | SELL
        quantity: int,
        order_type: str = "MARKET",
        price: float = 0,
        product: str = "MIS",
        validity: str = "DAY",
        tag: str = "NiftySniper",
    ) -> str:
        order_id = self.kite.place_order(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            price=price,
            product=product,
            validity=validity,
            variety=self.kite.VARIETY_REGULAR,
            tag=tag,
        )
        logger.info("order_placed", order_id=order_id, symbol=tradingsymbol, qty=quantity)
        return order_id

    def cancel_order(self, order_id: str) -> str:
        return self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=order_id)

    def get_positions(self) -> dict:
        return self.kite.positions()

    def get_orders(self) -> list[dict]:
        return self.kite.orders()

    def get_margins(self) -> dict:
        return self.kite.margins()


kite_client = KiteClient()

"""Detects anomalous order patterns: oversized positions and order bursts."""

import time
from collections import deque
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


class AnomalyDetector:
    def __init__(
        self,
        avg_order_size: float = 25000,
        size_multiplier_threshold: float = 2.0,
        max_orders_per_window: int = 5,
        window_seconds: int = 60,
    ):
        self._avg_order_size = avg_order_size
        self._size_multiplier = size_multiplier_threshold
        self._max_orders = max_orders_per_window
        self._window = window_seconds
        self._order_timestamps: deque = deque()
        self._order_values: list[float] = []

    def check_order(self, order_value: float) -> tuple[bool, Optional[str]]:
        """
        Returns (is_anomalous, reason).
        is_anomalous=True means the order should be blocked or flagged.
        """
        now = time.time()

        # Burst check — purge old timestamps outside the rolling window
        while self._order_timestamps and now - self._order_timestamps[0] > self._window:
            self._order_timestamps.popleft()

        if len(self._order_timestamps) >= self._max_orders:
            reason = (
                f"BURST ANOMALY: {len(self._order_timestamps)} orders in "
                f"{self._window}s (max allowed: {self._max_orders})"
            )
            logger.warning("anomaly_burst_detected", reason=reason)
            return True, reason

        # Size anomaly check
        avg = self._avg_order_size if not self._order_values else sum(self._order_values) / len(self._order_values)
        if avg > 0 and order_value > avg * self._size_multiplier:
            reason = (
                f"SIZE ANOMALY: order ₹{order_value:,.0f} is "
                f"{order_value/avg:.1f}x average ₹{avg:,.0f} "
                f"(threshold: {self._size_multiplier}x)"
            )
            logger.warning("anomaly_size_detected", reason=reason)
            return True, reason

        # All clear — record the timestamp for burst tracking. The order's
        # value is recorded separately via update_average() once the order
        # actually completes, so a checked-but-not-yet-placed order doesn't
        # get double-counted into the rolling size average.
        self._order_timestamps.append(now)

        return False, None

    def update_average(self, completed_order_value: float):
        """Call after each completed order to update rolling average."""
        self._order_values.append(completed_order_value)
        if len(self._order_values) > 500:
            self._order_values = self._order_values[-500:]


anomaly_detector = AnomalyDetector()

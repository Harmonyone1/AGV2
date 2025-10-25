"""Resilient OrderService wrapping TLAPI."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from tradelocker_api import TLAPI
from utils.constants import MAX_RETRIES, BACKOFF_FACTOR, new_corr_id


class OrderService:
    """Wraps TLAPI with retries and a simple circuit breaker."""

    def __init__(self, api: TLAPI, breaker_timeout: int = 30) -> None:
        self.log = logging.getLogger(self.__class__.__name__)
        self.api = api
        self.max_retries = MAX_RETRIES
        self.backoff = BACKOFF_FACTOR
        self.breaker_timeout = breaker_timeout
        self._failures = 0
        self._breaker_open_until = 0.0

    def _check_breaker(self) -> None:
        if time.time() < self._breaker_open_until:
            raise RuntimeError("OrderService circuit breaker open")

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.max_retries:
            self._breaker_open_until = time.time() + self.breaker_timeout
            self.log.error("Circuit breaker opened for %ds", self.breaker_timeout)
            self._failures = 0

    async def create_order(self, *args, **kwargs) -> Optional[int]:
        self._check_breaker()
        for attempt in range(self.max_retries):
            try:
                oid = await asyncio.get_running_loop().run_in_executor(
                    None, self.api.create_order, *args
                )
                self._failures = 0
                return oid
            except Exception as e:
                self.log.warning("create_order attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(self.backoff * (2 ** attempt))
                self._record_failure()
        return None

    async def modify_order(self, order_id: int, params: Dict[str, Any]) -> bool:
        self._check_breaker()
        for attempt in range(self.max_retries):
            try:
                ok = await asyncio.get_running_loop().run_in_executor(
                    None, self.api.modify_order, order_id, params
                )
                self._failures = 0
                return bool(ok)
            except Exception as e:
                self.log.warning("modify_order attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(self.backoff * (2 ** attempt))
                self._record_failure()
        return False

    async def get_position_id(self, order_id: int) -> Optional[int]:
        self._check_breaker()
        try:
            pos_id = await asyncio.get_running_loop().run_in_executor(
                None, self.api.get_position_id_from_order_id, order_id
            )
            self._failures = 0
            return pos_id
        except Exception as e:
            self.log.error("get_position_id failed: %s", e)
            self._record_failure()
            return None

    async def subscribe_order_updates(self):
        """Yield push order updates from the API if supported."""
        if hasattr(self.api, "stream_order_updates"):
            async for update in self.api.stream_order_updates():
                update["corr_id"] = new_corr_id()
                yield update
        else:
            raise NotImplementedError("API does not support order update streaming")

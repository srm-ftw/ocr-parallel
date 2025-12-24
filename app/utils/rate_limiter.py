"""Rate limiting utilities."""

import asyncio
import logging
from typing import Optional

from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter using token bucket algorithm."""

    def __init__(self, max_rate: int, time_period: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_rate: Maximum number of requests allowed
            time_period: Time period in seconds (default: 60 for per-minute rate)
        """
        self.limiter = AsyncLimiter(max_rate=max_rate, time_period=time_period)
        self.max_rate = max_rate
        self.time_period = time_period
        logger.info(f"Rate limiter initialized: {max_rate} requests per {time_period} seconds")

    async def acquire(self):
        """Acquire a token from the rate limiter."""
        await self.limiter.acquire()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


class ConcurrencyLimiter:
    """Concurrency limiter using semaphore."""

    def __init__(self, max_concurrent: int):
        """
        Initialize concurrency limiter.

        Args:
            max_concurrent: Maximum number of concurrent operations
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        logger.info(f"Concurrency limiter initialized: max {max_concurrent} concurrent operations")

    async def acquire(self):
        """Acquire a semaphore slot."""
        await self.semaphore.acquire()

    def release(self):
        """Release a semaphore slot."""
        self.semaphore.release()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release()


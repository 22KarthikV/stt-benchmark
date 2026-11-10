"""Exponential backoff for transient provider errors."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .providers.base import TransientProviderError

T = TypeVar("T")


def _default_jitter() -> float:
    return random.uniform(0, 0.25)


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[], float] = _default_jitter,
    on_attempt: Callable[[int], None] | None = None,
) -> T:
    """Call fn up to ``attempts`` times; sleep 1s, 2s, 4s (+jitter) between transient failures.

    Permanent errors (and anything that is not TransientProviderError) propagate immediately.
    """
    for attempt in range(1, attempts + 1):
        if on_attempt:
            on_attempt(attempt)
        try:
            return await fn()
        except TransientProviderError:
            if attempt == attempts:
                raise
            await sleep(base_delay * 2 ** (attempt - 1) + jitter())
    raise AssertionError("unreachable")

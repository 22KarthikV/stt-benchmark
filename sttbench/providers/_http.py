"""Shared HTTP / WebSocket error classification for the cloud adapters."""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
import websockets

from .base import PermanentProviderError, StreamEvent, TransientProviderError


def raise_for_status(resp: httpx.Response) -> None:
    """429 and 5xx are transient (retried); every other non-2xx is permanent."""
    if resp.status_code < 400:
        return
    detail = f"HTTP {resp.status_code}: {resp.text[:300]}"
    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientProviderError(detail)
    raise PermanentProviderError(detail)


async def http_call(fn: Callable[[], Awaitable[Any]]) -> Any:
    try:
        return await fn()
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise TransientProviderError(f"network error: {type(e).__name__}: {e}") from e


async def guard_ws(events: AsyncIterator[StreamEvent]) -> AsyncIterator[StreamEvent]:
    """Translate websocket failures into the provider error classes."""
    try:
        async for ev in events:
            yield ev
    except websockets.exceptions.InvalidStatus as e:
        code = e.response.status_code
        cls = TransientProviderError if code == 429 or code >= 500 else PermanentProviderError
        raise cls(f"websocket handshake rejected: HTTP {code}") from e
    except (OSError, websockets.exceptions.WebSocketException) as e:
        raise TransientProviderError(f"websocket error: {type(e).__name__}: {e}") from e

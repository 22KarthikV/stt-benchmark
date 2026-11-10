"""Provider interface, result types and error classes."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class ProviderError(Exception):
    pass


class TransientProviderError(ProviderError):
    """Retryable: HTTP 429/5xx, network errors, timeouts."""


class PermanentProviderError(ProviderError):
    """Not retryable: auth failures, bad requests, malformed responses."""


@dataclass(frozen=True)
class Transcript:
    text: str
    provider: str
    model: str
    duration_s: float
    latency_s: float = 0.0  # filled in by the runner (harness wall-clock)


@dataclass(frozen=True)
class StreamEvent:
    kind: Literal["partial", "final"]
    text: str
    t_recv: float  # time.monotonic() stamped by the harness on receipt
    audio_start_s: float | None = None  # offsets into the stream, when the provider reports them
    audio_end_s: float | None = None


class Provider(ABC):
    name: str
    model: str
    live: bool = True  # False for fakes

    @abstractmethod
    async def transcribe_file(self, path: Path) -> Transcript: ...

    @abstractmethod
    def stream(self, pcm_chunks: AsyncIterator[bytes]) -> AsyncIterator[StreamEvent]: ...

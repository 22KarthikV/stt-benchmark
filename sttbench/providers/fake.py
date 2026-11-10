"""Deterministic fake provider for tests and the no-key demo. Output is NOT real ASR."""

import asyncio
import math
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from ..audio import read_wav
from .base import (
    PermanentProviderError,
    Provider,
    StreamEvent,
    Transcript,
    TransientProviderError,
)

DEFAULT_TEXT = "the quick brown fox"


class FakeProvider(Provider):
    name = "fake"
    model = "fake-1"
    live = False

    def __init__(
        self,
        transcripts: dict[str, str] | None = None,
        errors: dict[str, Exception] | None = None,
        fail_times: dict[str, int] | None = None,
        utterances: list[str] | None = None,
        chunks_per_utterance: int = 10,
        chunk_s: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.transcripts = transcripts or {}
        self.errors = errors or {}
        self.fail_times = dict(fail_times or {})
        self.utterances = utterances or [DEFAULT_TEXT]
        self.chunks_per_utterance = chunks_per_utterance
        self.chunk_s = chunk_s
        self.clock = clock

    async def transcribe_file(self, path: Path) -> Transcript:
        stem = Path(path).stem
        if stem in self.errors:
            raise self.errors[stem]
        if self.fail_times.get(stem, 0) > 0:
            self.fail_times[stem] -= 1
            raise TransientProviderError(f"fake transient failure for {stem}")
        try:
            duration = read_wav(Path(path)).duration_s
        except Exception as e:
            raise PermanentProviderError(str(e)) from e
        await asyncio.sleep(0)
        return Transcript(self.transcripts.get(stem, DEFAULT_TEXT), self.name, self.model, duration)

    async def stream(self, pcm_chunks: AsyncIterator[bytes]) -> AsyncIterator[StreamEvent]:
        n = self.chunks_per_utterance
        utt_index = 0
        in_utt = 0  # chunks received in the current utterance
        total = 0
        start_chunk = 0

        def words() -> list[str]:
            return self.utterances[utt_index % len(self.utterances)].split()

        async for _ in pcm_chunks:
            await asyncio.sleep(0)
            total += 1
            in_utt += 1
            w = words()
            start_s, end_s = start_chunk * self.chunk_s, total * self.chunk_s
            if in_utt >= n:
                yield StreamEvent("final", " ".join(w), self.clock(), start_s, end_s)
                utt_index += 1
                in_utt = 0
                start_chunk = total
            else:
                shown = max(1, math.ceil(len(w) * in_utt / n))
                yield StreamEvent("partial", " ".join(w[:shown]), self.clock(), start_s, end_s)
        if in_utt:
            yield StreamEvent(
                "final", " ".join(words()), self.clock(), start_chunk * self.chunk_s,
                total * self.chunk_s,
            )

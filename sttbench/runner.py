"""Benchmark loop with per-clip error isolation."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .audio import read_wav
from .providers.base import Provider
from .retry import retry_async
from .wer import word_errors


async def _run_clip(
    provider: Provider,
    clip: Path,
    reference: str,
    *,
    attempts: int,
    sleep: Callable[[float], Awaitable[None]],
    clock: Callable[[], float],
) -> dict[str, Any]:
    record: dict[str, Any] = {"clip": clip.stem, "reference": reference}
    count = 0

    def on_attempt(n: int) -> None:
        nonlocal count
        count = n

    async def attempt():
        t0 = clock()
        transcript = await provider.transcribe_file(clip)
        return transcript, clock() - t0

    try:
        transcript, latency = await retry_async(
            attempt, attempts=attempts, sleep=sleep, on_attempt=on_attempt
        )
        edits, n_ref = word_errors(reference, transcript.text)
        record.update(
            status="ok",
            text=transcript.text,
            edits=edits,
            ref_words=n_ref,
            wer=edits / n_ref,
            latency_s=latency,
            duration_s=transcript.duration_s,
        )
    except Exception as e:  # isolate every per-clip failure; the run continues
        record.update(status="failed", error=f"{type(e).__name__}: {e}")
    record["attempts"] = count
    return record


async def run_benchmark(
    providers: dict[str, Provider],
    skipped: dict[str, str],
    clips_dir: Path,
    truth_dir: Path,
    *,
    meta: dict[str, Any],
    attempts: int = 4,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.perf_counter,
    concurrency: int = 1,
) -> dict[str, Any]:
    clips = sorted(Path(clips_dir).glob("*.wav"))
    refs = {c.stem: (Path(truth_dir) / f"{c.stem}.txt").read_text(encoding="utf-8") for c in clips}
    audio_s = sum(read_wav(c).duration_s for c in clips)
    results: dict[str, Any] = {"meta": dict(meta), "providers": {}}
    for name, reason in skipped.items():
        results["providers"][name] = {"status": "skipped", "reason": reason}
    for name, provider in providers.items():
        sem = asyncio.Semaphore(max(1, concurrency))

        async def guarded(clip: Path, provider: Provider = provider, sem=sem) -> dict[str, Any]:
            async with sem:
                return await _run_clip(
                    provider, clip, refs[clip.stem], attempts=attempts, sleep=sleep, clock=clock
                )

        records = await asyncio.gather(*(guarded(c) for c in clips))
        results["providers"][name] = {
            "status": "ok",
            "model": provider.model,
            "live": provider.live,
            "audio_s_sent": audio_s,
            "clips": list(records),
        }
    return results

"""Latency metrics for the streaming demo.

``sent[i]`` is the monotonic time chunk ``i`` (covering [i*chunk_s, (i+1)*chunk_s) of the audio)
was handed to the provider. An utterance ends at each ``final`` event.

* time_to_first_partial = first partial's t_recv - send time of the utterance's first audio chunk
* final_latency         = final's t_recv - send time of the chunk holding the utterance's last audio

Where the provider reports audio offsets (``audio_start_s`` / ``audio_end_s``) they locate those
chunks exactly. Without offsets the fallback is: the utterance starts right after the previous
one's last chunk, and ends at the last chunk sent before the final arrived (this under-reports
latency for providers that lag behind the audio). File replay measures provider + network latency,
not microphone/capture latency.
"""

import math
from collections.abc import Sequence
from typing import Any

from ..providers.base import StreamEvent
from ..report import percentile


def _start_chunk(offset_s: float, chunk_s: float) -> int:
    return int(round(offset_s / chunk_s, 6))


def _end_chunk(offset_s: float, chunk_s: float) -> int:
    return max(0, math.ceil(round(offset_s / chunk_s, 6)) - 1)


def compute_stream_metrics(
    sent: Sequence[float], events: Sequence[StreamEvent], chunk_s: float = 0.1
) -> list[dict[str, Any]]:
    if not sent:
        return []
    last = len(sent) - 1

    def clamp(i: int) -> int:
        return max(0, min(i, last))

    out: list[dict[str, Any]] = []
    first_partial: StreamEvent | None = None
    prev_end = -1
    for ev in events:
        if ev.kind == "partial":
            if first_partial is None:
                first_partial = ev
            continue
        if ev.audio_start_s is not None:
            start = clamp(_start_chunk(ev.audio_start_s, chunk_s))
        elif first_partial is not None and first_partial.audio_start_s is not None:
            start = clamp(_start_chunk(first_partial.audio_start_s, chunk_s))
        else:
            start = clamp(prev_end + 1)
        if ev.audio_end_s is not None:
            end = clamp(_end_chunk(ev.audio_end_s, chunk_s))
        else:
            end = clamp(max((i for i, t in enumerate(sent) if t <= ev.t_recv), default=0))
        out.append(
            {
                "utterance": len(out) + 1,
                "text": ev.text,
                "time_to_first_partial": (
                    None if first_partial is None else first_partial.t_recv - sent[start]
                ),
                "final_latency": ev.t_recv - sent[end],
            }
        )
        prev_end = end
        first_partial = None
    return out


def summarize_stream(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def p50(key: str) -> float | None:
        vals = [m[key] for m in metrics if m[key] is not None]
        return percentile(vals, 50) if vals else None

    return {
        "utterances": len(metrics),
        "final_latency_p50": p50("final_latency"),
        "time_to_first_partial_p50": p50("time_to_first_partial"),
    }

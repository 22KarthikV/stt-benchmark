"""AssemblyAI adapter (raw HTTP/WebSocket, no SDK).

Endpoints and message shapes follow AssemblyAI's docs as read on 2026-11-10:
* file: POST /v2/upload (raw bytes) -> upload_url; POST /v2/transcript {audio_url, speech_models};
  poll GET /v2/transcript/{id} until status is completed|error. Auth header: ``authorization``.
* stream: wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&speech_model=<m>; header
  ``Authorization``; binary PCM16 frames; server sends Begin / Turn / Termination; client sends
  {"type": "Terminate"} when done. Streaming uses the singular ``speech_model`` field.

Model names change often (the docs show different ones on different pages), so both are
configurable (ASSEMBLYAI_MODEL, ASSEMBLYAI_STREAM_MODEL) and recorded in results.
UNTESTED AGAINST THE LIVE API until a real run is recorded in results.json.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import websockets

from ..audio import read_wav
from ._http import guard_ws, http_call, raise_for_status
from .base import (
    PermanentProviderError,
    Provider,
    StreamEvent,
    Transcript,
    TransientProviderError,
)

API = "https://api.assemblyai.com"
WS_URL = "wss://streaming.assemblyai.com/v3/ws"
DEFAULT_MODEL = "universal-3-5-pro"
DEFAULT_STREAM_MODEL = "universal-streaming-english"


class AssemblyAIProvider(Provider):
    name = "assemblyai"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        stream_model: str = DEFAULT_STREAM_MODEL,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        connect: Callable[..., Any] = websockets.connect,
        sleep: Callable[[float], Any] = asyncio.sleep,
        poll_interval: float = 1.0,
        timeout_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.api_key, self.model, self.stream_model = api_key, model, stream_model
        self._transport, self._connect = transport, connect
        self._sleep, self._poll_interval = sleep, poll_interval
        self._timeout_s, self._clock = timeout_s, clock

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "AssemblyAIProvider":
        return cls(
            env["ASSEMBLYAI_API_KEY"],
            model=env.get("ASSEMBLYAI_MODEL") or DEFAULT_MODEL,
            stream_model=env.get("ASSEMBLYAI_STREAM_MODEL") or DEFAULT_STREAM_MODEL,
        )

    async def transcribe_file(self, path: Path) -> Transcript:
        duration = read_wav(Path(path)).duration_s
        headers = {"authorization": self.api_key}
        async with httpx.AsyncClient(transport=self._transport, timeout=60.0) as client:
            up = await http_call(
                lambda: client.post(
                    f"{API}/v2/upload", headers=headers, content=Path(path).read_bytes()
                )
            )
            raise_for_status(up)
            upload_url = up.json().get("upload_url")
            if not upload_url:
                raise PermanentProviderError("upload response had no upload_url")

            created = await http_call(
                lambda: client.post(
                    f"{API}/v2/transcript",
                    headers=headers,
                    json={"audio_url": upload_url, "speech_models": [self.model]},
                )
            )
            raise_for_status(created)
            tid = created.json().get("id")
            if not tid:
                raise PermanentProviderError("transcript response had no id")

            deadline = self._clock() + self._timeout_s
            while True:
                polled = await http_call(
                    lambda: client.get(f"{API}/v2/transcript/{tid}", headers=headers)
                )
                raise_for_status(polled)
                body = polled.json()
                status = body.get("status")
                if status == "completed":
                    return Transcript(body.get("text") or "", self.name, self.model, duration)
                if status == "error":
                    raise PermanentProviderError(f"transcription failed: {body.get('error')}")
                if self._clock() > deadline:
                    raise TransientProviderError(f"timed out waiting for transcript {tid}")
                await self._sleep(self._poll_interval)

    def stream(self, pcm_chunks: AsyncIterator[bytes]) -> AsyncIterator[StreamEvent]:
        return guard_ws(self._stream(pcm_chunks))

    async def _stream(self, pcm_chunks: AsyncIterator[bytes]) -> AsyncIterator[StreamEvent]:
        url = f"{WS_URL}?{urlencode({'sample_rate': 16000, 'speech_model': self.stream_model})}"
        async with self._connect(url, additional_headers={"Authorization": self.api_key}) as ws:

            async def send_audio() -> None:
                async for chunk in pcm_chunks:
                    await ws.send(chunk)
                await ws.send(json.dumps({"type": "Terminate"}))

            sender = asyncio.create_task(send_audio())
            try:
                async for raw in ws:
                    event = parse_message(json.loads(raw), time.monotonic())
                    if event is not None:
                        yield event
                    if sender.done() and sender.exception():
                        raise sender.exception()
            finally:
                sender.cancel()


def parse_message(msg: dict[str, Any], t_recv: float) -> StreamEvent | None:
    """Turn messages -> StreamEvent; Begin/Termination/empty turns -> None."""
    if msg.get("type") != "Turn":
        return None
    text = msg.get("transcript") or ""
    if not text:
        return None
    words = msg.get("words") or []
    start = words[0].get("start") / 1000 if words and "start" in words[0] else None
    end = words[-1].get("end") / 1000 if words and "end" in words[-1] else None
    kind = "final" if msg.get("end_of_turn") else "partial"
    return StreamEvent(kind, text, t_recv, start, end)

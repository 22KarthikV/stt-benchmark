"""Deepgram adapter (raw HTTP/WebSocket, no SDK).

Per Deepgram's docs as read on 2026-11-10:
* file: POST https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true, header
  ``Authorization: Token <key>``, body = audio bytes (``Content-Type: audio/wav``); transcript at
  results.channels[0].alternatives[0].transcript.
* stream: wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&interim_results=true;
  raw audio needs encoding + sample_rate; client sends {"type": "CloseStream"} at the end.
  Results messages carry ``is_final`` (segment finalised) and ``speech_final`` (utterance ended),
  plus ``start`` and ``duration`` in seconds.

Mapping to our events: interim results -> "partial"; finalised segments are accumulated and one
"final" is emitted when ``speech_final`` arrives (or the socket closes with text pending).
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
from .base import PermanentProviderError, Provider, StreamEvent, Transcript

HTTP_URL = "https://api.deepgram.com/v1/listen"
WS_URL = "wss://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-3"


def parse_file_response(body: dict[str, Any]) -> str:
    try:
        return body["results"]["channels"][0]["alternatives"][0]["transcript"]
    except (KeyError, IndexError, TypeError) as e:
        raise PermanentProviderError(f"unexpected Deepgram response shape: {e!r}") from e


class StreamParser:
    """Stateful mapping from Deepgram Results messages to partial/final events."""

    def __init__(self) -> None:
        self._text: list[str] = []
        self._start: float | None = None
        self._end: float | None = None

    def feed(self, msg: dict[str, Any], t_recv: float) -> StreamEvent | None:
        if msg.get("type") != "Results":
            return None
        try:
            text = msg["channel"]["alternatives"][0]["transcript"]
        except (KeyError, IndexError, TypeError):
            return None
        start = msg.get("start")
        end = start + msg["duration"] if start is not None and "duration" in msg else None
        if msg.get("is_final"):
            if text:
                self._text.append(text)
                if self._start is None:
                    self._start = start
                self._end = end
            if msg.get("speech_final") and self._text:
                return self.flush(t_recv)
            return None
        if not text:
            return None
        shown = " ".join([*self._text, text])
        return StreamEvent(
            "partial", shown, t_recv, self._start if self._start is not None else start, end
        )

    def flush(self, t_recv: float) -> StreamEvent | None:
        if not self._text:
            return None
        ev = StreamEvent("final", " ".join(self._text), t_recv, self._start, self._end)
        self._text, self._start, self._end = [], None, None
        return ev


class DeepgramProvider(Provider):
    name = "deepgram"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        connect: Callable[..., Any] = websockets.connect,
    ):
        self.api_key, self.model = api_key, model
        self._transport, self._connect = transport, connect

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "DeepgramProvider":
        return cls(env["DEEPGRAM_API_KEY"], model=env.get("DEEPGRAM_MODEL") or DEFAULT_MODEL)

    async def transcribe_file(self, path: Path) -> Transcript:
        duration = read_wav(Path(path)).duration_s
        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"}
        params = {"model": self.model, "smart_format": "true"}
        async with httpx.AsyncClient(transport=self._transport, timeout=120.0) as client:
            resp = await http_call(
                lambda: client.post(
                    HTTP_URL, params=params, headers=headers, content=Path(path).read_bytes()
                )
            )
        raise_for_status(resp)
        return Transcript(parse_file_response(resp.json()), self.name, self.model, duration)

    def stream(self, pcm_chunks: AsyncIterator[bytes]) -> AsyncIterator[StreamEvent]:
        return guard_ws(self._stream(pcm_chunks))

    async def _stream(self, pcm_chunks: AsyncIterator[bytes]) -> AsyncIterator[StreamEvent]:
        query = urlencode(
            {
                "model": self.model,
                "encoding": "linear16",
                "sample_rate": 16000,
                "interim_results": "true",
            }
        )
        headers = {"Authorization": f"Token {self.api_key}"}
        parser = StreamParser()
        async with self._connect(f"{WS_URL}?{query}", additional_headers=headers) as ws:

            async def send_audio() -> None:
                async for chunk in pcm_chunks:
                    await ws.send(chunk)
                await ws.send(json.dumps({"type": "CloseStream"}))

            sender = asyncio.create_task(send_audio())
            try:
                async for raw in ws:
                    ev = parser.feed(json.loads(raw), time.monotonic())
                    if ev is not None:
                        yield ev
                    if sender.done() and sender.exception():
                        raise sender.exception()
                tail = parser.flush(time.monotonic())
                if tail is not None:
                    yield tail
            finally:
                sender.cancel()

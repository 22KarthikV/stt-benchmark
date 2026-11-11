"""Vosk (Kaldi, local, Apache 2.0) provider. Optional extra: ``pip install -e ".[vosk]"``.

Model: download and unzip e.g. vosk-model-small-en-us-0.15 (Apache 2.0, ~40 MB, from
https://alphacephei.com/vosk/models; checked 2026-11-10) and point VOSK_MODEL_PATH at the folder.
Needs 16 kHz mono 16-bit PCM. ``data/fetch_models.py`` can download it.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any

from ..audio import AudioError, read_pcm, read_wav, require_stream_format
from .base import PermanentProviderError, Provider, StreamEvent, Transcript

SAMPLE_RATE = 16000


class VoskProvider(Provider):
    name = "vosk"

    def __init__(self, model_path: str | Path, recognizer_factory: Callable[[], Any] | None = None):
        self.model_path = Path(model_path)
        self.model = f"vosk:{self.model_path.name}"
        self._factory = recognizer_factory
        self._model: Any = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "VoskProvider":
        return cls(env["VOSK_MODEL_PATH"])

    def _recognizer(self) -> Any:
        if self._factory is not None:
            return self._factory()
        try:
            import vosk
        except ImportError as e:
            raise PermanentProviderError('vosk not installed: pip install -e ".[vosk]"') from e
        if self._model is None:
            if not self.model_path.is_dir():
                raise PermanentProviderError(f"Vosk model folder not found: {self.model_path}")
            vosk.SetLogLevel(-1)
            self._model = vosk.Model(str(self.model_path))
        rec = vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
        rec.SetWords(True)
        return rec

    def _decode(self, pcm: bytes) -> str:
        rec = self._recognizer()
        chunk = 4000 * 2
        for i in range(0, len(pcm), chunk):
            rec.AcceptWaveform(pcm[i : i + chunk])
        return json.loads(rec.FinalResult()).get("text", "")

    async def transcribe_file(self, path: Path) -> Transcript:
        try:
            info = read_wav(Path(path))
            require_stream_format(info)
            pcm = read_pcm(Path(path))
        except AudioError as e:
            raise PermanentProviderError(str(e)) from e
        text = await asyncio.to_thread(self._decode, pcm)
        return Transcript(text, self.name, self.model, info.duration_s)

    async def stream(self, pcm_chunks: AsyncIterator[bytes]) -> AsyncIterator[StreamEvent]:
        rec = self._recognizer()
        utt_start: float | None = None
        async for chunk in pcm_chunks:
            if rec.AcceptWaveform(chunk):
                res = json.loads(rec.Result())
                ev = _final_event(res)
                if ev is not None:
                    yield ev
            else:
                text = json.loads(rec.PartialResult()).get("partial", "")
                if text:
                    yield StreamEvent("partial", text, time.monotonic(), utt_start, None)
            await asyncio.sleep(0)
        tail = _final_event(json.loads(rec.FinalResult()))
        if tail is not None:
            yield tail


def _final_event(res: dict[str, Any]) -> StreamEvent | None:
    text = res.get("text", "")
    if not text:
        return None
    words = res.get("result") or []
    start = words[0].get("start") if words else None
    end = words[-1].get("end") if words else None
    return StreamEvent("final", text, time.monotonic(), start, end)

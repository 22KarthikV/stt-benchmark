"""Vosk adapter logic with a scripted fake recognizer (no model, no vosk install needed)."""

import json

import pytest

from sttbench.providers.base import PermanentProviderError
from sttbench.providers.registry import build_providers
from sttbench.providers.vosk_provider import VoskProvider
from tests.conftest import make_wav


class FakeRec:
    def __init__(self):
        self.n = 0

    def AcceptWaveform(self, data):
        self.n += 1
        return self.n % 3 == 0

    def PartialResult(self):
        return json.dumps({"partial": f"part {self.n}"})

    def Result(self):
        return json.dumps({"text": "hello world", "result": [{"start": 0.1, "end": 0.9}]})

    def FinalResult(self):
        return json.dumps({"text": "the end"})


async def test_file_transcription(tmp_path):
    clip = make_wav(tmp_path / "x.wav", 1)
    p = VoskProvider("models/vosk-model-small-en-us-0.15", recognizer_factory=FakeRec)
    t = await p.transcribe_file(clip)
    assert (t.text, t.provider, t.model) == ("the end", "vosk", "vosk:vosk-model-small-en-us-0.15")


async def test_wrong_format_is_permanent(tmp_path):
    clip = make_wav(tmp_path / "x.wav", 1, sample_rate=8000)
    with pytest.raises(PermanentProviderError):
        await VoskProvider("m", recognizer_factory=FakeRec).transcribe_file(clip)


async def test_stream_partials_finals_and_tail():
    async def chunks():
        for _ in range(4):
            yield b"\x00" * 3200

    events = [e async for e in VoskProvider("m", recognizer_factory=FakeRec).stream(chunks())]
    assert [(e.kind, e.text) for e in events] == [
        ("partial", "part 1"),
        ("partial", "part 2"),
        ("final", "hello world"),
        ("partial", "part 4"),
        ("final", "the end"),
    ]
    assert events[2].audio_start_s == 0.1 and events[2].audio_end_s == 0.9


def test_registry_needs_model_path():
    providers, skipped = build_providers(["vosk"], env={})
    assert providers == {} and "VOSK_MODEL_PATH" in skipped["vosk"]
    providers, _ = build_providers(["vosk"], env={"VOSK_MODEL_PATH": "x/model"})
    assert providers["vosk"].model == "vosk:model"


async def test_missing_model_folder_is_permanent(tmp_path):
    pytest.importorskip("vosk")
    with pytest.raises(PermanentProviderError, match="not found"):
        await VoskProvider(tmp_path / "nope").transcribe_file(make_wav(tmp_path / "x.wav", 1))

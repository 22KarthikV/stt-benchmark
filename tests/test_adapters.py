"""Adapter tests: request construction + response parsing against shapes written from the docs.

These use mock transports / fake websockets. They prove parsing and request logic only,
NOT correctness against the live APIs.
"""

import json

import httpx
import pytest

from sttbench.providers.assemblyai import AssemblyAIProvider
from sttbench.providers.assemblyai import parse_message as aai_parse
from sttbench.providers.base import PermanentProviderError, StreamEvent, TransientProviderError
from sttbench.providers.deepgram import DeepgramProvider, StreamParser
from sttbench.providers.registry import build_providers
from tests.conftest import make_wav


async def no_sleep(_):
    return None


# ---------- AssemblyAI file ----------


def aai_handler(statuses, seen):
    polls = iter(statuses)

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path == "/v2/upload":
            return httpx.Response(200, json={"upload_url": "https://cdn/x"})
        if req.url.path == "/v2/transcript" and req.method == "POST":
            return httpx.Response(200, json={"id": "t1", "status": "queued"})
        if req.url.path == "/v2/transcript/t1":
            return httpx.Response(200, json=next(polls))
        return httpx.Response(404)

    return handler


async def test_assemblyai_file_flow(tmp_path):
    clip = make_wav(tmp_path / "x.wav", 1)
    seen = []
    p = AssemblyAIProvider(
        "KEY",
        model="m1",
        transport=httpx.MockTransport(
            aai_handler([{"status": "processing"}, {"status": "completed", "text": "Hello."}], seen)
        ),
        sleep=no_sleep,
    )
    t = await p.transcribe_file(clip)
    assert (t.text, t.provider, t.model) == ("Hello.", "assemblyai", "m1")
    assert t.duration_s == pytest.approx(1.0)
    assert all(r.headers["authorization"] == "KEY" for r in seen)
    create = next(r for r in seen if r.url.path == "/v2/transcript" and r.method == "POST")
    assert json.loads(create.content) == {"audio_url": "https://cdn/x", "speech_models": ["m1"]}
    upload = next(r for r in seen if r.url.path == "/v2/upload")
    assert upload.content == clip.read_bytes()


async def test_assemblyai_transcript_error_is_permanent(tmp_path):
    clip = make_wav(tmp_path / "x.wav", 1)
    p = AssemblyAIProvider(
        "K",
        transport=httpx.MockTransport(aai_handler([{"status": "error", "error": "bad audio"}], [])),
        sleep=no_sleep,
    )
    with pytest.raises(PermanentProviderError, match="bad audio"):
        await p.transcribe_file(clip)


@pytest.mark.parametrize(
    "code,exc",
    [(429, TransientProviderError), (503, TransientProviderError), (401, PermanentProviderError)],
)
async def test_assemblyai_http_error_classification(tmp_path, code, exc):
    clip = make_wav(tmp_path / "x.wav", 1)
    p = AssemblyAIProvider(
        "K", transport=httpx.MockTransport(lambda r: httpx.Response(code, text="x")), sleep=no_sleep
    )
    with pytest.raises(exc):
        await p.transcribe_file(clip)


async def test_assemblyai_network_error_is_transient(tmp_path):
    clip = make_wav(tmp_path / "x.wav", 1)

    def boom(req):
        raise httpx.ConnectTimeout("t")

    p = AssemblyAIProvider("K", transport=httpx.MockTransport(boom), sleep=no_sleep)
    with pytest.raises(TransientProviderError):
        await p.transcribe_file(clip)


async def test_assemblyai_poll_timeout(tmp_path):
    clip = make_wav(tmp_path / "x.wav", 1)
    ticks = iter(range(0, 1000, 100))
    p = AssemblyAIProvider(
        "K",
        transport=httpx.MockTransport(aai_handler([{"status": "processing"}] * 50, [])),
        sleep=no_sleep,
        timeout_s=250,
        clock=lambda: next(ticks),
    )
    with pytest.raises(TransientProviderError, match="timed out"):
        await p.transcribe_file(clip)


# ---------- AssemblyAI stream ----------


class FakeWS:
    """Stands in for a websockets connection: records sends, replays scripted messages."""

    def __init__(self, incoming):
        self.incoming, self.sent = incoming, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        async def gen():
            import asyncio

            while len(self.sent) < 2:  # wait until some audio + terminate went out
                await asyncio.sleep(0)
            for m in self.incoming:
                yield json.dumps(m)

        return gen()


def connector(ws, record):
    def connect(url, additional_headers=None, **kw):
        record.update(url=url, headers=additional_headers)
        return ws

    return connect


async def chunks(n=2):
    for _ in range(n):
        yield b"\x00" * 3200


async def test_assemblyai_stream_protocol():
    ws, rec = (
        FakeWS(
            [
                {"type": "Begin", "id": "s"},
                {"type": "Turn", "transcript": "hel", "end_of_turn": False},
                {
                    "type": "Turn",
                    "transcript": "hello there",
                    "end_of_turn": True,
                    "words": [{"start": 100, "end": 900}, {"start": 900, "end": 1500}],
                },
                {"type": "Termination", "audio_duration_seconds": 0.2},
            ]
        ),
        {},
    )
    p = AssemblyAIProvider("KEY", stream_model="sm", connect=connector(ws, rec))
    events = [e async for e in p.stream(chunks())]
    assert [(e.kind, e.text) for e in events] == [("partial", "hel"), ("final", "hello there")]
    assert events[1].audio_start_s == pytest.approx(0.1)
    assert events[1].audio_end_s == pytest.approx(1.5)
    assert "sample_rate=16000" in rec["url"] and "speech_model=sm" in rec["url"]
    assert rec["headers"] == {"Authorization": "KEY"}
    assert ws.sent[:2] == [b"\x00" * 3200] * 2
    assert json.loads(ws.sent[-1]) == {"type": "Terminate"}


def test_assemblyai_parse_ignores_non_turn_and_empty():
    assert aai_parse({"type": "Begin"}, 0.0) is None
    assert aai_parse({"type": "Turn", "transcript": "", "end_of_turn": False}, 0.0) is None


# ---------- Deepgram ----------


async def test_deepgram_file_request_and_parse(tmp_path):
    clip = make_wav(tmp_path / "x.wav", 1)
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(
            200, json={"results": {"channels": [{"alternatives": [{"transcript": "Hi there."}]}]}}
        )

    p = DeepgramProvider("KEY", model="nova-x", transport=httpx.MockTransport(handler))
    t = await p.transcribe_file(clip)
    assert (t.text, t.model) == ("Hi there.", "nova-x")
    r = seen[0]
    assert r.headers["authorization"] == "Token KEY" and r.headers["content-type"] == "audio/wav"
    assert r.url.params["model"] == "nova-x" and r.url.params["smart_format"] == "true"
    assert r.content == clip.read_bytes()


async def test_deepgram_bad_shape_is_permanent(tmp_path):
    clip = make_wav(tmp_path / "x.wav", 1)
    p = DeepgramProvider("K", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with pytest.raises(PermanentProviderError):
        await p.transcribe_file(clip)


@pytest.mark.parametrize(
    "code,exc",
    [
        (429, TransientProviderError),
        (500, TransientProviderError),
        (401, PermanentProviderError),
        (400, PermanentProviderError),
    ],
)
async def test_deepgram_http_errors(tmp_path, code, exc):
    clip = make_wav(tmp_path / "x.wav", 1)
    p = DeepgramProvider("K", transport=httpx.MockTransport(lambda r: httpx.Response(code)))
    with pytest.raises(exc):
        await p.transcribe_file(clip)


def result(text, final, speech_final=False, start=0.0, dur=1.0):
    return {
        "type": "Results",
        "is_final": final,
        "speech_final": speech_final,
        "start": start,
        "duration": dur,
        "channel": {"alternatives": [{"transcript": text}]},
    }


def test_deepgram_parser_accumulates_until_speech_final():
    sp = StreamParser()
    e1 = sp.feed(result("hel", False), 1.0)
    assert (e1.kind, e1.text) == ("partial", "hel")
    assert sp.feed(result("hello", True, start=0.0, dur=1.0), 2.0) is None
    e3 = sp.feed(result("wor", False, start=1.0), 3.0)
    assert (e3.kind, e3.text) == ("partial", "hello wor")
    fin = sp.feed(result("world", True, True, start=1.0, dur=0.5), 4.0)
    assert (fin.kind, fin.text) == ("final", "hello world")
    assert fin.audio_start_s == 0.0 and fin.audio_end_s == pytest.approx(1.5)
    assert sp.flush(5.0) is None


def test_deepgram_parser_flush_and_ignores():
    sp = StreamParser()
    assert sp.feed({"type": "Metadata"}, 0.0) is None
    assert sp.feed(result("", False), 0.0) is None
    sp.feed(result("tail", True), 1.0)
    assert sp.flush(2.0) == StreamEvent("final", "tail", 2.0, 0.0, 1.0)


async def test_deepgram_stream_protocol():
    ws, rec = (
        FakeWS(
            [{"type": "Metadata"}, result("hi", False), result("hi there", True, True, 0.0, 0.2)]
        ),
        {},
    )
    p = DeepgramProvider("KEY", model="nova-3", connect=connector(ws, rec))
    events = [e async for e in p.stream(chunks())]
    assert [(e.kind, e.text) for e in events] == [("partial", "hi"), ("final", "hi there")]
    for needle in (
        "encoding=linear16",
        "sample_rate=16000",
        "interim_results=true",
        "model=nova-3",
    ):
        assert needle in rec["url"]
    assert rec["headers"] == {"Authorization": "Token KEY"}
    assert json.loads(ws.sent[-1]) == {"type": "CloseStream"}


# ---------- registry ----------


def test_registry_builds_cloud_adapters_when_keys_present():
    providers, skipped = build_providers(
        ["assemblyai", "deepgram"],
        env={"ASSEMBLYAI_API_KEY": "a", "DEEPGRAM_API_KEY": "d", "DEEPGRAM_MODEL": "nova-x"},
    )
    assert skipped == {} and providers["deepgram"].model == "nova-x"
    assert providers["assemblyai"].api_key == "a"

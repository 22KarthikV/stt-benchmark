import pytest

from sttbench.providers.base import (
    PermanentProviderError,
    ProviderError,
    StreamEvent,
    Transcript,
    TransientProviderError,
)
from sttbench.providers.fake import FakeProvider
from sttbench.providers.registry import build_providers
from tests.conftest import make_wav


def test_error_hierarchy():
    assert issubclass(TransientProviderError, ProviderError)
    assert issubclass(PermanentProviderError, ProviderError)
    assert not issubclass(PermanentProviderError, TransientProviderError)


async def test_fake_transcribe_file(tmp_path):
    p = make_wav(tmp_path / "x.wav", 1)
    t = await FakeProvider(transcripts={"x": "hello world"}).transcribe_file(p)
    assert isinstance(t, Transcript)
    assert (t.text, t.provider) == ("hello world", "fake")
    assert t.model.startswith("fake")


async def test_fake_fails_transiently_then_succeeds(tmp_path):
    p = make_wav(tmp_path / "x.wav", 1)
    fake = FakeProvider(transcripts={"x": "hi"}, fail_times={"x": 2})
    for _ in range(2):
        with pytest.raises(TransientProviderError):
            await fake.transcribe_file(p)
    assert (await fake.transcribe_file(p)).text == "hi"


async def test_fake_permanent_error(tmp_path):
    p = make_wav(tmp_path / "x.wav", 1)
    fake = FakeProvider(errors={"x": PermanentProviderError("401")})
    with pytest.raises(PermanentProviderError):
        await fake.transcribe_file(p)


async def chunk_source(n, size=3200):
    for _ in range(n):
        yield b"\x00" * size


async def test_fake_stream_emits_partials_then_final():
    fake = FakeProvider(utterances=["one two three"], chunks_per_utterance=6)
    events = [e async for e in fake.stream(chunk_source(6))]
    assert all(isinstance(e, StreamEvent) for e in events)
    kinds = [e.kind for e in events]
    assert kinds.count("final") == 1 and kinds[-1] == "final"
    assert "partial" in kinds
    assert events[-1].text == "one two three"
    assert events[-1].audio_end_s == pytest.approx(0.6)
    assert [e.t_recv for e in events] == sorted(e.t_recv for e in events)


async def test_fake_stream_multiple_utterances_and_remainder():
    fake = FakeProvider(utterances=["a b", "c d"], chunks_per_utterance=4)
    events = [e async for e in fake.stream(chunk_source(10))]
    finals = [e for e in events if e.kind == "final"]
    assert [f.text for f in finals] == ["a b", "c d", "a b"]
    assert finals[0].audio_end_s == pytest.approx(0.4)
    assert finals[1].audio_start_s == pytest.approx(0.4)


async def test_fake_stream_empty_input():
    assert [e async for e in FakeProvider().stream(chunk_source(0))] == []


def test_registry_builds_fake():
    providers, skipped = build_providers(["fake"], env={})
    assert list(providers) == ["fake"] and skipped == {}


@pytest.mark.parametrize(
    "name,var", [("assemblyai", "ASSEMBLYAI_API_KEY"), ("deepgram", "DEEPGRAM_API_KEY")]
)
def test_registry_missing_key_is_skipped_not_scored(name, var):
    providers, skipped = build_providers([name], env={})
    assert providers == {}
    assert var in skipped[name] and "no key" in skipped[name]


def test_registry_unknown_provider():
    with pytest.raises(ValueError, match="nope"):
        build_providers(["nope"], env={})

import pytest

from sttbench.audio import (
    AudioError,
    iter_pcm_chunks,
    read_pcm,
    read_wav,
    require_stream_format,
    write_wav,
)
from tests.conftest import make_wav


def test_read_wav_info(tmp_path):
    p = make_wav(tmp_path / "x.wav", 2.5)
    info = read_wav(p)
    assert (info.sample_rate, info.channels, info.sample_width) == (16000, 1, 2)
    assert info.duration_s == pytest.approx(2.5)


def test_read_wav_unreadable(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"not a wav at all")
    with pytest.raises(AudioError):
        read_wav(p)


def test_write_then_read_roundtrip(tmp_path):
    p = tmp_path / "r.wav"
    write_wav(p, b"\x01\x00" * 1600)
    assert read_pcm(p) == b"\x01\x00" * 1600


def test_stream_format_accepts_16k_mono(tmp_path):
    require_stream_format(read_wav(make_wav(tmp_path / "ok.wav", 1)))


@pytest.mark.parametrize("kw", [{"sample_rate": 8000}, {"channels": 2}])
def test_stream_format_rejects_others(tmp_path, kw):
    with pytest.raises(AudioError):
        require_stream_format(read_wav(make_wav(tmp_path / "no.wav", 1, **kw)))


def test_chunks_exact_multiple():
    chunks = list(iter_pcm_chunks(b"\x00" * 32000))
    assert len(chunks) == 10
    assert all(len(c) == 3200 for c in chunks)


def test_chunks_remainder_is_shorter():
    chunks = list(iter_pcm_chunks(b"\x00" * (32000 + 1000)))
    assert [len(c) for c in chunks] == [3200] * 10 + [1000]


def test_chunks_empty():
    assert list(iter_pcm_chunks(b"")) == []


def test_chunks_odd_byte_count_rejected():
    with pytest.raises(ValueError):
        list(iter_pcm_chunks(b"\x00" * 3201))


def test_chunks_other_chunk_ms():
    chunks = list(iter_pcm_chunks(b"\x00" * 1280, chunk_ms=20))
    assert [len(c) for c in chunks] == [640, 640]


def test_chunks_content_preserved():
    pcm = bytes(range(256)) * 50
    assert b"".join(iter_pcm_chunks(pcm)) == pcm

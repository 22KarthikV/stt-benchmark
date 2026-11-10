"""WAV reading/validation and PCM chunking."""

import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

STREAM_SAMPLE_RATE = 16000
STREAM_CHANNELS = 1
STREAM_SAMPLE_WIDTH = 2


class AudioError(Exception):
    pass


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    sample_width: int
    n_frames: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.sample_rate


def read_wav(path: Path) -> WavInfo:
    try:
        with wave.open(str(path), "rb") as w:
            return WavInfo(w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes())
    except (wave.Error, EOFError, OSError) as e:
        raise AudioError(f"unreadable WAV {path}: {e}") from e


def read_pcm(path: Path) -> bytes:
    try:
        with wave.open(str(path), "rb") as w:
            return w.readframes(w.getnframes())
    except (wave.Error, EOFError, OSError) as e:
        raise AudioError(f"unreadable WAV {path}: {e}") from e


def write_wav(path: Path, pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def require_stream_format(info: WavInfo) -> None:
    """The streaming demo needs 16 kHz mono 16-bit PCM; no resampling is done."""
    got = (info.sample_rate, info.channels, info.sample_width)
    want = (STREAM_SAMPLE_RATE, STREAM_CHANNELS, STREAM_SAMPLE_WIDTH)
    if got != want:
        raise AudioError(
            f"need 16 kHz mono 16-bit PCM, got {info.sample_rate} Hz, "
            f"{info.channels} ch, {info.sample_width * 8}-bit"
        )


def iter_pcm_chunks(
    pcm: bytes, chunk_ms: int = 100, sample_rate: int = STREAM_SAMPLE_RATE, sample_width: int = 2
) -> Iterator[bytes]:
    """Yield fixed-size chunks (default 100 ms = 3200 bytes); the last may be shorter."""
    if len(pcm) % sample_width:
        raise ValueError(f"PCM length {len(pcm)} is not a multiple of sample width {sample_width}")
    size = sample_rate * chunk_ms // 1000 * sample_width
    for i in range(0, len(pcm), size):
        yield pcm[i : i + size]

import wave
from pathlib import Path

import pytest


def make_wav(path: Path, seconds: float, sample_rate: int = 16000, channels: int = 1) -> Path:
    """Write a silent synthetic WAV. Test fixture only, never real data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n * channels)
    return path


def words(n: int) -> str:
    vocab = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()
    return " ".join(vocab[i % len(vocab)] for i in range(n))


@pytest.fixture
def dataset(tmp_path):
    """A valid tiny dataset: 3 ten-second clips, 25 words each (2.5 wps)."""
    clips, truth = tmp_path / "clips", tmp_path / "truth"
    truth.mkdir()
    for stem in ("a", "b", "c"):
        make_wav(clips / f"{stem}.wav", 10)
        (truth / f"{stem}.txt").write_text(words(25), encoding="utf-8")
    return clips, truth

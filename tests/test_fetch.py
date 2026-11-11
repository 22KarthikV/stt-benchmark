"""Tests for data/fetch_librispeech.py using a tiny synthetic archive (no network)."""

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "fetch_librispeech", Path(__file__).parent.parent / "data" / "fetch_librispeech.py"
)
fl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fl)

_vspec = importlib.util.spec_from_file_location(
    "fetch_vosk_model", Path(__file__).parent.parent / "data" / "fetch_vosk_model.py"
)
fv = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(fv)


def test_select_subset_is_deterministic_and_order_independent():
    ids = [f"1-2-{i}" for i in range(100)]
    a = fl.select_subset(ids, 10, seed=1)
    assert a == fl.select_subset(list(reversed(ids)), 10, seed=1)
    assert a != fl.select_subset(ids, 10, seed=2)
    assert len(set(a)) == 10


def test_select_subset_too_many():
    with pytest.raises(ValueError):
        fl.select_subset(["1-2-3"], 2, seed=0)


def test_parse_transcripts_skips_junk():
    text = "1-2-0000 HELLO WORLD\nbad line\n1-2-0001 ANOTHER ONE\n../x HI\n"
    assert fl.parse_transcripts(text) == {"1-2-0000": "HELLO WORLD", "1-2-0001": "ANOTHER ONE"}


def test_md5_of(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert fl.md5_of(p) == "900150983cd24fb0d6963f7d28e17f72"


def test_extract_from_synthetic_archive(tmp_path):
    sf = pytest.importorskip("soundfile")
    import numpy as np

    arc = tmp_path / "a.tar.gz"
    with tarfile.open(arc, "w:gz") as tar:

        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        trans = ""
        for i in range(3):
            buf = io.BytesIO()
            sf.write(buf, np.zeros(16000, dtype="int16"), 16000, format="FLAC", subtype="PCM_16")
            add(f"LibriSpeech/test-clean/1/2/1-2-000{i}.flac", buf.getvalue())
            trans += f"1-2-000{i} WORD NUMBER {i}\n"
        add("LibriSpeech/test-clean/1/2/1-2.trans.txt", trans.encode())

    transcripts = fl.read_transcripts(arc)
    assert len(transcripts) == 3
    out = tmp_path / "out"
    fl.extract(arc, ["1-2-0000", "1-2-0002"], transcripts, out)
    assert sorted(p.name for p in (out / "clips").iterdir()) == ["1-2-0000.wav", "1-2-0002.wav"]
    assert (out / "truth" / "1-2-0002.txt").read_text().strip() == "WORD NUMBER 2"


def test_vosk_zip_slip_rejected(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with zipfile.ZipFile(z) as zf, pytest.raises(RuntimeError):
        fv.safe_extract(zf, tmp_path / "dest")

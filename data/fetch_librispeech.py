"""Fetch a deterministic subset of LibriSpeech test-clean (CC BY 4.0, OpenSLR SLR12).

Usage:  python data/fetch_librispeech.py [--n 30] [--seed 1234] [--archive PATH]

Downloads test-clean.tar.gz (~346 MB) into data/raw/ (git-ignored), verifies its MD5 against the
value published at https://www.openslr.org/resources/12/md5sum.txt, picks N utterances with a
seeded RNG, and writes data/clips/<id>.wav (16 kHz mono 16-bit) + data/truth/<id>.txt.
Needs the ``data`` extra (soundfile) for FLAC decoding.
"""

import argparse
import hashlib
import random
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
MD5 = "32fa31d27d2e1cad72775fee3f4849a9"
ID_RE = re.compile(r"^\d+-\d+-\d+$")
HERE = Path(__file__).resolve().parent


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_subset(ids: list[str], n: int, seed: int) -> list[str]:
    """Deterministic: same ids + n + seed -> same selection, independent of input order."""
    pool = sorted(set(ids))
    if n > len(pool):
        raise ValueError(f"asked for {n} clips but only {len(pool)} available")
    return sorted(random.Random(seed).sample(pool, n))


def parse_transcripts(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        utt_id, _, words = line.strip().partition(" ")
        if ID_RE.match(utt_id) and words:
            out[utt_id] = words
    return out


def read_transcripts(archive: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".trans.txt"):
                found.update(parse_transcripts(tar.extractfile(m).read().decode("utf-8")))
    return found


def extract(archive: Path, ids: list[str], transcripts: dict[str, str], out_dir: Path) -> None:
    """Write clips/truth for ``ids``. Never extracts tar members to disk by their own names."""
    import io

    import soundfile as sf

    wanted = set(ids)
    clips, truth = out_dir / "clips", out_dir / "truth"
    clips.mkdir(parents=True, exist_ok=True)
    truth.mkdir(parents=True, exist_ok=True)
    done = set()
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar:
            if not (m.isfile() and m.name.endswith(".flac")):
                continue
            utt_id = Path(m.name).stem
            if utt_id not in wanted:
                continue
            data, rate = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="int16")
            if rate != 16000 or data.ndim != 1:
                raise RuntimeError(f"{utt_id}: expected 16 kHz mono, got {rate} Hz {data.shape}")
            sf.write(str(clips / f"{utt_id}.wav"), data, rate, subtype="PCM_16")
            (truth / f"{utt_id}.txt").write_text(transcripts[utt_id] + "\n", encoding="utf-8")
            done.add(utt_id)
    missing = wanted - done
    if missing:
        raise RuntimeError(f"audio missing from archive for: {sorted(missing)}")


def download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} -> {dest}")
    urllib.request.urlretrieve(URL, dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--archive", type=Path, default=HERE / "raw" / "test-clean.tar.gz")
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args(argv)
    if not args.archive.exists():
        download(args.archive)
    digest = md5_of(args.archive)
    if digest != MD5:
        print(f"MD5 mismatch: got {digest}, expected {MD5}", file=sys.stderr)
        return 1
    transcripts = read_transcripts(args.archive)
    ids = select_subset(list(transcripts), args.n, args.seed)
    extract(args.archive, ids, transcripts, args.out)
    print(f"wrote {len(ids)} clips to {args.out / 'clips'} (seed={args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Download the small English Vosk model (Apache 2.0, ~40 MB) into data/models/.

Source: https://alphacephei.com/vosk/models (license and size checked 2026-11-10).
Then:  set VOSK_MODEL_PATH=data/models/vosk-model-small-en-us-0.15   (in .env)
"""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

NAME = "vosk-model-small-en-us-0.15"
URL = f"https://alphacephei.com/vosk/models/{NAME}.zip"
HERE = Path(__file__).resolve().parent


def safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for member in zf.namelist():
        if not (dest / member).resolve().is_relative_to(root):
            raise RuntimeError(f"unsafe path in archive: {member}")
    zf.extractall(dest)


def main() -> int:
    models = HERE / "models"
    models.mkdir(exist_ok=True)
    target = models / NAME
    if target.is_dir():
        print(f"already present: {target}")
        return 0
    archive = HERE / "raw" / f"{NAME}.zip"
    archive.parent.mkdir(exist_ok=True)
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, archive)
    print(f"sha256 {hashlib.sha256(archive.read_bytes()).hexdigest()} (no published checksum)")
    with zipfile.ZipFile(archive) as zf:
        safe_extract(zf, models)
    print(f"extracted to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

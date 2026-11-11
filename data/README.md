# Data

- `clips/` 30 WAV files (16 kHz mono 16-bit), **git-ignored**. Created by `fetch_librispeech.py`.
- `truth/` the matching transcripts, one `<utterance-id>.txt` each (LibriSpeech upper case; the
  normaliser lowercases).
- `models/`, `raw/` downloads, git-ignored.

Source: LibriSpeech `test-clean` from OpenSLR SLR12, CC BY 4.0. The archive MD5 is checked against
the value published at <https://www.openslr.org/resources/12/md5sum.txt>. Selection is
deterministic: `random.Random(seed).sample(sorted(ids), n)` with defaults n=30, seed=1234.

## Normalisation (`norm-v1`)

Same rules for reference and hypothesis: NFKC + lowercase; digits to words; punctuation removed
except intra-word apostrophes; hyphens to spaces; whitespace collapsed. See the root README for
what is deliberately not normalised.

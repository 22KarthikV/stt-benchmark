"""Truth-file validation. Runs before any provider call.

Honest limit: these are offline heuristics. They catch gross errors (wrong clip's text,
truncation, junk) but not a single wrong word, and not a same-length mislabel; only an ASR
sanity pass (``--sanity-provider``, optional, costs API calls) could catch the latter.
"""

import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .audio import AudioError, read_wav, require_stream_format
from .normalize import normalize

MISSING_TRUTH = "MISSING_TRUTH"
MISSING_CLIP = "MISSING_CLIP"
EMPTY_TRANSCRIPT = "EMPTY_TRANSCRIPT"
DURATION_MISMATCH = "DURATION_MISMATCH"
MISALIGNED = "MISALIGNED"
BAD_AUDIO = "BAD_AUDIO"

MIN_ALPHA_RATIO = 0.7
MAX_TOP_TOKEN_SHARE = 0.5
MIN_TOKENS_FOR_REPEAT_CHECK = 6
MAX_NON_LATIN_RATIO = 0.2


@dataclass(frozen=True)
class Issue:
    code: str
    path: Path
    message: str
    severity: str = "error"

    def __str__(self) -> str:
        return f"{self.severity.upper()} {self.code} {self.path}: {self.message}"


def has_errors(issues: Sequence[Issue]) -> bool:
    return any(i.severity == "error" for i in issues)


def _misaligned_reason(raw: str, normalised: str) -> str | None:
    letters = [c for c in raw if not c.isspace()]
    if letters:
        alpha = sum(c.isalpha() or c == "'" for c in letters) / len(letters)
        if alpha < MIN_ALPHA_RATIO:
            return f"only {alpha:.0%} of characters are letters"
        non_latin = sum(c.isalpha() and "LATIN" not in unicodedata.name(c, "") for c in letters)
        if non_latin / len(letters) > MAX_NON_LATIN_RATIO:
            return "text is mostly non-Latin script"
    tokens = normalised.split()
    if len(tokens) >= MIN_TOKENS_FOR_REPEAT_CHECK:
        token, count = Counter(tokens).most_common(1)[0]
        if count / len(tokens) > MAX_TOP_TOKEN_SHARE:
            return f"token '{token}' makes up {count / len(tokens):.0%} of the text"
    return None


def validate_dataset(
    clips_dir: Path,
    truth_dir: Path,
    *,
    strict: bool = False,
    min_wps: float = 0.5,
    max_wps: float = 6.0,
    stream_format: bool = False,
) -> list[Issue]:
    issues: list[Issue] = []
    clips = sorted(Path(clips_dir).glob("*.wav"))
    stems = {c.stem for c in clips}

    for clip in clips:
        truth_path = Path(truth_dir) / f"{clip.stem}.txt"
        if not truth_path.exists():
            issues.append(Issue(MISSING_TRUTH, truth_path, f"no truth file for clip {clip.name}"))
            continue
        try:
            info = read_wav(clip)
            if stream_format:
                require_stream_format(info)
        except AudioError as e:
            issues.append(Issue(BAD_AUDIO, clip, str(e)))
            info = None
        raw = truth_path.read_text(encoding="utf-8")
        norm = normalize(raw)
        if not norm:
            issues.append(Issue(EMPTY_TRANSCRIPT, truth_path, "empty after normalisation"))
            continue
        if info is not None and info.duration_s > 0:
            wps = len(norm.split()) / info.duration_s
            if not min_wps <= wps <= max_wps:
                issues.append(
                    Issue(
                        DURATION_MISMATCH,
                        truth_path,
                        f"{wps:.2f} words/s over {info.duration_s:.1f}s audio "
                        f"(plausible {min_wps}-{max_wps}); wrong or truncated text?",
                    )
                )
        reason = _misaligned_reason(raw, norm)
        if reason:
            issues.append(Issue(MISALIGNED, truth_path, reason))

    for truth_path in sorted(Path(truth_dir).glob("*.txt")):
        if truth_path.stem not in stems:
            issues.append(
                Issue(
                    MISSING_CLIP,
                    truth_path,
                    "truth file has no matching clip",
                    "error" if strict else "warning",
                )
            )
    return issues

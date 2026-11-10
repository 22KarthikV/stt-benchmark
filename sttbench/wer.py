"""Word error rate on normalised text (jiwer does the alignment)."""

from collections.abc import Iterable

import jiwer

from .normalize import normalize


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return (edits, reference_word_count) after normalising both sides."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    n_ref = len(ref.split())
    if n_ref == 0:
        raise ValueError("reference is empty after normalisation")
    if not hyp:
        return n_ref, n_ref
    out = jiwer.process_words(ref, hyp)
    return out.substitutions + out.deletions + out.insertions, n_ref


def wer(reference: str, hypothesis: str) -> float:
    edits, n = word_errors(reference, hypothesis)
    return edits / n


def corpus_wer(pairs: Iterable[tuple[str, str]]) -> float:
    """Total edits / total reference words over all (reference, hypothesis) pairs."""
    edits = words = 0
    for ref, hyp in pairs:
        e, n = word_errors(ref, hyp)
        edits += e
        words += n
    if words == 0:
        raise ValueError("no pairs to score")
    return edits / words

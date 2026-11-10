"""Text normalisation applied identically to reference and hypothesis.

Version ``norm-v1``:
1. Unicode NFKC, lowercase, curly apostrophes -> ASCII apostrophe.
2. Digits expanded to words (integers up to 999,999,999,999, simple decimals).
   No "and" ("101" -> "one hundred one"), no year reading ("2024" -> "two thousand twenty four").
3. Punctuation removed except intra-word apostrophes; hyphens become spaces.
4. Whitespace collapsed.

Not done: spelling variants, contractions, filler words, ordinals ("1st"), currency/percent symbols.
"""

import re
import unicodedata

NORM_VERSION = "norm-v1"

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = [(10**9, "billion"), (10**6, "million"), (10**3, "thousand")]

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _below_1000(n: int) -> list[str]:
    out: list[str] = []
    if n >= 100:
        out += [_ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        out.append(_TENS[n // 10])
        n %= 10
        if n:
            out.append(_ONES[n])
    elif n:
        out.append(_ONES[n])
    return out


def _int_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    if n >= 10**12:
        return " ".join(_ONES[int(d)] for d in str(n))  # too large: read digit by digit
    parts: list[str] = []
    for scale, name in _SCALES:
        if n >= scale:
            parts += [*_below_1000(n // scale), name]
            n %= scale
    if n:
        parts += _below_1000(n)
    return " ".join(parts)


def _expand_number(match: re.Match[str]) -> str:
    text = match.group(0).replace(",", "")
    whole, _, frac = text.partition(".")
    words = _int_to_words(int(whole))
    if frac:
        words += " point " + " ".join(_ONES[int(d)] for d in frac)
    return f" {words} "


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = _NUMBER.sub(_expand_number, text)
    text = text.replace("-", " ")
    # keep letters (any script), digits-already-expanded, apostrophes, whitespace
    text = "".join(c if (c.isalnum() or c == "'" or c.isspace()) else " " for c in text)
    tokens = [t.strip("'") for t in text.split()]
    return " ".join(t for t in tokens if t)

import pytest

from sttbench.normalize import NORM_VERSION, normalize


def test_version():
    assert NORM_VERSION == "norm-v1"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Hello, World!", "hello world"),
        ("DON'T stop", "don't stop"),
        ("don’t", "don't"),
        ("'quoted' word", "quoted word"),
        ("well-known fact", "well known fact"),
        ("  a \n b\t c ", "a b c"),
        ("", ""),
        ("...", ""),
        ("ＡＢＣ", "abc"),
        ("café", "café"),
        ("I have 3 apples", "i have three apples"),
        ("0", "zero"),
        ("21", "twenty one"),
        ("100", "one hundred"),
        ("101", "one hundred one"),
        ("1,234", "one thousand two hundred thirty four"),
        ("2024", "two thousand twenty four"),
        ("3.5", "three point five"),
        ("Mr. Smith, aged 40.", "mr smith aged forty"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_idempotent():
    s = "It's 21 well-known, DON'T you think?"
    assert normalize(normalize(s)) == normalize(s)

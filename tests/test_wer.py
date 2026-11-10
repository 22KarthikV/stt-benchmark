import pytest

from sttbench.wer import corpus_wer, wer, word_errors


def test_identical():
    assert wer("the cat sat", "the cat sat") == 0.0


def test_substitution():
    assert wer("a b c d", "a b x d") == 0.25


def test_insertion():
    assert wer("a b", "a b c") == 0.5


def test_deletion():
    assert wer("a b c d", "a b c") == 0.25


def test_empty_hypothesis_is_total_error():
    assert wer("a b c", "") == 1.0


def test_normalisation_applied_to_both():
    assert wer("Hello, World!", "hello world") == 0.0
    assert wer("I have three apples", "I have 3 apples") == 0.0


def test_empty_reference_rejected():
    with pytest.raises(ValueError):
        wer("", "anything")


def test_word_errors_counts():
    assert word_errors("a b c d", "a x c") == (2, 4)


def test_corpus_wer_is_weighted_by_reference_length():
    pairs = [("a b c d", "a b c d"), ("a", "b")]
    assert corpus_wer(pairs) == pytest.approx(1 / 5)


def test_corpus_wer_empty():
    with pytest.raises(ValueError):
        corpus_wer([])

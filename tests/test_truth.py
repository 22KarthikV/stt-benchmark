from sttbench.truth import has_errors, validate_dataset
from tests.conftest import make_wav, words


def codes(issues):
    return sorted(i.code for i in issues)


def test_valid_dataset_accepted(dataset):
    clips, truth = dataset
    assert validate_dataset(clips, truth) == []


def test_missing_truth(dataset):
    clips, truth = dataset
    (truth / "b.txt").unlink()
    issues = validate_dataset(clips, truth)
    assert codes(issues) == ["MISSING_TRUTH"]
    assert "b" in str(issues[0].path)
    assert has_errors(issues)


def test_orphan_truth_is_warning_unless_strict(dataset):
    clips, truth = dataset
    (truth / "orphan.txt").write_text("hello there", encoding="utf-8")
    lax = validate_dataset(clips, truth)
    assert codes(lax) == ["MISSING_CLIP"] and not has_errors(lax)
    assert has_errors(validate_dataset(clips, truth, strict=True))


def test_empty_transcript(dataset):
    clips, truth = dataset
    (truth / "a.txt").write_text("  \n", encoding="utf-8")
    (truth / "b.txt").write_text("...", encoding="utf-8")
    assert codes(validate_dataset(clips, truth)) == ["EMPTY_TRANSCRIPT"] * 2


def test_too_few_words_for_duration(dataset):
    clips, truth = dataset
    (truth / "a.txt").write_text("yes sir", encoding="utf-8")
    assert codes(validate_dataset(clips, truth)) == ["DURATION_MISMATCH"]


def test_too_many_words_for_duration(dataset):
    clips, truth = dataset
    make_wav(clips / "a.wav", 1)
    (truth / "a.txt").write_text(words(30), encoding="utf-8")
    assert codes(validate_dataset(clips, truth)) == ["DURATION_MISMATCH"]


def test_wps_bounds_configurable(dataset):
    clips, truth = dataset
    (truth / "a.txt").write_text("yes sir", encoding="utf-8")
    assert validate_dataset(clips, truth, min_wps=0.1) == []


def test_mislabelled_long_clip_short_text_rejected(tmp_path):
    clips, truth = tmp_path / "clips", tmp_path / "truth"
    truth.mkdir()
    make_wav(clips / "A.wav", 30)
    (truth / "A.txt").write_text("yes sir", encoding="utf-8")  # text of a short clip B
    issues = validate_dataset(clips, truth)
    assert has_errors(issues) and codes(issues) == ["DURATION_MISMATCH"]


def test_mislabelled_short_clip_long_text_rejected(tmp_path):
    clips, truth = tmp_path / "clips", tmp_path / "truth"
    truth.mkdir()
    make_wav(clips / "A.wav", 2)
    (truth / "A.txt").write_text(words(60), encoding="utf-8")
    assert has_errors(validate_dataset(clips, truth))


def test_same_length_mislabel_passes_offline_heuristic(dataset):
    """Documented limit: only --sanity-provider could catch this."""
    clips, truth = dataset
    (truth / "a.txt").write_text("zulu " * 24 + "yankee", encoding="utf-8")
    assert not any(i.code == "DURATION_MISMATCH" for i in validate_dataset(clips, truth))


def test_misaligned_non_alphabetic(dataset):
    clips, truth = dataset
    (truth / "a.txt").write_text("abc " + "123 " * 24, encoding="utf-8")
    assert "MISALIGNED" in codes(validate_dataset(clips, truth))


def test_misaligned_repeated_tokens(dataset):
    clips, truth = dataset
    (truth / "a.txt").write_text("the " * 25, encoding="utf-8")
    assert "MISALIGNED" in codes(validate_dataset(clips, truth))


def test_misaligned_wrong_script(dataset):
    clips, truth = dataset
    text = " ".join(["привет", "мир"] * 12 + ["x"])
    (truth / "a.txt").write_text(text, encoding="utf-8")
    assert "MISALIGNED" in codes(validate_dataset(clips, truth))


def test_bad_audio_unreadable(dataset):
    clips, truth = dataset
    (clips / "a.wav").write_bytes(b"garbage")
    assert codes(validate_dataset(clips, truth)) == ["BAD_AUDIO"]


def test_bad_audio_wrong_format_only_when_stream_format_required(tmp_path):
    clips, truth = tmp_path / "clips", tmp_path / "truth"
    truth.mkdir()
    make_wav(clips / "a.wav", 10, sample_rate=8000)
    (truth / "a.txt").write_text(words(25), encoding="utf-8")
    assert validate_dataset(clips, truth) == []
    assert codes(validate_dataset(clips, truth, stream_format=True)) == ["BAD_AUDIO"]


def test_all_problems_reported_together(dataset):
    clips, truth = dataset
    (truth / "a.txt").unlink()
    (truth / "b.txt").write_text("", encoding="utf-8")
    (truth / "c.txt").write_text("yes", encoding="utf-8")
    assert codes(validate_dataset(clips, truth)) == [
        "DURATION_MISMATCH",
        "EMPTY_TRANSCRIPT",
        "MISSING_TRUTH",
    ]


def test_issue_message_names_file(dataset):
    clips, truth = dataset
    (truth / "a.txt").write_text("yes", encoding="utf-8")
    issue = validate_dataset(clips, truth)[0]
    assert "a.txt" in str(issue.path) and issue.message

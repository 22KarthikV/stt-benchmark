import pytest

from sttbench.report import (
    README_END,
    README_START,
    percentile,
    render_markdown,
    summarize,
    update_readme,
)


def clip(name, wer, lat, ref_words=10, status="ok"):
    c = {"clip": name, "status": status, "reference": " ".join(["w"] * ref_words)}
    if status == "ok":
        c.update(wer=wer, edits=round(wer * ref_words), ref_words=ref_words, latency_s=lat)
    else:
        c["error"] = "boom"
    return c


RESULTS = {
    "meta": {
        "run_date": "2026-11-10T12:00:00Z",
        "dataset": "LibriSpeech test-clean",
        "n_clips": 5,
        "norm_version": "norm-v1",
        "git_sha": "abc1234",
    },
    "providers": {
        "deepgram": {
            "status": "ok",
            "model": "nova-3",
            "live": True,
            "clips": [clip(f"c{i}", 0.1, float(i + 1)) for i in range(4)]
            + [clip("c4", 0, 0, status="failed")],
        },
        "assemblyai": {"status": "skipped", "reason": "skipped: no key (ASSEMBLYAI_API_KEY)"},
        "fake": {
            "status": "ok",
            "model": "fake-1",
            "live": False,
            "clips": [clip("c0", 0.0, 0.1)],
        },
    },
}


def test_percentile_linear_interpolation():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    assert percentile([7], 95) == 7


def test_percentile_empty():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarize_corpus_wer_and_counts():
    rows = {r["provider"]: r for r in summarize(RESULTS)}
    d = rows["deepgram"]
    assert d["wer"] == pytest.approx(0.1)
    assert (d["clips"], d["failures"]) == (4, 1)
    assert d["p50"] == pytest.approx(2.5)
    assert d["p95"] == pytest.approx(3.85)
    assert d["model"] == "nova-3"
    assert rows["assemblyai"]["status"] == "skipped"


def test_markdown_table_and_metadata():
    md = render_markdown(RESULTS)
    assert "| Provider | WER | Clips | Failures | p50 (s) | p95 (s) | Model |" in md
    assert "| deepgram | 10.00% | 4 | 1 | 2.50 | 3.85 | nova-3 |" in md
    assert "skipped: no key (ASSEMBLYAI_API_KEY)" in md
    assert "FAKE" in md
    for needle in ("2026-11-10", "LibriSpeech test-clean", "norm-v1", "abc1234", "n=4"):
        assert needle in md


def test_markdown_labels_untested_live_adapters():
    res = {**RESULTS, "providers": {"deepgram": {**RESULTS["providers"]["deepgram"], "live": False}}}
    assert "untested against live API" in render_markdown(res)


def test_readme_not_yet_run_when_no_results():
    base = f"# T\n\n{README_START}\nold\n{README_END}\n\nafter\n"
    out = update_readme(base, None)
    assert "not yet run" in out and "old" not in out
    assert out.startswith("# T") and out.endswith("after\n")


def test_readme_block_replaced_with_results_and_idempotent():
    base = f"intro\n{README_START}\nx\n{README_END}\nouttro\n"
    once = update_readme(base, RESULTS)
    assert "10.00%" in once and "intro" in once and "outtro" in once
    assert update_readme(once, RESULTS) == once


def test_readme_without_markers_raises():
    with pytest.raises(ValueError):
        update_readme("no markers", None)

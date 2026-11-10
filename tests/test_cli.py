import json

from sttbench.__main__ import main
from tests.conftest import make_wav


def test_validate_ok(dataset, capsys):
    clips, truth = dataset
    assert main(["validate", "--clips", str(clips), "--truth", str(truth)]) == 0


def test_validate_failure_exit_2_lists_every_problem(dataset, capsys):
    clips, truth = dataset
    (truth / "a.txt").unlink()
    (truth / "b.txt").write_text("yes", encoding="utf-8")
    assert main(["validate", "--clips", str(clips), "--truth", str(truth)]) == 2
    err = capsys.readouterr().err
    assert "MISSING_TRUTH" in err and "DURATION_MISMATCH" in err


def test_run_aborts_before_any_provider_call_on_bad_truth(dataset, tmp_path, capsys):
    clips, truth = dataset
    (truth / "a.txt").unlink()
    out = tmp_path / "out"
    code = main(
        ["run", "--providers", "fake", "--clips", str(clips), "--truth", str(truth),
         "--out", str(out)]
    )
    assert code == 2 and not (out / "results.json").exists()


def test_run_fake_writes_results(dataset, tmp_path):
    clips, truth = dataset
    out = tmp_path / "out"
    code = main(
        ["run", "--providers", "fake", "--clips", str(clips), "--truth", str(truth),
         "--out", str(out), "--dataset-name", "unit"]
    )
    assert code == 0
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    meta = data["meta"]
    assert meta["dataset"] == "unit" and meta["n_clips"] == 3
    assert meta["norm_version"] == "norm-v1" and meta["run_date"].endswith("Z")
    assert "git_sha" in meta
    assert data["providers"]["fake"]["status"] == "ok"
    assert (out / "results.md").exists()


def test_run_with_only_skipped_providers_exits_1(dataset, tmp_path, monkeypatch):
    clips, truth = dataset
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    code = main(
        ["run", "--providers", "deepgram", "--clips", str(clips), "--truth", str(truth),
         "--out", str(tmp_path / "o")]
    )
    assert code == 1


def test_report_update_readme(dataset, tmp_path):
    clips, truth = dataset
    out = tmp_path / "out"
    main(["run", "--providers", "fake", "--clips", str(clips), "--truth", str(truth),
          "--out", str(out)])
    readme = tmp_path / "README.md"
    readme.write_text("# X\n<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n", encoding="utf-8")
    assert main(["report", "--results", str(out / "results.json"), "--update-readme",
                 "--readme", str(readme)]) == 0
    assert "FAKE" in readme.read_text(encoding="utf-8")


def test_report_without_results_says_not_yet_run(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n", encoding="utf-8")
    assert main(["report", "--results", str(tmp_path / "missing.json"), "--update-readme",
                 "--readme", str(readme)]) == 0
    assert "not yet run" in readme.read_text(encoding="utf-8")


def test_stream_format_flag(tmp_path):
    clips, truth = tmp_path / "c", tmp_path / "t"
    truth.mkdir()
    make_wav(clips / "a.wav", 10, sample_rate=8000)
    (truth / "a.txt").write_text(" ".join(["w"] * 20), encoding="utf-8")
    base = ["validate", "--clips", str(clips), "--truth", str(truth)]
    assert main(base) == 0
    assert main([*base, "--stream-format"]) == 2

import itertools

import pytest

from sttbench.providers.base import PermanentProviderError
from sttbench.providers.fake import FakeProvider
from sttbench.runner import run_benchmark

META = {"dataset": "unit-test", "run_date": "2026-01-01T00:00:00Z"}


async def no_sleep(_):
    return None


def ref(truth, stem):
    return (truth / f"{stem}.txt").read_text(encoding="utf-8")


async def test_basic_run_perfect_transcripts(dataset):
    clips, truth = dataset
    fake = FakeProvider(transcripts={s: ref(truth, s) for s in "abc"})
    res = await run_benchmark({"fake": fake}, {}, clips, truth, meta=META, sleep=no_sleep)
    p = res["providers"]["fake"]
    assert res["meta"] == META and p["status"] == "ok"
    assert [c["clip"] for c in p["clips"]] == ["a", "b", "c"]
    assert all(c["status"] == "ok" and c["wer"] == 0.0 for c in p["clips"])
    assert p["clips"][0]["reference"] == ref(truth, "a")
    assert p["model"].startswith("fake")
    assert p["audio_s_sent"] == pytest.approx(30)


async def test_per_clip_failure_does_not_abort(dataset):
    clips, truth = dataset
    fake = FakeProvider(
        transcripts={s: ref(truth, s) for s in "abc"},
        errors={"b": PermanentProviderError("400 bad")},
    )
    res = await run_benchmark({"fake": fake}, {}, clips, truth, meta=META, sleep=no_sleep)
    by = {c["clip"]: c for c in res["providers"]["fake"]["clips"]}
    assert by["a"]["status"] == "ok" and by["c"]["status"] == "ok"
    assert by["b"]["status"] == "failed" and "400 bad" in by["b"]["error"]
    assert by["b"].get("wer") is None


async def test_transient_errors_are_retried(dataset):
    clips, truth = dataset
    fake = FakeProvider(transcripts={s: ref(truth, s) for s in "abc"}, fail_times={"a": 2})
    delays = []

    async def sleep(d):
        delays.append(d)

    res = await run_benchmark({"fake": fake}, {}, clips, truth, meta=META, sleep=sleep)
    a = res["providers"]["fake"]["clips"][0]
    assert a["status"] == "ok" and a["attempts"] == 3
    assert len(delays) == 2


async def test_transient_exhaustion_recorded_as_failure(dataset):
    clips, truth = dataset
    fake = FakeProvider(transcripts={s: ref(truth, s) for s in "abc"}, fail_times={"a": 99})
    res = await run_benchmark({"fake": fake}, {}, clips, truth, meta=META, sleep=no_sleep)
    a = res["providers"]["fake"]["clips"][0]
    assert a["status"] == "failed" and a["attempts"] == 4


async def test_skipped_provider_recorded(dataset):
    clips, truth = dataset
    res = await run_benchmark(
        {}, {"deepgram": "skipped: no key (DEEPGRAM_API_KEY)"}, clips, truth, meta=META
    )
    assert res["providers"]["deepgram"] == {
        "status": "skipped",
        "reason": "skipped: no key (DEEPGRAM_API_KEY)",
    }


async def test_latency_is_wall_clock_of_successful_attempt(dataset):
    clips, truth = dataset
    fake = FakeProvider(transcripts={s: ref(truth, s) for s in "abc"})
    ticks = itertools.count(0, 0.5)  # each clock() call advances 0.5
    res = await run_benchmark(
        {"fake": fake}, {}, clips, truth, meta=META, sleep=no_sleep, clock=lambda: next(ticks)
    )
    assert all(c["latency_s"] == pytest.approx(0.5) for c in res["providers"]["fake"]["clips"])


async def test_concurrency_keeps_clip_order(dataset):
    clips, truth = dataset
    fake = FakeProvider(transcripts={s: ref(truth, s) for s in "abc"})
    res = await run_benchmark(
        {"fake": fake}, {}, clips, truth, meta=META, sleep=no_sleep, concurrency=3
    )
    assert [c["clip"] for c in res["providers"]["fake"]["clips"]] == ["a", "b", "c"]

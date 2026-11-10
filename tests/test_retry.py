import pytest

from sttbench.providers.base import PermanentProviderError, TransientProviderError
from sttbench.retry import retry_async


class Flaky:
    def __init__(self, failures, exc=TransientProviderError):
        self.failures, self.exc, self.calls = failures, exc, 0

    async def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc(f"fail {self.calls}")
        return "ok"


def recorder():
    delays = []

    async def sleep(d):
        delays.append(d)

    return delays, sleep


async def test_success_first_try_no_sleep():
    delays, sleep = recorder()
    assert await retry_async(Flaky(0), sleep=sleep, jitter=lambda: 0) == "ok"
    assert delays == []


async def test_transient_then_success_backoff_1_2():
    delays, sleep = recorder()
    f = Flaky(2)
    assert await retry_async(f, sleep=sleep, jitter=lambda: 0) == "ok"
    assert f.calls == 3 and delays == [1, 2]


async def test_exhausts_after_four_attempts_with_1_2_4():
    delays, sleep = recorder()
    f = Flaky(99)
    with pytest.raises(TransientProviderError, match="fail 4"):
        await retry_async(f, sleep=sleep, jitter=lambda: 0)
    assert f.calls == 4 and delays == [1, 2, 4]


async def test_permanent_not_retried():
    delays, sleep = recorder()
    f = Flaky(99, PermanentProviderError)
    with pytest.raises(PermanentProviderError):
        await retry_async(f, sleep=sleep, jitter=lambda: 0)
    assert f.calls == 1 and delays == []


async def test_jitter_added_to_delay():
    delays, sleep = recorder()
    await retry_async(Flaky(1), sleep=sleep, jitter=lambda: 0.25)
    assert delays == [1.25]


async def test_attempts_configurable():
    f = Flaky(99)
    _, sleep = recorder()
    with pytest.raises(TransientProviderError):
        await retry_async(f, attempts=2, sleep=sleep, jitter=lambda: 0)
    assert f.calls == 2

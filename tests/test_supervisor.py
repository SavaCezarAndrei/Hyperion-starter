"""The supervisor restarts workers that overrun their estimate + grace (no LLM, ~3 s).

    uv run pytest tests/test_supervisor.py -v

Time is scaled down: estimates of 0.2 s and a grace of 0.3 s instead of 1 minute.
"""

import asyncio

import pytest

import estimates
import progress
import supervisor
from conftest import loop


@pytest.fixture(autouse=True)
def fast_clock(monkeypatch):
    monkeypatch.setattr(supervisor, "GRACE", 0.3)
    monkeypatch.setattr(supervisor, "CHECK_EVERY", 0.05)
    monkeypatch.setattr(estimates, "DEFAULTS", {"test": 0.2, "web": 1.0, "question": 0.2})
    monkeypatch.setattr(estimates, "MINIMUM", 0)
    monkeypatch.setattr(estimates, "_average", {})


def run(coro):
    async def inside_a_request():
        channel = progress.open_channel()
        result = await coro
        statuses = []
        while not channel.queue.empty():
            statuses.append(channel.queue.get_nowait()["status"])
        return result, statuses

    return loop.run_until_complete(inside_a_request())


def counting(behaviour):
    """A worker that runs `behaviour(attempt)`; returns (worker, attempts list)."""
    attempts = []

    async def worker():
        attempts.append(1)
        return await behaviour(len(attempts))

    return worker, attempts


def test_worker_over_estimate_plus_grace_is_restarted():
    async def behaviour(attempt):
        await asyncio.sleep(10 if attempt == 1 else 0)  # first attempt hangs
        return "done"

    worker, attempts = counting(behaviour)
    result, statuses = run(supervisor.supervise("nginx.yaml", "test", worker, lambda: "gave up"))
    assert result == "done" and len(attempts) == 2
    assert any("nginx.yaml" in s and "longer than expected" in s and "restarting" in s
               for s in statuses)


def test_worker_within_estimate_plus_grace_is_left_alone():
    async def behaviour(attempt):
        await asyncio.sleep(0.4)  # over the 0.2 s estimate, but within + 0.3 s grace
        return "done"

    worker, attempts = counting(behaviour)
    result, _ = run(supervisor.supervise("slow", "test", worker, lambda: "gave up"))
    assert result == "done" and len(attempts) == 1


def test_extending_the_budget_prevents_a_restart():
    async def behaviour(attempt):
        progress.extend(1.0, "🌐 Searching the web")  # a slower path, announced
        await asyncio.sleep(0.8)  # > 0.2 + 0.3 without the extension
        return "done"

    worker, attempts = counting(behaviour)
    result, statuses = run(supervisor.supervise("q", "test", worker, lambda: "gave up"))
    assert result == "done" and len(attempts) == 1
    assert any("Searching the web (≈1s more) [q]" in s for s in statuses)


def test_crashed_worker_is_restarted():
    async def behaviour(attempt):
        if attempt < 3:
            raise RuntimeError("boom")
        return "done"

    worker, attempts = counting(behaviour)
    result, statuses = run(supervisor.supervise("task", "test", worker, lambda: "gave up"))
    assert result == "done" and len(attempts) == 3
    assert sum("ran into a problem" in s for s in statuses) == 2


def test_gives_up_after_max_attempts():
    async def behaviour(attempt):
        await asyncio.sleep(10)

    worker, attempts = counting(behaviour)
    result, statuses = run(supervisor.supervise("hopeless", "test", worker, lambda: "gave up"))
    assert result == "gave up" and len(attempts) == supervisor.MAX_ATTEMPTS
    assert "skipping it" in statuses[-1]


def test_estimates_learn_from_finished_tasks():
    assert estimates.estimate("test") == 0.2  # the default before any measurement
    estimates.record("test", 4.0)
    assert estimates.estimate("test") == round(4.0 * estimates.HEADROOM)
    estimates.record("test", 14.0)  # the server got slower: the estimate grows
    assert estimates.estimate("test") > round(4.0 * estimates.HEADROOM)


def test_reports_are_attributed_to_their_task():
    async def worker():
        progress.report("✍️ Writing")
        return "done"

    _, statuses = run(supervisor.supervise("demo/app.yaml", "test", worker, lambda: "gave up"))
    assert statuses == ["✍️ Writing [demo/app.yaml]"]

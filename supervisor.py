"""Supervisor: watches every worker task and restarts the ones that run too long.

It is deliberately plain code, not an LLM: the component that judges "too slow"
must not be able to stall or misjudge itself.

The orchestrator (the worker manager) gives every task an estimated duration
(estimates.py). A task still running GRACE seconds after its estimate is considered
stuck: the supervisor tells the user, cancels it and starts it again, up to
MAX_ATTEMPTS times. A task that crashes is restarted the same way - except for
errors no retry can fix (a rejected API key). Restarting is safe because workers
never change the workspace themselves: they only return the actions, which are
sent after they finish.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import estimates
import progress
import prompts
from llm import is_fatal

log = logging.getLogger("hyperion.supervisor")

T = TypeVar("T")

GRACE = 60  # seconds allowed beyond the estimate before a restart
MAX_ATTEMPTS = 3  # the first run + 2 restarts
CHECK_EVERY = 2  # seconds between checks


async def supervise(
    name: str,
    kind: str,
    start: Callable[[], Awaitable[T]],
    give_up: Callable[[], T],
) -> T:
    """Run `start()` as task `name` (of type `kind`, see estimates.py); restart it
    when it overruns its estimate + GRACE or crashes; after MAX_ATTEMPTS, `give_up()`."""
    for attempt in range(1, MAX_ATTEMPTS + 1):

        async def run() -> T:
            progress.set_task(name)  # this task's reports are attributed to `name`
            return await start()

        estimate = estimates.estimate(kind)
        progress.start_budget(name, estimate)
        began = time.monotonic()
        worker = asyncio.create_task(run())
        problem = ""
        while True:
            done, _ = await asyncio.wait({worker}, timeout=CHECK_EVERY)
            if done:
                try:
                    result = worker.result()
                except Exception as exc:
                    if is_fatal(exc):
                        raise  # e.g. a rejected API key: restarting cannot help
                    log.exception("task %r crashed (attempt %d)", name, attempt)
                    problem = prompts.SUPERVISOR_CRASHED.format(task=name)
                    break
                # Time the worker added for a slower path (e.g. the web) is not this
                # kind's normal duration: leave it out of the running average.
                _, budget = progress.overrun(name)
                extra = budget - estimate if budget != float("inf") else 0
                estimates.record(kind, max(1.0, time.monotonic() - began - extra))
                return result
            running, budget = progress.overrun(name)
            if running > budget + GRACE:
                worker.cancel()
                log.warning("task %r ran %.0fs, estimate %.0fs (attempt %d)",
                            name, running, budget, attempt)
                problem = prompts.SUPERVISOR_SLOW.format(
                    task=name, seconds=int(running), estimate=int(budget)
                )
                break
        if attempt < MAX_ATTEMPTS:
            progress.report(
                problem + prompts.SUPERVISOR_RESTART.format(next=attempt + 1, total=MAX_ATTEMPTS)
            )
        else:
            progress.report(problem + prompts.SUPERVISOR_GIVE_UP.format(total=MAX_ATTEMPTS))
    log.error("task %r gave up after %d attempts", name, MAX_ATTEMPTS)
    return give_up()

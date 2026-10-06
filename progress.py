"""Live progress for one request: any part of the agent can report what it is doing.

The shared LLM server can be slow. Instead of cutting calls short, the user sees
each step as it happens ("✍️ Writing nginx.yaml...", "🔁 Fixing 1 problem...") and,
when nothing new happens for a while, a heartbeat with the elapsed time.

Each worker task also has a time budget here: the orchestrator's estimate, which the
worker can extend when it takes a slower path (e.g. the web fallback). The
supervisor (supervisor.py) restarts a task that runs past its budget + 1 minute.

main.py opens a channel per request; `report` finds it - and the task it runs in -
through context variables, so it also works inside the orchestrator's parallel
worker tasks (asyncio tasks inherit the context they were created in).
"""

import asyncio
import time
from contextvars import ContextVar

_channel: ContextVar["Channel | None"] = ContextVar("progress_channel", default=None)
_task: ContextVar[str | None] = ContextVar("progress_task", default=None)


class Channel:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self.started = time.monotonic()
        self.step = "starting"
        self.budgets: dict[str, list[float]] = {}  # task -> [started, estimated seconds]

    def elapsed(self) -> int:
        return int(time.monotonic() - self.started)


def open_channel() -> Channel:
    channel = Channel()
    _channel.set(channel)
    return channel


def current() -> Channel | None:
    return _channel.get()


def set_task(name: str) -> None:
    """Called by the supervisor inside a worker task: its reports are attributed to it."""
    _task.set(name)


def start_budget(name: str, seconds: float) -> None:
    """The supervisor starts the clock for one attempt of a task."""
    channel = _channel.get()
    if channel is not None:
        channel.budgets[name] = [time.monotonic(), float(seconds)]


def extend(seconds: float, why: str) -> None:
    """A worker taking a slower path adds time to its own budget, and says so."""
    task, channel = _task.get(), _channel.get()
    if task and channel is not None and task in channel.budgets:
        channel.budgets[task][1] += seconds
        report(f"{why} (≈{int(seconds)}s more)")


def overrun(name: str) -> tuple[float, float]:
    """(seconds running, estimated seconds) of the task's current attempt."""
    channel = _channel.get()
    if channel is None or name not in channel.budgets:
        return 0.0, float("inf")
    started, estimate = channel.budgets[name]
    return time.monotonic() - started, estimate


def report(step: str) -> None:
    """Show a step to the user; inside a worker task it is labelled with the task."""
    task = _task.get()
    channel = _channel.get()
    if channel is None:  # outside a request (tests, scripts): nothing to show
        return
    text = f"{step} [{task}]" if task else step
    channel.step = text
    channel.queue.put_nowait({"status": text})

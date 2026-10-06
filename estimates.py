"""How long a worker task should take: the orchestrator's estimate per task.

Estimates come from measurements, not from asking the LLM (which cannot predict its
own speed): each task type starts from a default measured on legion1 with a safety
margin, and every finished task updates a running average for its type. So when
the shared server gets slower, the estimates grow with it.

The supervisor restarts a worker that runs past its estimate + supervisor.GRACE.
"""

import logging

log = logging.getLogger("hyperion.estimates")

# Seconds, measured on legion1 (typical x ~2 for headroom).
DEFAULTS = {
    "question": 10,  # retrieve + draft + verify: ~2-3 s measured
    "web": 25,  # extra when a question falls back to the web search: ~10-20 s measured
    "create_file": 20,  # draft + backend validation + LLM check, up to 3 attempts: 4-7 s
    "edit_file": 20,
    "read_file": 3,
    "validate_file": 3,
    "delete_file": 2,
    "delete_folder": 2,
    "create_folder": 2,
}
SMOOTHING = 0.3  # weight of the newest measurement in the running average
HEADROOM = 1.5  # estimate = running average x HEADROOM
MINIMUM = 2

_average: dict[str, float] = {}


def estimate(kind: str) -> int:
    """Seconds this kind of task should take."""
    default = DEFAULTS.get(kind, DEFAULTS["question"])
    if kind not in _average:
        return default
    return max(MINIMUM, round(_average[kind] * HEADROOM))


def record(kind: str, seconds: float) -> None:
    """Learn from a finished task."""
    previous = _average.get(kind)
    _average[kind] = seconds if previous is None else (
        SMOOTHING * seconds + (1 - SMOOTHING) * previous
    )
    log.info("%s took %.1fs; running average %.1fs", kind, seconds, _average[kind])

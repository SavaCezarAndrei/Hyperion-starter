"""Orchestrator: plans a request into tasks and runs one worker per task.

The number of workers is dynamic - one per task, at most MAX_WORKERS at a time.
"What is a DeviceNode?" -> 1 worker; "What is a DeviceNode and who coordinates
HYPER-AI?" -> 2 workers in parallel. Results are streamed in the order of the plan
as soon as each one (and every one before it) is ready.
"""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

import memory
import prompts
import workers
from llm import llm_json

log = logging.getLogger("hyperion.orchestrator")

MAX_TASKS = 4
MAX_WORKERS = 4  # parallel LLM calls; legion1 is a single shared GPU
WORKER_TIMEOUT = 90  # seconds; the shared server sometimes stalls

# Only requests that look compound pay for a planning call.
COMPOUND = re.compile(r"\?.+\?|\b(and|also|plus|as well as)\b|;", re.I | re.S)


async def plan_questions(text: str) -> list[str]:
    if not COMPOUND.search(text):
        return [text]
    raw = ""
    try:
        raw = (
            await llm_json.ainvoke(
                [SystemMessage(prompts.PLANNER), HumanMessage(f"Request: {text}")]
            )
        ).text
        questions = [q.strip() for q in json.loads(raw).get("questions", []) if str(q).strip()]
    except Exception as exc:
        log.warning("planner failed (%r), output %r - single task", exc, raw)
        return [text]
    log.info("plan: %s", questions)
    return questions[:MAX_TASKS] or [text]


def render(result: workers.Result) -> str:
    """A worker result as chat text. Anything not from the documentation is marked red."""
    if result.kind == "docs":
        return result.text + "\n\nSources (HYPER-AI documentation): " + "; ".join(result.sources)
    if result.kind == "web":
        links = "\n".join(f"  {link}" for link in result.sources)
        return (
            prompts.NOT_IN_DOCS
            + prompts.WEB_HEADER
            + result.text
            + f"\n\n🔴 Web sources (not verified by HYPER-AI):\n{links}"
        )
    if result.kind == "web_no_answer":
        links = "\n".join(f"  {link}" for link in result.sources)
        return prompts.NOT_IN_DOCS + prompts.WEB_NO_ANSWER + links
    if result.kind == "unverified":
        # With sources it was the web that failed; without, the documentation itself.
        prefix = prompts.NOT_IN_DOCS + " " if result.sources else ""
        return prefix + prompts.COULD_NOT_VERIFY
    return prompts.NOT_IN_DOCS + prompts.WEB_UNAVAILABLE


async def answer_questions(session: memory.Session, text: str) -> AsyncIterator[dict]:
    yield {"response": prompts.STATUS_SEARCHING}
    tasks = await plan_questions(text)
    if len(tasks) > 1:
        yield {"response": prompts.STATUS_SPLIT.format(n=len(tasks))}

    limit = asyncio.Semaphore(MAX_WORKERS)

    async def run(task: str) -> workers.Result:
        async with limit:
            try:
                return await asyncio.wait_for(
                    workers.question_worker(session, task), WORKER_TIMEOUT
                )
            except Exception:
                log.exception("worker failed for %r", task)
                return workers.Result(task, "none")

    running = [asyncio.create_task(run(t)) for t in tasks]
    for i, job in enumerate(running, 1):
        result = await job
        prefix = f"{i}) {result.task}\n" if len(tasks) > 1 else ""
        separator = "\n\n" if i < len(tasks) else ""
        yield {"response": prefix + render(result) + separator}

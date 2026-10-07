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
import time
from collections.abc import AsyncIterator
from dataclasses import asdict

from langchain_core.messages import HumanMessage, SystemMessage

import confirm
import estimates
import knowledge
import memory
import progress
import prompts
import supervisor
import workers
import workspace
from llm import llm_json

log = logging.getLogger("hyperion.orchestrator")

MAX_TASKS = 4
MAX_WORKERS = 4  # parallel LLM calls; legion1 is a single shared GPU

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
    if result.kind == "file":
        return prompts.FROM_FILES.format(files=", ".join(result.sources)) + result.text
    if result.kind == "project":
        return prompts.FROM_PROJECT.format(files=", ".join(result.sources)) + result.text
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


MAX_LISTED_FILES = 60  # workspace paths shown to the planner


def transcript(session: memory.Session, turns: int = 2) -> str:
    lines = [f"{m.type}: {m.text[:300]}" for m in memory.history(session, max_turns=turns)]
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n" if lines else ""


async def plan_actions(
    session: memory.Session, text: str, entries: workspace.Entries
) -> list[workers.Task]:
    # Each file with what the knowledge graph knows about it (app, image, ports), so
    # "change the port of the nginx app" finds the right file.
    files = knowledge.describe_files(sorted(entries.files)[:MAX_LISTED_FILES])
    system = prompts.ACTION_PLANNER.replace(
        "{last_file}", session.last_file or "none yet"
    ).replace("{files}", "\n" + "\n".join(f"- {f}" for f in files) if files else "(the workspace is empty)")
    raw = ""
    try:
        raw = (
            await llm_json.ainvoke(
                [SystemMessage(system), HumanMessage(f"{transcript(session)}Request: {text}")]
            )
        ).text
        items = json.loads(raw).get("tasks", [])
        tasks = [
            workers.Task(
                op=str(t.get("op", "")).strip(),
                path=str(t.get("path") or ""),
                kind=t.get("kind") if t.get("kind") in ("native", "device") else None,
                workload=t.get("workload"),
                request=str(t.get("request") or text),
            )
            for t in items
            if isinstance(t, dict)
        ]
    except Exception as exc:
        log.warning("action planner failed (%r), output %r", exc, raw)
        return []
    # create_file makes its folders itself; a separate create_folder for the same
    # folder would race it in the IDE (actions run concurrently) and log an error.
    file_folders = {
        t.path.replace("\\", "/").strip("/").rsplit("/", 1)[0]
        for t in tasks
        if t.op == "create_file" and "/" in t.path
    }
    tasks = [
        t for t in tasks if not (t.op == "create_folder" and t.path.strip("/") in file_folders)
    ]
    log.info("action plan: %s", tasks)
    return tasks[:MAX_TASKS]


TARGETS_EXISTING = ("edit_file", "delete_file", "read_file", "validate_file", "delete_folder")


def guesses_target(task: workers.Task, text: str, session: memory.Session) -> bool:
    """A change to an existing file or folder must target one the user named, or the
    file this conversation is working on ("it") - never one the planner picked."""
    if task.op not in TARGETS_EXISTING:
        return False
    name = task.path.strip()
    if name.lower() in workers.LAST_FILE:
        return False  # "it": the worker uses session.last_file, or asks
    if session.last_file and name.strip("/") in (
        session.last_file, session.last_file.rsplit("/", 1)[-1]
    ):
        return False
    if any(knowledge.identifies(p, text) for p in workspace.resolve(name, knowledge.known_files())):
        return False  # named through the graph: "the nginx app", "my-web-app"
    return not workspace.mentioned(name, text)


async def run_task(
    session: memory.Session, task: workers.Task, entries: workspace.Entries
) -> workers.ActionResult:
    return await supervisor.supervise(
        task.path or task.op,
        task.op,
        lambda: workers.action_worker(session, task, entries),
        lambda: workers.ActionResult(
            f"I couldn't finish {task.path or 'that request'} - nothing was "
            "changed. Please try again."
        ),
    )


def pending_item(task: workers.Task, result: workers.ActionResult) -> dict:
    item = {"question": result.question, "actions": result.actions,
            "done": result.message, "path": result.path}
    if task.op == "edit_file":
        # Kept so the user can answer "no, in other.yaml" and get the same change there.
        item["task"] = asdict(task)
    return item


# A reply that also asks for something else is a new request, not "use this file instead".
OTHER_REQUEST = re.compile(
    r"\b(delete|remove|create|read|show|validate|check|rename|what|how|why)\b", re.I
)
ANOTHER_FILE = re.compile(r"\b(another|other|different|wrong)\b.*\bfile\b", re.I)


async def retarget(session: memory.Session, text: str) -> AsyncIterator[dict]:
    """A pending edit, answered with another file's name: make the same change there.

    Yields nothing if the reply is not about choosing another file (the caller then
    handles it as a yes/no or a new message).
    """
    items = (session.pending or {}).get("items", [])
    if len(items) != 1 or "task" not in items[0] or confirm.answer(text) == "yes":
        return
    if OTHER_REQUEST.search(text):
        return
    current = items[0]["path"]
    try:
        entries = await workspace.list_entries()
    except Exception as exc:
        log.warning("cannot list the workspace: %s", exc)
        return
    others = sorted(entries.files - {current})
    said = text.lower()
    named = [p for p in others if workspace.mentioned(p, text)]
    exact = [p for p in named if p.lower() in said or p.lower().rsplit("/", 1)[-1] in said]
    named = exact or named
    if not named:
        if ANOTHER_FILE.search(text):  # "use another file": keep the change waiting, ask which
            session.pending["created"] = time.time()
            listed = ", ".join(others[:MAX_LISTED_FILES]) or "(no other files)"
            yield {"response": prompts.WHICH_OTHER_FILE.format(files=listed)}
        return
    if len(named) > 1:
        session.pending["created"] = time.time()
        yield {"response": prompts.SEVERAL_FILES_MATCH.format(files=", ".join(named))}
        return

    session.pending = None
    task = workers.Task(**{**items[0]["task"], "path": named[0]})
    log.info("retarget pending edit %s -> %s", current, task.path)
    progress.report(f"📋 Making the same change in {task.path} instead...")
    result = await run_task(session, task, entries)
    if result.question and result.actions:
        item = pending_item(task, result)
        confirm.store(session, [item])
        yield {"response": confirm.ask([item])}
    else:
        yield {"response": result.message}


async def run_actions(session: memory.Session, text: str) -> AsyncIterator[dict]:
    if unsafe := workspace.unsafe_paths(text):
        # Checked on the user's words: the planner would "repair" these paths.
        yield {"response": prompts.UNSAFE_PATH.format(path=unsafe[0])}
        return
    progress.report("🗂️ Looking at your workspace...")
    try:
        entries = await workspace.list_entries()
    except Exception as exc:
        log.warning("cannot list the workspace: %s", exc)
        entries = workspace.Entries()
    progress.report("📋 Planning the changes...")
    tasks = await plan_actions(session, text, entries)
    if not tasks:
        yield {"response": prompts.ACTION_NOTHING_TO_DO}
        return
    if guessed := [t for t in tasks if guesses_target(t, text, session)]:
        # The planner picked an existing file the user never named: ask instead.
        log.info("planner guessed targets %s for %r", [t.path for t in guessed], text)
        listed = ", ".join(sorted(entries.files)[:MAX_LISTED_FILES]) or "(no files yet)"
        yield {"response": prompts.WHICH_FILE.format(files=listed)}
        return
    progress.report(
        "📋 Plan: "
        + "; ".join(
            f"{t.op.replace('_', ' ')} {t.path} (≈{estimates.estimate(t.op)}s)" for t in tasks
        )
    )

    limit = asyncio.Semaphore(MAX_WORKERS)

    async def run(task: workers.Task) -> workers.ActionResult:
        async with limit:
            return await run_task(session, task, entries)

    running = [asyncio.create_task(run(t)) for t in tasks]
    to_confirm = []
    try:
        for task, job in zip(tasks, running):
            result = await job
            if result.question and result.actions:
                # State-changing: held back until the user says yes (confirm.py).
                to_confirm.append(pending_item(task, result))
                continue
            for action in result.actions:
                yield action
            if result.path:
                session.last_file = result.path
            yield {"response": result.message + "\n"}
    finally:
        for job in running:  # e.g. a rejected API key in one task: stop the others
            job.cancel()
    if to_confirm:
        confirm.store(session, to_confirm)
        yield {"response": confirm.ask(to_confirm)}


async def answer_questions(session: memory.Session, text: str) -> AsyncIterator[dict]:
    tasks = await plan_questions(text)
    estimate = estimates.estimate("question")
    if len(tasks) > 1:
        progress.report(prompts.STATUS_SPLIT.format(n=len(tasks), seconds=estimate))
    else:
        progress.report(prompts.STATUS_ONE_QUESTION.format(seconds=estimate))

    limit = asyncio.Semaphore(MAX_WORKERS)

    async def run(task: str) -> workers.Result:
        async with limit:
            return await supervisor.supervise(
                task,
                "question",
                lambda: workers.question_worker(session, task),
                lambda: workers.Result(task, "none"),
            )

    running = [asyncio.create_task(run(t)) for t in tasks]
    try:
        for i, job in enumerate(running, 1):
            result = await job
            prefix = f"{i}) {result.task}\n" if len(tasks) > 1 else ""
            separator = "\n\n" if i < len(tasks) else ""
            yield {"response": prefix + render(result) + separator}
    finally:
        for job in running:  # e.g. a rejected API key in one task: stop the others
            job.cancel()

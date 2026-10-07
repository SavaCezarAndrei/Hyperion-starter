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
import rag
import supervisor
import workers
import workspace
from llm import llm_json

log = logging.getLogger("hyperion.orchestrator")

MAX_TASKS = 4
MAX_WORKERS = 4  # parallel LLM calls; legion1 is a single shared GPU

# Only requests that look compound pay for a planning call.
COMPOUND = re.compile(r"\?.+\?|\b(and|also|plus|as well as)\b|;", re.I | re.S)
LEADING_JOIN = re.compile(r"^\s*(and|also|plus)\b", re.I)
MIN_PLAN_OVERLAP = 0.5  # share of a planned question's words that the user wrote


def overlap(words: list[str], asked: set[str]) -> float:
    return len(set(words) & asked) / len(set(words)) if words else 0.0


async def plan_questions(text: str) -> tuple[list[str], list[str]]:
    """(questions to answer, questions over the MAX_TASKS limit - reported, not dropped)."""
    # A leading "and"/"also" continues the conversation ("and how is it registered?");
    # it does not join two questions.
    if not COMPOUND.search(LEADING_JOIN.sub("", text)):
        return [text], []
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
        return [text], []
    log.info("plan: %s", questions)
    # The planner can invent questions from its own examples (a bare "and how is it
    # registered?" became "What is a DeviceNode?" + ...). Keep only questions made of
    # the user's own words.
    asked = set(rag.tokenize(text))
    kept = [q for q in questions if overlap(rag.tokenize(q), asked) >= MIN_PLAN_OVERLAP]
    if len(kept) < len(questions):
        log.warning("planner invented %s for %r", [q for q in questions if q not in kept], text)
    questions = kept
    if not questions:
        return [text], []
    return questions[:MAX_TASKS], questions[MAX_TASKS:]


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
) -> tuple[list[workers.Task], list[str]]:
    """(tasks to run, notes for the user about tasks that were left out)."""
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
        return [], []
    # A path the user typed is used as typed: the planner sometimes "corrects" it
    # (qa-auto/web.yaml -> qa/auto/web.yaml) and then a different file is named.
    typed = [p.replace("\\", "/") for p in YAML_NAME.findall(text)]
    for t in tasks:
        name = t.path.replace("\\", "/").strip("/")
        if name and name not in typed:
            same = [p for p in typed if p.rsplit("/", 1)[-1] == name.rsplit("/", 1)[-1]]
            if len(set(same)) == 1:
                t.path = same[0]
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
    # Two tasks writing the same path would race in the IDE: keep the first, say so.
    notes, seen, unique = [], set(), []
    for t in tasks:
        key = t.path.replace("\\", "/").strip("/").lower()
        if t.op in ("create_file", "edit_file") and key in seen:
            notes.append(prompts.SAME_PATH_TWICE.format(path=t.path))
            continue
        seen.add(key)
        unique.append(t)
    if len(unique) > MAX_TASKS:
        listed = "\n".join(f"- {t.op.replace('_', ' ')} {t.path}" for t in unique[MAX_TASKS:])
        notes.append(prompts.TOO_MANY_TASKS.format(n=MAX_TASKS, skipped=listed))
    log.info("action plan: %s", unique)
    return unique[:MAX_TASKS], notes


TARGETS_EXISTING = ("edit_file", "delete_file", "read_file", "validate_file", "delete_folder")


IT = re.compile(r"\b(it|this|that)(\s+(file|one|app))?\b", re.I)
RENAME = re.compile(r"\brename\b", re.I)
# "create something": nothing to build from - ask instead of inventing a whole app.
VAGUE_CREATE = re.compile(
    r"^\s*(please\s+)?(create|make|generate|build|write)\s+(something|anything|stuff|a file|"
    r"some file|it|one)\s*(for me)?\s*[.!?]*\s*$", re.I)
MAX_LISTED_MATCHES = 10


def several(paths: list[str]) -> str:
    paths = sorted(paths)
    listed = ", ".join(paths[:MAX_LISTED_MATCHES])
    if len(paths) > MAX_LISTED_MATCHES:
        listed += f" and {len(paths) - MAX_LISTED_MATCHES} more"
    return prompts.SEVERAL_FILES_MATCH.format(files=listed)


def target_problem(
    task: workers.Task, text: str, session: memory.Session, entries: workspace.Entries
) -> str | None:
    """None when the task's existing file or folder is one the user clearly pointed
    at; otherwise the question to ask. The planner may pick any file, so only the
    user's own words count, strongest first:
      the full path ("qa/web.yaml") - always unique;
      the file name ("web.yaml") or the app name - unique only if one file has it;
      a name stem or part, or an image ("the nginx app") - unique only if one file fits;
      nothing named at all: the file this conversation is working on ("it")."""
    if task.op not in TARGETS_EXISTING:
        return None
    name = task.path.strip()
    if name.lower() in workers.LAST_FILE:
        return None  # "it": the worker uses session.last_file, or asks
    folder = task.op == "delete_folder"
    pool = entries.folders if folder else entries.files
    targets = workspace.resolve(name, pool)
    if not targets:
        return None  # nothing by that name: the worker says so
    if len(targets) > 1:
        return several(targets)
    target = targets[0]
    if target == session.last_file and IT.search(text):
        return None
    if workspace.names_path(target, text):
        return None
    if workspace.names(target, text):
        same_name = [p for p in pool if p.rsplit("/", 1)[-1] == target.rsplit("/", 1)[-1]]
        return None if len(same_name) == 1 else several(same_name)
    by_app = [] if folder else sorted(set(knowledge.files_by_app_name(text)) & pool)
    if target in by_app:
        return None if len(by_app) == 1 else several(by_app)
    loose = {p for p in pool if workspace.mentioned(p, text)}
    if not folder:
        loose |= set(knowledge.files_by_image(text)) & pool
    if target in loose:
        return None if len(loose) == 1 else several(sorted(loose))
    if not loose and not by_app and target == session.last_file:
        return None  # "change the port to 8080" right after working on this file
    listed = ", ".join(sorted(entries.files)[:MAX_LISTED_FILES]) or "(no files yet)"
    return prompts.WHICH_FILE.format(files=listed)


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
YAML_NAME = re.compile(r"[\w./-]+\.ya?ml\b", re.I)


async def retarget(session: memory.Session, text: str) -> AsyncIterator[dict]:
    """A pending edit, answered with another file's name: make the same change there.

    Yields nothing if the reply is not about choosing another file (the caller then
    handles it as a yes/no or a new message).
    """
    items = (session.pending or {}).get("items", [])
    if len(items) != 1 or "task" not in items[0] or confirm.starts_with_yes(text):
        return  # "yes but also add port 443" is a new request, never a file choice
    if OTHER_REQUEST.search(text):
        return
    current = items[0]["path"]
    try:
        entries = await workspace.list_entries()
    except Exception as exc:
        log.warning("cannot list the workspace: %s", exc)
        return
    others = sorted(entries.files - {current})
    # Strongest evidence first: a full path, a file name, an app name or image
    # ("the python one"), then a name stem or part of 3+ letters.
    named = [p for p in others if workspace.names_path(p, text)]
    named = named or [p for p in others if workspace.names(p, text)]
    named = named or [p for p in knowledge.files_named_by(text) if p in others]
    named = named or [p for p in others if workspace.mentioned(p, text)]
    if not named:
        if missing := YAML_NAME.search(text):  # "no, in ghost.yaml": say so, keep waiting
            session.pending["created"] = time.time()
            yield {"response": prompts.NO_SUCH_OTHER_FILE.format(name=missing.group(0), current=current)}
            return
        if ANOTHER_FILE.search(text):  # "use another file": keep the change waiting, ask which
            session.pending["created"] = time.time()
            listed = ", ".join(others[:MAX_LISTED_FILES]) or "(no other files)"
            yield {"response": prompts.WHICH_OTHER_FILE.format(files=listed)}
        return
    if len(named) > 1:
        session.pending["created"] = time.time()
        yield {"response": several(named)}
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
    if RENAME.search(text):
        # The IDE has no rename action; the planner would "rename" the app inside the file.
        yield {"response": prompts.RENAME_UNSUPPORTED}
        return
    if VAGUE_CREATE.match(text):
        yield {"response": prompts.ACTION_NOTHING_TO_DO}
        return
    progress.report("🗂️ Looking at your workspace...")
    try:
        entries = await workspace.list_entries()
    except Exception as exc:
        log.warning("cannot list the workspace: %s", exc)
        entries = workspace.Entries()
    await knowledge.ensure_fresh()
    progress.report("📋 Planning the changes...")
    tasks, notes = await plan_actions(session, text, entries)
    if not tasks:
        yield {"response": prompts.ACTION_NOTHING_TO_DO}
        return
    for task in tasks:
        if problem := target_problem(task, text, session, entries):
            # The planner picked a file the user did not clearly name: ask instead.
            log.info("unclear target %s for %r", task.path, text)
            yield {"response": problem}
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
    if notes:
        yield {"response": "\n".join(notes) + "\n"}
    if to_confirm:
        confirm.store(session, to_confirm)
        yield {"response": confirm.ask(to_confirm)}


async def answer_questions(session: memory.Session, text: str) -> AsyncIterator[dict]:
    tasks, skipped = await plan_questions(text)
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
    if skipped:
        listed = "\n".join(f"- {q}" for q in skipped)
        yield {"response": prompts.TOO_MANY_QUESTIONS.format(n=MAX_TASKS, skipped=listed)}

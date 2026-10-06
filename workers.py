"""Workers: each one handles a single task from the orchestrator's plan.

A worker only ever sees its own task and the evidence it gathered for it - the
small, focused context is what keeps an 8B model from mixing facts up. Every
output is checked by the verifier; a rejected output is retried once with the
verifier's objections, and an output that still fails is never presented as fact.
"""

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

import confirm
import dsl
import estimates
import memory
import progress
import prompts
import rag
import verifier
import web
import workspace
from llm import llm

log = logging.getLogger("hyperion.workers")

ATTEMPTS = 2  # first draft + one retry with the verifier's feedback
FILE_ATTEMPTS = 3  # YAML gets two retries: validator errors are precise and cheap to fix
FLAGGED_OVERLAP = 0.6  # share of a sentence's words found in a flagged claim
MIN_KEPT_CHARS = 20  # less than this left after cutting -> nothing usable


@dataclass
class Result:
    task: str
    kind: Literal["docs", "web", "web_no_answer", "unverified", "none"]
    text: str = ""
    sources: list[str] = field(default_factory=list)


def remove_flagged(answer: str, problems: list[str]) -> str:
    """Drop the sentences of `answer` that the verifier quoted as unsupported."""
    flagged = [set(rag.tokenize(p)) for p in problems]
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        words = set(rag.tokenize(sentence))
        if not words:
            continue
        if any(len(words & f) / len(words) >= FLAGGED_OVERLAP for f in flagged if f):
            continue
        kept.append(sentence.strip())
    return "\n".join(kept) if sum(len(k) for k in kept) >= MIN_KEPT_CHARS else ""


async def draft_and_verify(
    question: str, evidence: str, write: Callable[[str], Awaitable[str]]
) -> tuple[str | None, bool]:
    """Draft with `write(feedback)` and verify, retrying once.

    Returns (answer, covered): covered is False when the writer said the evidence
    lacks the answer. If the retry is rejected too, the flagged sentences are cut
    out; answer is None when nothing verified is left.
    """
    feedback = ""
    for attempt in range(1, ATTEMPTS + 1):
        progress.report("✍️ Writing the answer..." if attempt == 1 else "✍️ Rewriting the answer...")
        answer = (await write(feedback)).strip()
        if rag.not_covered(answer):
            return None, False
        progress.report("🧪 Checking the answer against the sources...")
        verdict = await verifier.check_answer(question, evidence, answer)
        if verdict.ok:
            return answer, True
        log.info("attempt %d rejected for %r: %s", attempt, question, verdict.problems)
        progress.report(f"🔁 The check found {len(verdict.problems)} unsupported claim(s)")
        feedback = prompts.RETRY_FEEDBACK.format(
            problems="\n".join(f"- {p}" for p in verdict.problems), previous=answer
        )
    return remove_flagged(answer, verdict.problems) or None, True


async def question_worker(session: memory.Session, question: str) -> Result:
    # 1. The HYPER-AI documentation.
    progress.report("🔎 Searching the HYPER-AI documentation...")
    docs = await rag.retrieve(session, question)
    if docs:
        evidence = rag.evidence(docs)
        answer, covered = await draft_and_verify(
            question, evidence, lambda fb: rag.draft(session, question, evidence, fb)
        )
        if answer:
            return Result(question, "docs", answer, rag.sources(docs))
        if covered:
            # Relevant documents, but nothing verified: the web would only add noise
            # (a search for a HYPER-AI term returns other products with that name).
            return Result(question, "unverified")

    # 2. The web, clearly marked as such.
    progress.extend(
        estimates.estimate("web"), "🌐 Not in the documentation - searching the web"
    )
    found = await web.gather(rag.search_query(session, question))
    if not found.text:
        return Result(question, "none")
    answer, covered = await draft_and_verify(
        question, found.text, lambda fb: web.draft(question, found.text, fb)
    )
    if answer:
        return Result(question, "web", answer, found.links)
    if not covered:
        return Result(question, "web_no_answer", sources=found.links)
    return Result(question, "unverified", sources=found.links)


# --- IDE actions --------------------------------------------------------------

FILE_OPS = ("create_file", "edit_file", "delete_file", "read_file", "validate_file")
FOLDER_OPS = ("create_folder", "delete_folder")
OPS = FILE_OPS + FOLDER_OPS
SHOWN_CHARS = 3000  # read_file output shown in the chat
LAST_FILE = {"", "it", "this", "that", "<the last file>", "the last file", "last file"}


@dataclass
class Task:
    op: str
    path: str
    kind: str | None = None
    workload: str | None = None
    request: str = ""


@dataclass
class ActionResult:
    message: str
    actions: list[dict] = field(default_factory=list)
    path: str | None = None  # the file this result touched; becomes session.last_file
    # Set for state-changing results: shown to the user, who must confirm (confirm.py).
    question: str | None = None


def describe_errors(report: dict) -> list[str]:
    return [
        f"line {e.get('line', '?')}: {e.get('field') or 'file'} {e.get('message', '')}".strip()
        for e in report.get("errors", [])
    ]


async def file_problems(content: str, kind: str, request: str, original: str | None) -> list[str]:
    """① the ide-backend's validator (or the local DSL check), ② does it do what was asked."""
    report = await workspace.validate_content(content)
    if report is None:
        errors = dsl.check(content, kind)
    else:
        errors = describe_errors(report)
        if not errors and report.get("type") not in (None, kind):
            errors = [f"this is a {report['type']} application, but a {kind} one was requested"]
    if errors:
        return errors
    progress.report("🧐 Checking it matches your request...")
    verdict = await verifier.check_file(request, content, original)
    return verdict.problems


async def write_file(task: Task, original: str | None = None) -> tuple[str, list[str], str]:
    """Draft a YAML file and verify it, retrying with the exact problems.

    Returns (content, remaining problems, kind).
    """
    kind = task.kind if task.kind in ("native", "device") else None
    if original is not None:
        try:
            kind = dsl.kind_of(yaml.safe_load(original)) or kind
        except yaml.YAMLError:
            pass
    kind = kind or "native"
    workload = task.workload if kind == "device" and task.workload in dsl.WORKLOADS else None
    reference = dsl.evidence(kind, workload)
    if original is None:
        base = [
            SystemMessage(prompts.FILE_WRITER_CREATE),
            HumanMessage(f"{reference}\n\nREQUEST: {task.request}"),
        ]
    else:
        base = [
            SystemMessage(prompts.FILE_WRITER_EDIT),
            HumanMessage(
                f"Reference:\n{reference}\n\nCURRENT FILE:\n{original}\n\nCHANGE: {task.request}"
            ),
        ]
    content, problems = "", []
    for attempt in range(1, FILE_ATTEMPTS + 1):
        messages = base
        if problems:
            progress.report(f"🔁 Fixing {len(problems)} problem(s) (attempt {attempt})...")
            fix = prompts.FILE_FIX.format(
                problems="\n".join(f"- {p}" for p in problems), previous=content
            )
            messages = [*base, HumanMessage(fix)]
        else:
            progress.report(f"✍️ Writing the {kind} application profile...")
        content = dsl.strip_fences((await llm.ainvoke(messages)).text)
        progress.report("🧪 Validating with the HYPER-AI rules...")
        problems = await file_problems(content, kind, task.request, original)
        log.info("%s attempt %d: %s", task.path, attempt, problems or "ok")
        if not problems:
            break
    return content, problems, kind


def written(verb: str, path: str, kind: str, problems: list[str]) -> str:
    if not problems:
        return f"✅ {verb} {path} - a {kind} application profile that passes HYPER-AI validation."
    listed = "\n".join(f"  - {p}" for p in problems)
    return (
        f"⚠️ {verb} {path}, but I could not fix every problem - please check these:\n{listed}"
    )


def pick(name: str, candidates: set[str], what: str) -> tuple[str | None, str | None]:
    """(path, None) for a unique match, or (None, message for the user)."""
    matches = workspace.resolve(name, candidates)
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, f"There is no {what} called {name} in your workspace."
    return None, f"Several {what}s match {name}: {', '.join(matches)}. Which one do you mean?"


async def action_worker(
    session: memory.Session, task: Task, entries: workspace.Entries
) -> ActionResult:
    """One planned task -> the IDE actions to stream and a line for the user."""
    if task.op not in OPS:
        return ActionResult(f"I can't do '{task.op}' in the workspace.")
    name = task.path.strip()
    if task.op in ("edit_file", "read_file", "validate_file") and name.lower() in LAST_FILE:
        if not session.last_file:
            return ActionResult("Which file do you mean? I haven't worked on one yet.")
        name = session.last_file
    try:
        path = workspace.clean_path(name, is_file=task.op in FILE_OPS)
    except workspace.PathError as exc:
        return ActionResult(f"I can't use that path: {exc}.")

    if task.op == "create_folder":
        if path in entries.folders:
            return ActionResult(f"The folder {path} already exists.")
        return ActionResult(f"📁 Created folder {path}.", [{"action": "create_folder", "path": path}])

    if task.op == "delete_folder":
        found, problem = pick(path, entries.folders, "folder")
        if problem:
            return ActionResult(problem)
        inside = sum(f.startswith(found + "/") for f in entries.files)
        return ActionResult(
            f"🗑️ Deleted folder {found} and everything in it.",
            [{"action": "delete_folder", "path": found}],
            question=f"- 🗑️ delete the folder {found}"
            + (f" and the {inside} file(s) inside it" if inside else ""),
        )

    if task.op == "create_file" and path not in entries.files:
        content, problems, kind = await write_file(task)
        return ActionResult(
            written("Created", path, kind, problems),
            [{"action": "create_file", "path": path, "content": content}],
            path,
        )

    # Everything else works on an existing file.
    found, problem = pick(path, entries.files, "file")
    if problem:
        return ActionResult(problem)

    if task.op == "delete_file":
        return ActionResult(
            f"🗑️ Deleted {found}.",
            [{"action": "delete_file", "path": found}],
            question=f"- 🗑️ delete the file {found}",
        )

    if task.op == "validate_file":
        report = await validate_file(found)
        if report.get("valid"):
            return ActionResult(f"✅ {found} passes HYPER-AI validation.", path=found)
        listed = "\n".join(f"  - {p}" for p in describe_errors(report))
        return ActionResult(f"❌ {found} has problems:\n{listed}", path=found)

    original = await workspace.read(found)
    if task.op == "read_file":
        shown = original if len(original) <= SHOWN_CHARS else original[:SHOWN_CHARS] + "\n..."
        return ActionResult(f"📄 {found}:\n\n{shown}", path=found)

    # edit_file, or create_file over an existing file: both replace the whole file.
    content, problems, kind = await write_file(task, original)
    if task.op == "edit_file":
        verb, header = "Updated", f"- ✏️ change {found}:"
    else:
        verb, header = "Replaced", f"- ♻️ overwrite the existing {found} with a new file:"
    details = "\n".join(f"    {c}" for c in confirm.changes(original, content))
    status = "" if not problems else f"\n    (⚠️ {len(problems)} validation problem(s) remain)"
    return ActionResult(
        written(verb, found, kind, problems),
        [{"action": "edit_file", "path": found, "content": content}],
        found,
        question=f"{header}\n{details}{status}",
    )


async def validate_file(path: str) -> dict:
    try:
        return await workspace.validate_file(path)
    except workspace.ValidateFileError as exc:
        return {"valid": False, "errors": [{"message": str(exc)}]}

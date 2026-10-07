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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import confirm
import dsl
import estimates
import knowledge
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
    kind: Literal["docs", "file", "project", "web", "web_no_answer", "unverified", "none"]
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
    # A short follow-up ("and how does it register them?") depends on the conversation:
    # it is answered normally, never stored or matched as a question of its own.
    standalone = rag.search_query(session, question) == question
    return await with_knowledge(
        question, lambda: search_question(session, question), learn=standalone
    )


async def btw_worker(question: str) -> Result:
    return await with_knowledge(question, lambda: search_btw(question))


async def with_knowledge(
    question: str, search: Callable[[], Awaitable[Result]], learn: bool = True
) -> Result:
    """The project knowledge graph first: a verified answer learned in this or an
    earlier session, or the project's own facts; otherwise search, and learn."""
    # About the user's own project: answered from the graph only, and never learned -
    # the answer changes with the workspace.
    if result := await answer_from_project(question):
        return result
    if learn and (known := await knowledge.recall(question)):
        progress.report(prompts.KNOWLEDGE_REUSED)
        return Result(question, known.kind, known.text, known.sources)
    result = await search()
    if learn:
        await knowledge.remember(question, result.kind, result.text, result.sources)
    return result


PROJECT_FRESHNESS = 5  # seconds; questions about the user's own files re-read them this often


async def answer_from_project(question: str) -> Result | None:
    """Questions about the user's own apps, answered from the workspace graph:
    in code when the question is structured, otherwise drafted and verified. If no
    draft passes, the graph's facts are shown as they are - never the docs or web."""
    if knowledge.PROJECT_WORDS.search(question) or knowledge.files_by_app_name(question):
        await knowledge.ensure_fresh(PROJECT_FRESHNESS)  # a file edited by hand a moment ago counts
    if found := knowledge.lookup(question):
        progress.report("🧠 Answered from your project's knowledge graph")
        text, paths = found
        return Result(question, "project", text, paths or ["all apps"])
    evidence, paths = knowledge.project_evidence(question)
    if not evidence:
        return None
    progress.report("🧠 Looking at your project's knowledge graph...")
    answer, _ = await draft_and_verify(
        question,
        evidence,
        lambda fb: file_chain.ainvoke(
            {"evidence": evidence, "question": question, "feedback": f"\n\n{fb}" if fb else ""}
        ),
    )
    if not answer:
        answer = prompts.PROJECT_FACTS + "\n".join(f"- {line}" for line in evidence.splitlines())
    return Result(question, "project", answer, paths)


async def search_question(session: memory.Session, question: str) -> Result:
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


MAX_BTW_FILES = 2  # workspace files read for one /btw question
BTW_FILE_CHARS = 3000  # per file; the model has 8192 tokens

file_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", prompts.SYSTEM + "\n\n" + prompts.BTW_FILE_INSTRUCTIONS),
            ("human", "Workspace files:\n\n{evidence}\n\nQuestion: {question}{feedback}"),
        ]
    )
    | llm
    | StrOutputParser()
)


names_file = workspace.names  # the question names the file itself, not a word of it


async def workspace_evidence(question: str) -> tuple[str, list[str], bool]:
    """The workspace files the question names, as evidence, and whether it names
    them exactly; ("", [], False) if none."""
    try:
        entries = await workspace.list_entries()
    except Exception as exc:
        log.warning("cannot list the workspace: %s", exc)
        return "", [], False
    files = sorted(entries.files)
    named = [p for p in files if names_file(p, question)]
    exact = bool(named)
    named = named or [p for p in files if workspace.mentioned(p, question)]
    blocks, paths = [], []
    for path in named[:MAX_BTW_FILES]:
        try:
            content = await workspace.read(path)
        except Exception as exc:
            log.info("could not read %s: %s", path, exc)
            continue
        blocks.append(f"--- {path} ---\n{content[:BTW_FILE_CHARS]}")
        paths.append(path)
    return "\n\n".join(blocks), paths, exact


async def answer_from_docs(question: str) -> Result | None:
    fresh = memory.Session()  # a side question is not a follow-up: search it alone
    progress.report("🔎 Searching the HYPER-AI documentation...")
    if docs := await rag.retrieve(fresh, question):
        evidence = rag.evidence(docs)
        answer, _ = await draft_and_verify(
            question, evidence, lambda fb: rag.draft(fresh, question, evidence, fb)
        )
        if answer:
            return Result(question, "docs", answer, rag.sources(docs))
    return None


async def answer_from_files(question: str, evidence: str, paths: list[str]) -> Result | None:
    if not evidence:
        return None
    progress.report("🗂️ Reading " + ", ".join(paths) + "...")
    answer, _ = await draft_and_verify(
        question,
        evidence,
        lambda fb: file_chain.ainvoke(
            {"evidence": evidence, "question": question, "feedback": f"\n\n{fb}" if fb else ""}
        ),
    )
    return Result(question, "file", answer, paths) if answer else None


async def search_btw(question: str) -> Result:
    """/btw: the documentation and the workspace files the question names, then the web.

    A question that names a file exactly ("what image does x.yaml use?") reads the
    file first; otherwise the documentation comes first. Unlike question_worker it
    accepts any topic.
    """
    evidence, paths, exact = await workspace_evidence(question)
    def from_docs() -> Awaitable[Result | None]:
        return answer_from_docs(question)

    def from_files() -> Awaitable[Result | None]:
        return answer_from_files(question, evidence, paths)

    for step in (from_files, from_docs) if exact else (from_docs, from_files):
        if result := await step():
            return result

    progress.extend(
        estimates.estimate("web"), "🌐 Not in the documentation or your files - searching the web"
    )
    found = await web.gather(question)
    if not found.text:
        return Result(question, "none")
    answer, covered = await draft_and_verify(
        question, found.text, lambda fb: web.draft(question, found.text, fb, general=True)
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


async def validation_errors(content: str, kind: str) -> list[str]:
    """The ide-backend's validator, or the local DSL check when it is unreachable."""
    report = await workspace.validate_content(content)
    if report is None:
        return dsl.check(content, kind)
    errors = describe_errors(report)
    if not errors and report.get("type") not in (None, kind):
        errors = [f"this is a {report['type']} application, but a {kind} one was requested"]
    return errors


# Words in a request that allow changing a field whose path does not contain them.
FIELD_WORDS = {
    "image": {"uri", "tag", "container"}, "version": {"tag"}, "tag": {"uri"},
    "port": {"ports", "network"}, "ports": {"port", "network"}, "expose": {"ports", "port"},
    "memory": {"resources"}, "cpu": {"resources"}, "storage": {"resources"},
    "rename": {"name"}, "name": {"metadata"}, "describe": {"description"},
}
GENERIC_KEY_PARTS = {"specs", "spec", "application", "profile"}


def unrequested_changes(original: str, content: str, request: str) -> list[str]:
    """Fields an edit changed that the request never mentions: the model rewrote them
    (seen: "set the port" also renamed the app, raised the memory, and changed the
    --port argument instead of the network port)."""
    asked = set(re.findall(r"[a-z]+", request.lower()))
    asked |= {w for word in list(asked) for w in FIELD_WORDS.get(word, ())}
    problems = []
    for line in confirm.changes(original, content):
        if line.startswith(("... and", "no field changes", "the whole file")):
            continue
        key = re.sub(r"^(remove|add) ", "", line).split(":")[0].split(" (")[0].strip()
        parts = set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", key).lower()))
        if not (parts - GENERIC_KEY_PARTS) & asked:
            problems.append(
                f"{key} was changed, but the request does not ask for that - keep it exactly "
                "as in the CURRENT FILE"
            )
    return problems


async def file_problems(content: str, kind: str, request: str, original: str | None) -> list[str]:
    """① the validator, ② for an edit: no field changed that was not asked for,
    ③ an LLM check that the file does what was asked."""
    if errors := await validation_errors(content, kind):
        return errors
    if original is not None and (extra := unrequested_changes(original, content, request)):
        return extra
    progress.report("🧐 Checking it matches your request...")
    verdict = await verifier.check_file(request, content, original)
    return verdict.problems


# Simple field changes are made in code, on the exact line: precise, and no LLM call.
# request word -> (value pattern, YAML block the field is in, field)
QUICK_FIELDS = {
    "port": (r"\bport\b(?:\s+[\w-]+){0,4}?\s+(?:to|=|:)\s*(\d{1,5})\b", "ports", "port"),
    "memory": (r"\bmemory\b(?:\s+[\w-]+){0,4}?\s+(?:to|=|:)\s*(\d+(?:\.\d+)?(?:[KMGTE]i?)?)\b",
               "resources", "memory"),
    "cpu": (r"\bcpu\b(?:\s+[\w-]+){0,4}?\s+(?:to|=|:)\s*(\d+m|\d+(?:\.\d+)?)\b", "resources", "cpu"),
    "storage": (r"\bstorage\b(?:\s+[\w-]+){0,4}?\s+(?:to|=|:)\s*(\d+(?:\.\d+)?(?:[KMGTE]i?)?)\b",
                "resources", "storage"),
}


def _set_field(text: str, block: str, field_name: str, value: str) -> str | None:
    """`text` with the first `field_name:` inside `block:` set to `value`, or None."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if re.match(rf"^\s*{block}:\s*$", l)), None)
    if start is None:
        return None
    indent = len(lines[start]) - len(lines[start].lstrip())
    for i in range(start + 1, len(lines)):
        line = lines[i].rstrip("\r\n")
        own = len(line) - len(line.lstrip())
        if line.strip() and own <= indent and not line.lstrip().startswith("-"):
            return None  # left the block
        m = re.match(rf"^(\s*(?:-\s*)?{field_name}:\s*)(['\"]?)([^'\"#]*?)\2(\s*(?:#.*)?)$", line)
        if m:
            ending = lines[i][len(line):]
            lines[i] = f"{m.group(1)}{m.group(2)}{value}{m.group(2)}{m.group(4)}{ending}"
            return "".join(lines)
    return None


def quick_edit(original: str, request: str) -> str | None:
    """The edited file when every part of the request is a simple field change
    ("set the port to 8081 and the memory to 2Gi"); None to let the model do it."""
    content, fields = original, set()
    for clause in [c for c in re.split(r"\band\b|,|;", request) if c.strip()]:
        for word, (pattern, block, field_name) in QUICK_FIELDS.items():
            if m := re.search(pattern, clause, re.I):
                content = _set_field(content, block, field_name, m.group(1))
                if content is None:
                    return None
                fields.add(field_name)
                break
        else:
            return None  # a part we cannot do exactly
    if content == original:
        return original  # already has these values
    changed = confirm.changes(original, content)
    if any(not re.sub(r"\[\d+\]", "", line.split(":")[0]).endswith(tuple(fields)) for line in changed):
        return None
    return content


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
    if original is not None and (quick := quick_edit(original, task.request)) is not None:
        progress.report("✍️ Changing the field directly...")
        if quick == original:
            return quick, [], kind
        progress.report("🧪 Validating with the HYPER-AI rules...")
        return quick, await validation_errors(quick, kind), kind
    content, problems, first_problems = "", [], []
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
        if attempt == 1:
            first_problems = problems
        if not problems:
            break
    if original is not None and same_document(content, original) and first_problems:
        # "Fixed" by putting the old values back ("memory: banana" -> unchanged): the
        # change could not be made. Report why, never "already has those values".
        return original, first_problems, kind
    return content, problems, kind


def same_document(a: str, b: str) -> bool:
    try:
        return yaml.safe_load(a) == yaml.safe_load(b)
    except yaml.YAMLError:
        return a == b


def written(verb: str, path: str, kind: str, problems: list[str]) -> str:
    if not problems and workspace.backend_available():
        return f"✅ {verb} {path} - a {kind} application profile that passes HYPER-AI validation."
    if not problems:  # only the local check ran: say so, don't claim the IDE's validation
        return f"✅ {verb} {path} - a {kind} application profile. {prompts.LOCAL_CHECK_ONLY}"
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
    if task.op in ("create_file", "edit_file") and not path.lower().endswith(workspace.YAML_SUFFIXES):
        return ActionResult(prompts.NOT_YAML.format(path=path))

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
    if same_document(content, original):
        if problems:
            listed = "\n".join(f"  - {p}" for p in problems)
            return ActionResult(prompts.CANNOT_APPLY.format(path=found, problems=listed), path=found)
        return ActionResult(prompts.NOTHING_TO_CHANGE.format(path=found), path=found)
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

"""The project knowledge graph: what Hyperion knows about this project, across sessions.

Two parts, both kept in the Redis container (fallback: in this process):

1. The workspace graph. Every YAML file becomes a node with the facts parsed from
   it (app name, kind, images, ports, resources), linked to its folder, its images
   and its ports - so two apps on the same image or port are connected. It is built
   when the project is first seen (startup), re-synced at the start of every new
   chat session, and updated at once whenever the agent creates, edits or deletes a
   file. The planner and the question workers read it instead of guessing from
   file names, so answers about "my apps" are precise.

2. Learned answers. Every verified documentation or web answer is remembered. When
   the same question comes back - in any session - it is answered from here in
   about a second, without searching or re-verifying. Web answers keep their red
   marking and expire after WEB_TTL; documentation answers expire when the
   documentation index is rebuilt.

One process serves the agent, so the working copy lives in memory and every change
is written through to Redis.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field

import memory
import yaml

import dsl
import rag
import workspace
from llm import embeddings

log = logging.getLogger("hyperion.knowledge")

KEY_PREFIX = "hyperion:kg:"
REDIS_URL = os.environ.get("REDIS_URL", "")

MAX_FILES = 300  # workspace files kept in the graph
READ_CONCURRENCY = 8  # parallel file reads during a sync
MIN_SYNC_INTERVAL = 30  # seconds; new sessions in a burst share one sync
MAX_ANSWERS = 500  # learned answers kept; the least used are dropped first
WEB_TTL = 7 * 24 * 3600  # web answers can go out of date
CACHED_KINDS = ("docs", "web")  # verified answers only; "unverified" is never stored
# A question counts as "the same" only if it means the same (embeddings) AND uses the
# same words: "What is a DeviceNode?" and "What is a NativeNode?" embed very closely.
SAME_MEANING = 0.9  # cosine similarity
SAME_WORDS = 0.75  # Jaccard overlap of the stemmed content words
# Questions about the user's own project ("which of my apps use port 80?").
PROJECT_WORDS = re.compile(
    r"\b(my|our)\s+(apps?|applications?|files?|workspace|project)\b"
    r"|\b(which|what)\s+(of\s+)?(my\s+|our\s+|the\s+)?(apps?|files?)\b|\bworkspace\b",
    re.I,
)


@dataclass
class FileFacts:
    path: str
    digest: str
    kind: str | None = None  # native | device | None (not an application profile)
    name: str | None = None
    images: list[str] = field(default_factory=list)
    ports: list[str] = field(default_factory=list)
    resources: dict[str, str] = field(default_factory=dict)  # cpu, memory, storage, replicas
    updated: float = 0.0


@dataclass
class Answer:
    question: str
    words: list[str]
    vector: list[float]
    kind: str  # a workers.Result kind: docs | web
    text: str
    sources: list[str]
    index: str  # the documentation index it was verified against
    created: float = 0.0
    hits: int = 0


_files: dict[str, FileFacts] = {}
_answers: dict[str, Answer] = {}
_meta: dict = {}
_redis = memory.redis_client(REDIS_URL)  # survives a Redis restart, see memory.py
_sync_lock = asyncio.Lock()
_last_sync = 0.0


def _index_version() -> str:
    try:
        return hashlib.sha1(rag.INDEX.read_bytes()).hexdigest()[:12]
    except OSError:
        return "none"


INDEX_VERSION = _index_version()


# --- storage -----------------------------------------------------------------------


async def _store(table: str, key: str, value: dict | None) -> None:
    if not _redis:
        return
    try:
        if value is None:
            await _redis.hdel(KEY_PREFIX + table, key)
        else:
            await _redis.hset(KEY_PREFIX + table, key, json.dumps(value))
    except Exception as exc:
        log.warning("redis unavailable (%s), the knowledge graph stays in memory", exc)


async def load() -> None:
    """Read the graph saved by earlier runs; build it if this project is new."""
    if _redis:
        try:
            for raw in (await _redis.hgetall(KEY_PREFIX + "files")).values():
                facts = FileFacts(**json.loads(raw))
                _files[facts.path] = facts
            for key, raw in (await _redis.hgetall(KEY_PREFIX + "answers")).items():
                _answers[key] = Answer(**json.loads(raw))
            _meta.update(json.loads(await _redis.get(KEY_PREFIX + "meta") or "{}"))
        except Exception:
            log.exception("could not load the knowledge graph from redis")
    for key in [k for k, a in _answers.items() if not learnable(a.question)]:
        del _answers[key]  # learned under older, looser rules (follow-ups, "today")
        await _store("answers", key, None)
    if not _meta.get("created"):
        _meta["created"] = time.time()
        log.info("new project: creating the knowledge graph")
    log.info("knowledge graph: %d files, %d learned answers", len(_files), len(_answers))
    await sync("startup")


async def _save_meta() -> None:
    if _redis:
        try:
            await _redis.set(KEY_PREFIX + "meta", json.dumps(_meta))
        except Exception as exc:
            log.warning("redis unavailable (%s), the knowledge graph stays in memory", exc)


def _count(name: str) -> None:
    _meta[name] = _meta.get(name, 0) + 1


# --- the workspace graph ---------------------------------------------------------------


def _flatten(node: object, prefix: str = "") -> dict[str, object]:
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            out |= _flatten(value, f"{prefix}.{key}" if prefix else str(key))
        return out
    if isinstance(node, list):
        out = {}
        for i, value in enumerate(node):
            out |= _flatten(value, f"{prefix}[{i}]")
        return out
    return {prefix: node}


def extract(path: str, content: str) -> FileFacts:
    """The facts of one file. Works for native and device profiles alike: it looks
    at field names (uri/image, port, memory...), not at one fixed layout."""
    facts = FileFacts(path, hashlib.sha1(content.encode()).hexdigest(), updated=time.time())
    try:
        doc = yaml.safe_load(content)
    except yaml.YAMLError:
        return facts
    facts.kind = dsl.kind_of(doc)
    flat = _flatten(doc)
    for key, value in flat.items():
        last = re.sub(r"\[\d+\]$", "", key.rsplit(".", 1)[-1])
        if value is None or isinstance(value, bool):
            continue
        if key.endswith("metadata.name") and not facts.name:
            facts.name = str(value)
        elif last in ("uri", "image") and isinstance(value, str):
            tag = flat.get(key.rsplit(".", 1)[0] + ".tag")
            facts.images.append(f"{value}:{tag}" if tag and ":" not in value else value)
        elif last in ("port", "containerPort", "targetPort", "hostPort"):
            facts.ports.append(str(value))
        elif last in ("cpu", "memory", "storage", "replicas") and last not in facts.resources:
            facts.resources[last] = str(value)
    facts.images = list(dict.fromkeys(facts.images))
    facts.ports = list(dict.fromkeys(facts.ports))
    return facts


async def learn_file(path: str, content: str) -> None:
    facts = extract(path, content)
    if (old := _files.get(path)) and old.digest == facts.digest:
        return
    _files[path] = facts
    await _store("files", path, asdict(facts))


async def forget(path: str, *, folder: bool = False) -> None:
    gone = [p for p in _files if p == path or (folder and p.startswith(path.rstrip("/") + "/"))]
    for p in gone:
        del _files[p]
        await _store("files", p, None)


async def apply(action: dict) -> None:
    """Keep the graph in step with an action the agent sent to the IDE."""
    kind, path = action.get("action"), action.get("path", "")
    try:
        if kind in ("create_file", "edit_file") and path.lower().endswith(workspace.YAML_SUFFIXES):
            await learn_file(path, action.get("content", ""))
        elif kind == "delete_file":
            await forget(path)
        elif kind == "delete_folder":
            await forget(path, folder=True)
    except Exception:
        log.exception("could not update the knowledge graph for %s", action)


async def sync(reason: str) -> None:
    """Re-read the workspace: files changed in the IDE by hand are picked up too."""
    global _last_sync
    if _sync_lock.locked():
        return  # one sync at a time; the running one will see the same workspace
    async with _sync_lock:
        started = time.perf_counter()
        try:
            entries = await workspace.list_entries()
        except Exception as exc:
            log.info("knowledge sync (%s) skipped: workspace unavailable (%s)", reason, exc)
            return
        paths = sorted(p for p in entries.files if p.lower().endswith(workspace.YAML_SUFFIXES))
        for gone in set(_files) - set(paths):
            await forget(gone)
        limit = asyncio.Semaphore(READ_CONCURRENCY)

        async def read(path: str) -> None:
            async with limit:
                try:
                    await learn_file(path, await workspace.read(path))
                except Exception as exc:
                    log.info("knowledge sync: cannot read %s (%s)", path, exc)

        await asyncio.gather(*(read(p) for p in paths[:MAX_FILES]))
        _last_sync = time.monotonic()
        _meta["last_sync"] = time.time()
        await _save_meta()
        log.info("knowledge sync (%s): %d files in %.1fs", reason, len(_files),
                 time.perf_counter() - started)


async def ensure_fresh(max_age: float = MIN_SYNC_INTERVAL) -> None:
    """Re-read the workspace before using the graph if the last sync is older than
    `max_age` seconds: a file edited by hand in the IDE is then seen in the same
    session (a sync of a few dozen files takes ~0.1 s)."""
    if time.monotonic() - _last_sync >= max_age:
        await sync("before use")


def on_new_session() -> None:
    """A new chat session: count it and refresh the workspace graph in the background."""
    _count("sessions")
    if time.monotonic() - _last_sync >= MIN_SYNC_INTERVAL:
        asyncio.get_running_loop().create_task(sync("new session"))
    else:
        asyncio.get_running_loop().create_task(_save_meta())


def describe(facts: FileFacts) -> str:
    """'demo/web.yaml: native app "web", image nginx:latest, port 80, memory 1Gi'."""
    if facts.kind is None and not facts.images:
        return f"{facts.path}: not an application profile"
    parts = [f"{facts.kind or 'unknown'} app" + (f' "{facts.name}"' if facts.name else "")]
    if facts.images:
        parts.append("image " + ", ".join(facts.images))
    if facts.ports:
        parts.append(("ports " if len(facts.ports) > 1 else "port ") + ", ".join(facts.ports))
    parts += [f"{k} {v}" for k, v in facts.resources.items()]
    return f"{facts.path}: " + ", ".join(parts)


def known_files() -> set[str]:
    return set(_files)


def describe_files(paths: list[str]) -> list[str]:
    """Each path with what the graph knows about it (the bare path if nothing)."""
    return [describe(_files[p]) if p in _files else p for p in paths]


def _short_image(image: str) -> str:
    return image.rsplit("/", 1)[-1].split(":")[0].lower()


def _says(word: str, text: str) -> bool:
    return len(word) >= 3 and re.search(rf"(?<![\w-]){re.escape(word.lower())}(?![\w-])", text.lower()) is not None


def files_named_by(text: str) -> list[str]:
    """Files the text identifies through the graph: by app name, or by an image only
    one file uses ("the nginx app")."""
    by_name = [p for p, f in _files.items() if f.name and _says(f.name, text)]
    if by_name:
        return sorted(by_name)
    images: dict[str, list[str]] = {}
    for p, f in _files.items():
        for image in {_short_image(i) for i in f.images}:
            images.setdefault(image, []).append(p)
    return sorted(ps[0] for image, ps in images.items() if len(ps) == 1 and _says(image, text))


def identifies(path: str, text: str) -> bool:
    """Whether the user's words point at this file through the graph."""
    return path in files_named_by(text)


def files_by_app_name(text: str) -> list[str]:
    return sorted(p for p, f in _files.items() if f.name and _says(f.name, text))


def files_by_image(text: str) -> list[str]:
    """Every file whose image the text names ("the nginx app"), shared or not."""
    return sorted(
        p for p, f in _files.items() if any(_says(_short_image(i), text) for i in f.images)
    )


def project_evidence(question: str) -> tuple[str, list[str]]:
    """The graph's facts for a question about the user's own project, or ("", [])."""
    if not _files:
        return "", []
    named = files_named_by(question)
    # An image alone ("how do I deploy nginx?") is usually a documentation question.
    by_name = any(_files[p].name and _says(_files[p].name, question) for p in named)
    if not by_name and not PROJECT_WORDS.search(question):
        return "", []
    paths = named or sorted(_files)[:60]
    return "\n".join(describe_files(paths)), paths


def lookup(question: str) -> tuple[str, list[str]] | None:
    """Structured questions about the project, answered in code - exact, no LLM:
    which apps use an image, expose a port, or are device / native apps."""
    if not PROJECT_WORDS.search(question) or not _files:
        return None
    apps = [f for f in _files.values() if f.kind is not None]
    tests, what = [], []
    if port := re.search(r"\bports?\s+(\d{1,5})\b", question, re.I):
        tests.append(lambda f, p=port.group(1): p in f.ports)
        what.append(f"port {port.group(1)}")
    images = {_short_image(i) for f in apps for i in f.images}
    if named := sorted(i for i in images if _says(i, question)):
        tests.append(lambda f: any(_short_image(i) in named for i in f.images))
        what.append("image " + " / ".join(named))
    if kind := re.search(r"\b(device|native)\b", question, re.I):
        tests.append(lambda f, k=kind.group(1).lower(): f.kind == k)
        what.append(f"{kind.group(1).lower()} apps")
    if not tests:
        return None
    found = [f for f in sorted(apps, key=lambda f: f.path) if all(t(f) for t in tests)]
    if not found:
        return f"None of the {len(apps)} apps in your workspace match ({', '.join(what)}).", []
    lines = "\n".join(f"- {describe(f)}" for f in found)
    return f"{len(found)} app(s) in your workspace match ({', '.join(what)}):\n{lines}", [
        f.path for f in found
    ]


def graph() -> dict:
    """Nodes and edges, for GET /graph and /graph."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def node(node_id: str, kind: str, label: str) -> str:
        nodes.setdefault(node_id, {"id": node_id, "type": kind, "label": label})
        return node_id

    for f in _files.values():
        file_id = node(f"file:{f.path}", "file", f.path)
        if "/" in f.path:
            folder = f.path.rsplit("/", 1)[0]
            edges.append({"from": node(f"folder:{folder}", "folder", folder), "rel": "contains", "to": file_id})
        if f.kind is None:
            continue
        app_id = node(f"app:{f.path}", f"{f.kind}_app", f.name or f.path)
        edges.append({"from": file_id, "rel": "defines", "to": app_id})
        for image in f.images:
            edges.append({"from": app_id, "rel": "uses_image", "to": node(f"image:{image}", "image", image)})
        for port in f.ports:
            edges.append({"from": app_id, "rel": "exposes", "to": node(f"port:{port}", "port", port)})
    return {"nodes": list(nodes.values()), "edges": edges, "stats": stats()}


def stats() -> dict:
    return {
        "created": _meta.get("created"),
        "last_sync": _meta.get("last_sync"),
        "sessions": _meta.get("sessions", 0),
        "files": len(_files),
        "apps": sum(f.kind is not None for f in _files.values()),
        "learned_answers": len(_answers),
        "answers_reused": _meta.get("answers_reused", 0),
    }


def overview() -> str:
    """Plain-text overview for the /graph command."""
    s = stats()
    lines = [
        "🧠 Project knowledge graph",
        f"- {s['files']} file(s), {s['apps']} application(s); {s['learned_answers']} learned "
        f"answer(s), reused {s['answers_reused']} time(s); {s['sessions']} chat session(s).",
    ]
    if _files:
        lines.append("\nWorkspace:")
        lines += [f"- {line}" for line in describe_files(sorted(_files))]
        shared: dict[str, list[str]] = {}
        for f in _files.values():
            for image in f.images:
                shared.setdefault(f"image {image}", []).append(f.path)
            for port in f.ports:
                shared.setdefault(f"port {port}", []).append(f.path)
        links = [f"- {what}: {', '.join(sorted(ps))}" for what, ps in sorted(shared.items()) if len(ps) > 1]
        if links:
            lines.append("\nShared by several apps:")
            lines += links
    if _answers:
        top = sorted(_answers.values(), key=lambda a: -a.hits)[:5]
        lines.append("\nLearned answers (most reused):")
        lines += [f"- {a.question} ({'🔴 web' if a.kind == 'web' else 'documentation'}, "
                  f"reused {a.hits}x)" for a in top]
    return "\n".join(lines)


# --- learned answers -----------------------------------------------------------------


def _key(question: str) -> str:
    return hashlib.sha1(" ".join(question.lower().split()).encode()).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def same_words(a: list[str], b: list[str]) -> bool:
    sa, sb = set(a), set(b)
    return bool(sa and sb) and len(sa & sb) / len(sa | sb) >= SAME_WORDS


def _expired(answer: Answer) -> bool:
    if answer.kind == "web":
        return time.time() - answer.created > WEB_TTL
    return answer.index != INDEX_VERSION


async def _vector(question: str) -> list[float] | None:
    try:
        return [round(x, 5) for x in await embeddings.aembed_query(question)]
    except Exception as exc:
        log.info("no embedding for the answer cache (%s)", exc)
        return None


# Only a self-contained question is worth remembering: not a follow-up that depends on
# its conversation ("and how is it registered?"), not a statement ("my name is X"),
# and not something whose answer changes by the day ("today's date", "latest news").
QUESTION_START = re.compile(
    r"^\s*(what|who|whom|whose|which|where|when|why|how|is|are|was|were|does|do|did|can|"
    r"could|should|would|will|list|explain|describe|define|tell me)\b", re.I)
FOLLOW_UP = re.compile(
    r"^\s*(and|but|so|also|then|or|what about|how about)\b"
    r"|\b(it|its|they|them|their|this|that|these|those|he|she|him|her)\b", re.I)
TIME_SENSITIVE = re.compile(
    r"\b(today|tonight|now|current(ly)?|latest|recent(ly)?|news|weather|price|prices|"
    r"yesterday|tomorrow|this (week|month|year)|date|time)\b", re.I)


def learnable(question: str) -> bool:
    q = question.strip()
    if not (q.endswith("?") or QUESTION_START.match(q)):
        return False
    if TIME_SENSITIVE.search(q):
        return False
    return not FOLLOW_UP.search(q)


async def recall(question: str) -> Answer | None:
    """A verified answer to the same question, learned in this or an earlier session."""
    if not _answers or not learnable(question):
        return None
    words = rag.tokenize(question)
    exact = _answers.get(_key(question))
    best = exact if exact and not _expired(exact) else None
    if best is None:
        candidates = [a for a in _answers.values() if same_words(words, a.words) and not _expired(a)]
        if not candidates or (vector := await _vector(question)) is None:
            return None
        scored = max(((_cosine(vector, a.vector), a) for a in candidates), key=lambda s: s[0])
        if scored[0] < SAME_MEANING:
            return None
        best = scored[1]
    best.hits += 1
    _count("answers_reused")
    log.info("answered %r from the knowledge graph (learned for %r)", question, best.question)
    await _store("answers", _key(best.question), asdict(best))
    await _save_meta()
    return best


async def remember(question: str, kind: str, text: str, sources: list[str]) -> None:
    if kind not in CACHED_KINDS or not text or not learnable(question):
        return
    if (vector := await _vector(question)) is None:
        return
    answer = Answer(question, rag.tokenize(question), vector, kind, text, sources,
                    INDEX_VERSION, created=time.time())
    _answers[_key(question)] = answer
    await _store("answers", _key(question), asdict(answer))
    if len(_answers) > MAX_ANSWERS:  # drop the least used, oldest first
        for key, _ in sorted(_answers.items(), key=lambda kv: (kv[1].hits, kv[1].created))[
            : len(_answers) - MAX_ANSWERS
        ]:
            del _answers[key]
            await _store("answers", key, None)

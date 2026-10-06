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

import memory
import prompts
import rag
import verifier
import web

log = logging.getLogger("hyperion.workers")

ATTEMPTS = 2  # first draft + one retry with the verifier's feedback
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
        answer = (await write(feedback)).strip()
        if rag.not_covered(answer):
            return None, False
        verdict = await verifier.check_answer(question, evidence, answer)
        if verdict.ok:
            return answer, True
        log.info("attempt %d rejected for %r: %s", attempt, question, verdict.problems)
        feedback = prompts.RETRY_FEEDBACK.format(
            problems="\n".join(f"- {p}" for p in verdict.problems), previous=answer
        )
    return remove_flagged(answer, verdict.problems) or None, True


async def question_worker(session: memory.Session, question: str) -> Result:
    # 1. The HYPER-AI documentation.
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

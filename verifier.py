"""Checks a worker's output against the evidence it was given.

Cheap code checks run first; the LLM verifier only runs when they pass. The
verifier fails open: if it cannot produce a verdict, the answer goes through
(it was already generated from the evidence only).
"""

import json
import logging
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

import prompts
from llm import llm_json

log = logging.getLogger("hyperion.verifier")

MIN_ANSWER_CHARS = 2


@dataclass
class Verdict:
    ok: bool
    problems: list[str] = field(default_factory=list)


def code_checks(answer: str) -> Verdict:
    text = answer.strip()
    if len(text) < MIN_ANSWER_CHARS:
        return Verdict(False, ["the answer is empty"])
    if prompts.NOT_COVERED in text and text != prompts.NOT_COVERED:
        return Verdict(False, [f"the answer mixes content with the {prompts.NOT_COVERED} marker"])
    return Verdict(True)


async def check_answer(question: str, evidence: str, answer: str) -> Verdict:
    verdict = code_checks(answer)
    if not verdict.ok:
        return verdict
    messages = [
        SystemMessage(prompts.VERIFIER),
        HumanMessage(f"EVIDENCE:\n{evidence}\n\nQUESTION: {question}\n\nANSWER:\n{answer}"),
    ]
    raw = ""
    try:
        raw = (await llm_json.ainvoke(messages)).text
        data = json.loads(raw)
        problems = [str(p) for p in data.get("problems", []) if str(p).strip()]
        ok = bool(data.get("supported", True))
    except Exception as exc:
        log.warning("verifier unavailable (%r), output %r - accepting answer", exc, raw)
        return Verdict(True)
    log.info("verdict supported=%s problems=%s", ok, problems)
    return Verdict(ok, problems if not ok else [])

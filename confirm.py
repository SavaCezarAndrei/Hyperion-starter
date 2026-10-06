"""Human in the loop: state-changing actions wait for the user's yes/no.

Deletes, overwrites and edits (edit_file replaces the whole file) are not sent to
the IDE right away. They are stored in session.pending with a precise summary of
what will change, and the next message decides - in code, no LLM involved:
yes -> the stored actions are sent; no -> cancelled; anything else -> the pending
change is dropped and the new message is handled normally.
"""

import re
import time
from collections.abc import AsyncIterator

import yaml

import memory
import prompts

PENDING_TTL = 10 * 60  # seconds; an old confirmation is not applied by a late "yes"
MAX_CHANGES_SHOWN = 8

YES = {
    "yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm", "confirmed",
    "do it", "go ahead", "apply", "proceed", "yes please", "please do", "da",
}
NO = {"no", "n", "nope", "cancel", "stop", "abort", "don't", "dont", "do not", "nu", "no thanks"}


def answer(text: str) -> str | None:
    """'yes', 'no' or None (the message is about something else)."""
    words = re.sub(r"[^\w' ]", " ", text.lower()).split()
    phrase = " ".join(words)
    if phrase in YES or (words and words[0] in YES and len(words) <= 3):
        return "yes"
    if phrase in NO or (words and words[0] in NO and len(words) <= 3):
        return "no"
    return None


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


def changes(before: str, after: str) -> list[str]:
    """What an edit changes, field by field: 'specs.network.ports[0].port: 80 -> 8080'."""
    try:
        old = _flatten(yaml.safe_load(before))
        new = _flatten(yaml.safe_load(after))
    except yaml.YAMLError:
        return ["the whole file is rewritten"]
    lines = []
    for key in sorted(old.keys() | new.keys()):
        short = key.removeprefix("applicationProfile.")
        if key not in new:
            lines.append(f"remove {short} ({old[key]})")
        elif key not in old:
            lines.append(f"add {short}: {new[key]}")
        elif old[key] != new[key]:
            lines.append(f"{short}: {old[key]} -> {new[key]}")
    if len(lines) > MAX_CHANGES_SHOWN:
        lines = lines[:MAX_CHANGES_SHOWN] + [f"... and {len(lines) - MAX_CHANGES_SHOWN} more"]
    return lines or ["no field changes (formatting only)"]


def ask(items: list[dict]) -> str:
    """The question shown to the user for the stored items."""
    listed = "\n".join(item["question"] for item in items)
    return prompts.CONFIRM_ASK.format(changes=listed)


def store(session: memory.Session, items: list[dict]) -> None:
    session.pending = {"items": items, "created": time.time()}


async def resolve(session: memory.Session, text: str) -> AsyncIterator[dict]:
    """Handle the reply to a pending confirmation. Yields nothing if the message is
    about something else (the caller then handles it normally)."""
    pending, session.pending = session.pending, None
    expired = time.time() - pending.get("created", 0) > PENDING_TTL
    decision = answer(text)
    if decision is None:
        return  # pending change dropped; the new message is handled normally
    if decision == "no":
        yield {"response": prompts.CONFIRM_CANCELLED}
        return
    if expired:
        yield {"response": prompts.CONFIRM_EXPIRED}
        return
    for i, item in enumerate(pending["items"]):
        for action in item["actions"]:
            yield action
        if item.get("path"):
            session.last_file = item["path"]
        yield {"response": item["done"] + ("\n" if i < len(pending["items"]) - 1 else "")}

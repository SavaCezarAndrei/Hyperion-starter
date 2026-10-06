"""Per-user session state, keyed by the user_id the IDE sends.

Stored in the Redis container (`memory` service in docker-compose) when REDIS_URL is
set; otherwise kept in this process. The fallback matters: the evaluators run our
image alone, without the memory container.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field

import redis.asyncio as redis
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

log = logging.getLogger("hyperion.memory")

MAX_TURNS = 8  # turns kept in the prompt; the model only has 8192 tokens
MAX_STORED_TURNS = 50  # older turns are dropped from storage
SESSION_TTL = 24 * 3600  # seconds; idle sessions are deleted after a day
KEY_PREFIX = "hyperion:session:"

REDIS_URL = os.environ.get("REDIS_URL", "")


@dataclass
class Session:
    user_id: str = ""
    messages: list[dict] = field(default_factory=list)  # [{"role": "user"|"assistant", "content": str}]
    last_file: str | None = None  # resolves "it" / "that file" in follow-ups
    pending: dict | None = None  # action waiting for the user's yes/no


_local: dict[str, Session] = {}
_redis = redis.from_url(REDIS_URL, decode_responses=True) if REDIS_URL else None
log.info("session store: %s", "redis" if _redis else "in-process")


async def get(user_id: str) -> Session:
    if _redis:
        try:
            raw = await _redis.get(KEY_PREFIX + user_id)
            return Session(**json.loads(raw)) if raw else Session(user_id=user_id)
        except Exception:
            log.exception("redis unavailable, using in-process memory")
    return _local.setdefault(user_id, Session(user_id=user_id))


async def save(session: Session) -> None:
    session.messages = session.messages[-MAX_STORED_TURNS * 2 :]
    if _redis:
        try:
            await _redis.set(
                KEY_PREFIX + session.user_id, json.dumps(asdict(session)), ex=SESSION_TTL
            )
            return
        except Exception:
            log.exception("redis unavailable, using in-process memory")
    _local[session.user_id] = session


def history(session: Session, max_turns: int = MAX_TURNS) -> list[BaseMessage]:
    """The most recent turns, ready to put in a prompt."""
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in session.messages[-max_turns * 2 :]
    ]


def add_turn(session: Session, user_text: str, assistant_text: str) -> None:
    session.messages.append({"role": "user", "content": user_text})
    session.messages.append(
        {"role": "assistant", "content": assistant_text or "(performed an action)"}
    )

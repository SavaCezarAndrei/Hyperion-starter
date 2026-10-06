"""Per-user session state, keyed by the user_id the IDE sends."""

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

MAX_TURNS = 8  # turns kept in the prompt; the model only has 8192 tokens


@dataclass
class Session:
    messages: list[BaseMessage] = field(default_factory=list)
    last_file: str | None = None  # resolves "it" / "that file" in follow-ups
    pending: dict | None = None  # action waiting for the user's yes/no


_sessions: dict[str, Session] = {}


def get(user_id: str) -> Session:
    return _sessions.setdefault(user_id, Session())


def history(session: Session, max_turns: int = MAX_TURNS) -> list[BaseMessage]:
    """The most recent turns, ready to put in a prompt."""
    return session.messages[-max_turns * 2 :]


def add_turn(session: Session, user_text: str, assistant_text: str) -> None:
    session.messages.append(HumanMessage(user_text))
    session.messages.append(AIMessage(assistant_text or "(performed an action)"))

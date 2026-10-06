"""The agent: turns one user message into a stream of events.

Every handler is an async generator yielding dicts, which main.py sends as SSE:
  {"response": "text chunk"}                         -> shown in the Hyperion panel
  {"action": "create_file", "path": ..., "content": ...} -> executed by the IDE
"""

import re
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

import confirm
import memory
import orchestrator
import progress
import prompts
import router
from llm import llm

Event = dict

BTW = "/btw"
MENTIONS_FILE = re.compile(r"\b[\w./-]+\.ya?ml\b", re.I)


def is_side_question(text: str) -> bool:
    """/btw questions are answered but never stored in the session (see main.py)."""
    return text.strip().lower().startswith(BTW)


async def handle(session: memory.Session, text: str) -> AsyncIterator[Event]:
    if is_side_question(text):
        async for event in side_answer(session, text.strip()[len(BTW) :].strip()):
            yield event
        return

    # Debug command: proves the action pipeline end to end without the LLM.
    if text.strip() == "/test-action":
        yield {"response": "Creating the folder `hyperion-test`…"}
        yield {"action": "create_folder", "path": "hyperion-test"}
        yield {"response": " Done — check the file explorer."}
        return

    # A change is waiting for yes/no: decided in code, before the router.
    if session.pending:
        replied = False
        async for event in confirm.resolve(session, text):
            replied = True
            yield event
        if replied:
            return
        yield {"response": prompts.CONFIRM_DROPPED}  # moved on: handle the new message

    progress.report("🧭 Understanding your request...")
    intent = await router.classify(session, text)
    if intent == "question" and MENTIONS_FILE.search(text):
        intent = "action"  # "what's inside app.yaml?" needs the file, not the docs
    if intent == "off_topic":
        yield {"response": prompts.REFUSAL}
    elif intent == "chitchat":
        async for event in chat(session, text):
            yield event
    elif intent == "question":
        async for event in orchestrator.answer_questions(session, text):
            yield event
    else:
        async for event in orchestrator.run_actions(session, text):
            yield event


async def side_answer(session: memory.Session, question: str) -> AsyncIterator[Event]:
    """Like Claude's /btw: a quick answer to anything, off the record.

    It sees the conversation for context but bypasses the router (so off-topic is
    allowed here, on explicit request) and is never written to memory.
    """
    if not question:
        yield {"response": prompts.BTW_USAGE}
        return
    messages = [
        SystemMessage(prompts.BTW_SYSTEM),
        *memory.history(session),
        HumanMessage(question),
    ]
    yield {"response": prompts.BTW_HEADER}
    started = False
    async for chunk in llm.astream(messages):
        text = chunk.text
        if not started:  # the model sometimes echoes the command: "/btw Canberra is..."
            text = text.lstrip()
            if text.lower().startswith(BTW):
                text = text[len(BTW) :].lstrip()
            started = bool(text)
        if text:
            yield {"response": text}


async def chat(session: memory.Session, text: str) -> AsyncIterator[Event]:
    """Plain LLM reply with the session history."""
    messages = [
        SystemMessage(prompts.SYSTEM),
        *memory.history(session),
        HumanMessage(text),
    ]
    async for chunk in llm.astream(messages):
        if chunk.text:
            yield {"response": chunk.text}

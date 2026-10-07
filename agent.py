"""The agent: turns one user message into a stream of events.

Every handler is an async generator yielding dicts, which main.py sends as SSE:
  {"response": "text chunk"}                         -> shown in the Hyperion panel
  {"action": "create_file", "path": ..., "content": ...} -> executed by the IDE
"""

import re
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

import confirm
import knowledge
import memory
import orchestrator
import progress
import prompts
import router
import supervisor
import workers
from llm import llm

Event = dict

BTW = "/btw"
MENTIONS_FILE = re.compile(r"\b[\w./-]+\.ya?ml\b", re.I)
# HTML/script markup is never part of a request; left in, it derails retrieval and
# the answer check ("<script>alert(1)</script> What is HYPER-AI?").
MARKUP = re.compile(r"<script\b.*?</script\s*>|<style\b.*?</style\s*>|</?[a-z][^<>]{0,200}>", re.I | re.S)
# Attempts to replace the assistant's instructions get the same refusal as off-topic.
INJECTION = re.compile(
    r"\b(ignore|disregard|forget|override)\b[^.?!]{0,40}\b(instructions|rules|prompt|guidelines)\b"
    r"|\byou are now\b|\bpretend (to be|you are)\b|\bact as (a|an)\b|\bjailbreak\b"
    r"|\b(print|show|reveal|repeat)\b[^.?!]{0,30}\b(system prompt|instructions)\b",
    re.I,
)
# "show it" / "delete that" right after working on a file: an action on that file.
ACT_ON_LAST_FILE = re.compile(
    r"^\s*(please\s+)?(show|open|read|display|print|view|validate|check|delete|remove)\s+"
    r"(me\s+)?(it|that|this)(\s+(file|one))?\s*(again)?\s*[.!?]*\s*$", re.I)


def strip_markup(text: str) -> str:
    return MARKUP.sub(" ", text).strip()


def is_side_question(text: str) -> bool:
    """/btw questions are answered but never stored in the session (see main.py)."""
    return text.strip().lower().startswith(BTW)


def off_the_record(text: str) -> bool:
    """Messages main.py does not store in the session: /btw and /graph."""
    return is_side_question(text) or text.strip().lower() == prompts.GRAPH_COMMAND


async def handle(session: memory.Session, text: str) -> AsyncIterator[Event]:
    text = strip_markup(text)
    if not text:
        yield {"response": prompts.EMPTY_MESSAGE}
        return
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

    if text.strip().lower() == prompts.GRAPH_COMMAND:
        yield {"response": knowledge.overview()}
        return

    # A change is waiting for yes/no: decided in code, before the router.
    if session.pending:
        replied = False
        # "no, in nginx.yaml": the same edit on another file, asked again.
        async for event in orchestrator.retarget(session, text):
            replied = True
            yield event
        if replied:
            return
        async for event in confirm.resolve(session, text):
            replied = True
            yield event
        if replied:
            return
        yield {"response": prompts.CONFIRM_DROPPED}  # moved on: handle the new message
    elif confirm.is_bare_answer(text):
        # A lone "yes"/"no" with nothing pending must never reach the router: it would
        # read the conversation ("delete X" ... "yes") and plan the change again.
        yield {"response": prompts.CONFIRM_NOTHING_PENDING}
        return

    if INJECTION.search(text):
        yield {"response": prompts.REFUSAL}
        return
    if session.last_file and ACT_ON_LAST_FILE.match(text):
        async for event in orchestrator.run_actions(session, text):
            yield event
        return

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

    It bypasses the router (so off-topic is allowed here, on explicit request) and is
    never written to memory. The answer comes from the documentation, the workspace
    files the question names, or the web - and web answers are marked 🔴.
    """
    if not question:
        yield {"response": prompts.BTW_USAGE}
        return
    yield {"response": prompts.BTW_HEADER}
    result = await supervisor.supervise(
        question,
        "question",
        lambda: workers.btw_worker(question),
        lambda: workers.Result(question, "none"),
    )
    yield {"response": orchestrator.render(result)}


async def chat(session: memory.Session, text: str) -> AsyncIterator[Event]:
    """Plain LLM reply with the session history."""
    messages = [
        SystemMessage(prompts.SYSTEM),
        *memory.history(session),
        HumanMessage(text),
    ]
    # Collected, not streamed: a chitchat reply is short, and it is checked first - asked
    # "what was my previous message?" with no history, the model quoted its instructions.
    reply = (await llm.ainvoke(messages)).text
    if leaks_instructions(reply):
        reply = prompts.CHAT_NO_INSTRUCTIONS
    yield {"response": reply}


def leaks_instructions(reply: str) -> bool:
    """The reply repeats a long stretch of the system prompt."""
    words = prompts.SYSTEM.lower().split()
    said = " ".join(reply.lower().split())
    return any(" ".join(words[i : i + 8]) in said for i in range(len(words) - 7))

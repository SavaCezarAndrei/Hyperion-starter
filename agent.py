"""The agent: turns one user message into a stream of events.

Every handler is an async generator yielding dicts, which main.py sends as SSE:
  {"response": "text chunk"}                         -> shown in the Hyperion panel
  {"action": "create_file", "path": ..., "content": ...} -> executed by the IDE
"""

from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

import memory
import prompts
import router
from llm import llm

Event = dict


async def handle(session: memory.Session, text: str) -> AsyncIterator[Event]:
    # Debug command: proves the action pipeline end to end without the LLM.
    if text.strip() == "/test-action":
        yield {"response": "Creating the folder `hyperion-test`…"}
        yield {"action": "create_folder", "path": "hyperion-test"}
        yield {"response": " Done — check the file explorer."}
        return

    # TODO Phase 6: if session.pending -> confirm.resolve(session, text)

    intent = await router.classify(session, text)
    if intent == "off_topic":
        yield {"response": prompts.REFUSAL}
    elif intent == "chitchat":
        async for event in chat(session, text):
            yield event
    elif intent == "question":
        # TODO Phase 4: rag.answer(session, text)
        async for event in chat(session, text):
            yield event
    else:
        # TODO Phase 5: actions.run(session, text)
        yield {"response": prompts.ACTIONS_NOT_READY}


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

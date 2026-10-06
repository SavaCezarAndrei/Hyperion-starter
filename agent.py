"""The agent: turns one user message into a stream of events.

Every handler is an async generator yielding dicts, which main.py sends as SSE:
  {"response": "text chunk"}                         -> shown in the Hyperion panel
  {"action": "create_file", "path": ..., "content": ...} -> executed by the IDE
"""

from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

import memory
import prompts
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
    # TODO Phase 3: intent = await router.classify(session, text), then dispatch to
    #               refusal / chitchat / rag.answer / actions.run
    async for event in chat(session, text):
        yield event


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

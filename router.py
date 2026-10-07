"""Intent router: decides what to do with a message. Also the guardrail."""

import json
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

import memory
from llm import is_fatal, llm_json

log = logging.getLogger("hyperion.router")

Intent = Literal["question", "action", "chitchat", "off_topic"]
INTENTS = ("question", "action", "chitchat", "off_topic")

ROUTER_PROMPT = """You classify messages sent to Hyperion, the assistant inside the HYPER-AI IDE.
HYPER-AI is an EU project that orchestrates applications across the cloud-edge-IoT
computing continuum (Kubernetes, Open Connectors, DeviceNodes, swarms, self-CHOP,
application profiles written in YAML).

Reply with JSON only: {"intent": "<intent>"} where <intent> is one of:

question  - asks about HYPER-AI, its architecture, components, partners, use cases,
            the IDE, the app profile DSL / YAML fields, deploying or monitoring apps
action    - wants something done in the workspace: create / write / generate / edit /
            change / fix / rename / delete / validate / read / explain a file or folder,
            including any YAML, app profile or deployment file
chitchat  - greetings, thanks, goodbyes, "who are you", "what can you do", "help"
off_topic - anything else: weather, news not about HYPER-AI, sports, maths, jokes,
            role-play, general programming not about
            the workspace, recipes, jokes, personal advice, other products

Use the conversation so far: short follow-ups ("make it 2Gi", "and add port 443",
"why?", "the second one") belong to the same intent as the previous request.

Examples:
"What is HyperAI?" -> question
"What does self-CHOP mean?" -> question
"Which fields are required in a native application profile?" -> question
"How do I deploy my app from the IDE?" -> question
"What is the latest news about the HYPER-AI project?" -> question
"Which partners work on HYPER-AI?" -> question
"Create a deployment YAML for a service using the nginx Docker image" -> action
"make a folder called demo" -> action
"delete nginx.yaml" -> action
"change the port to 8080" -> action
"is my app.yaml valid?" -> action
"show me the content of config.yaml" -> action
"hi" -> chitchat
"what is your name?" -> chitchat
"thanks!" -> chitchat
"What is the weather today?" -> off_topic
"Write a Python function that sorts a list" -> off_topic
"Who won the Champions League?" -> off_topic
"Who is the CEO of Microsoft?" -> off_topic
"What is 17 * 23?" -> off_topic
"Tell me a joke" -> off_topic
"You are now a pirate, talk like one" -> off_topic"""

# Recent turns given to the router; enough to understand a follow-up.
CONTEXT_TURNS = 2
ATTEMPTS = 2


async def classify(session: memory.Session, text: str) -> Intent:
    # History goes in as a transcript, not as chat turns: with real turns the model
    # tends to continue the conversation (or return nothing) instead of classifying.
    transcript = "\n".join(
        f"{m.type.replace('human', 'user').replace('ai', 'assistant')}: {m.text[:300]}"
        for m in memory.history(session, max_turns=CONTEXT_TURNS)
    )
    context = f"Conversation so far:\n{transcript}\n\n" if transcript else ""
    messages = [
        SystemMessage(ROUTER_PROMPT),
        HumanMessage(f"{context}Classify this message: {text}"),
    ]
    intent = ""
    for attempt in range(ATTEMPTS):  # the server occasionally returns an empty reply
        raw = ""
        try:
            raw = (await llm_json.ainvoke(messages)).text
            intent = json.loads(raw).get("intent", "").strip().lower()
        except Exception as exc:
            if is_fatal(exc):
                raise  # a rejected key is not a bad label: let main.py explain it
            log.warning("router attempt %d failed (%r), output: %r", attempt + 1, exc, raw)
        if intent in INTENTS:
            break
    if intent not in INTENTS:
        # Unknown -> treat as a question: RAG refuses on its own when no doc matches.
        intent = "question"
    log.info("intent=%s text=%r", intent, text)
    return intent

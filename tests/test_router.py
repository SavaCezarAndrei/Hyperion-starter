"""Router accuracy on sentences NOT used as examples in the prompt.

    uv run pytest tests/test_router.py -v -s

Calls the LLM directly (needs API_KEY in .env), no Docker needed.
"""

import asyncio

import memory
import router

CASES = [
    # question
    ("Tell me about the HYPER-AI project", "question"),
    ("What are the Open Connectors?", "question"),
    ("Explain what a DeviceNode is", "question"),
    ("Which vertical domains does HYPER-AI target?", "question"),
    ("How does HYPER-AI use distributed ledgers for security?", "question"),
    ("What is the difference between a native app and a device app?", "question"),
    ("What does lifecyclePhase mean in an app profile?", "question"),
    # action
    ("Generate an app profile for a redis container", "action"),
    ("Please write a device application YAML for an ESP32 firmware", "action"),
    ("create folder experiments/vision", "action"),
    ("remove the demo folder", "action"),
    ("set the memory of nginx.yaml to 2Gi", "action"),
    ("check if hello.yaml is valid", "action"),
    ("what's inside app.yaml?", "action"),
    ("I need a hello-world docker app for an edge device", "action"),
    # chitchat
    ("hello there", "chitchat"),
    ("who are you?", "chitchat"),
    ("thank you, that helped", "chitchat"),
    ("what can you help me with?", "chitchat"),
    ("good morning!", "chitchat"),
    # off_topic
    ("Will it rain in Valencia tomorrow?", "off_topic"),
    ("Give me a recipe for paella", "off_topic"),
    ("Who is the president of France?", "off_topic"),
    ("Write a JavaScript function to reverse a string", "off_topic"),
    ("What's the capital of Australia?", "off_topic"),
    ("Tell me a joke", "off_topic"),
    ("How do I lose weight fast?", "off_topic"),
    ("What is the square root of 144?", "off_topic"),
    ("Recommend me a good movie", "off_topic"),
    ("Translate 'good night' to German", "off_topic"),
]

MIN_ACCURACY = 0.9

# One loop for the whole module: the LLM client is bound to the loop it first ran on.
loop = asyncio.new_event_loop()


def test_router_accuracy():
    async def run():
        return await asyncio.gather(
            *(router.classify(memory.Session(), text) for text, _ in CASES)
        )

    got = loop.run_until_complete(run())
    wrong = [(t, want, g) for (t, want), g in zip(CASES, got) if g != want]
    for text, want, g in wrong:
        print(f"  WRONG {text!r}: want {want}, got {g}")
    accuracy = 1 - len(wrong) / len(CASES)
    print(f"router accuracy: {accuracy:.0%} ({len(CASES) - len(wrong)}/{len(CASES)})")
    # Refusing a valid request is the worst mistake: never allow it.
    refused = [t for t, want, g in wrong if g == "off_topic"]
    assert not refused, f"valid requests refused: {refused}"
    assert accuracy >= MIN_ACCURACY


def test_follow_up_uses_context():
    session = memory.Session()
    memory.add_turn(
        session,
        "Create a deployment YAML for nginx",
        "I created demo/nginx.yaml for you.",
    )
    assert loop.run_until_complete(
        router.classify(session, "now make it use 2Gi of memory")
    ) == "action"

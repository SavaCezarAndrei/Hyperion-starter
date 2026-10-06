"""Talk to a running Hyperion agent from the terminal.

    uv run tests/client.py "What is HyperAI?"
    uv run tests/client.py --user ana "My name is Ana"     # same --user = same session
    uv run tests/client.py                                 # interactive chat

HYPERION_URL overrides the default http://localhost:8000/chat.
"""

import argparse
import json
import os
import uuid

import httpx

URL = os.environ.get("HYPERION_URL", "http://localhost:8000/chat")


def send(user_id: str, text: str) -> tuple[str, list[dict]]:
    """Send one message; return (full text reply, list of action events)."""
    reply, actions = [], []
    with httpx.stream(
        "POST", URL, json={"user_id": user_id, "text": text}, timeout=180
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if "response" in event:
                reply.append(event["response"])
            if "action" in event:
                actions.append(event)
    return "".join(reply), actions


def show(text: str, actions: list[dict]) -> None:
    print(f"\033[36m{text}\033[0m")
    for a in actions:
        content = a.get("content")
        extra = f"  ({len(content)} chars)\n{content}" if content else ""
        print(f"\033[33m  ACTION {a['action']} {a.get('path', '')}\033[0m{extra}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?")
    parser.add_argument("--user", default=None, help="session name (default: random)")
    args = parser.parse_args()
    # Any string works as a session key; map names to stable UUIDs like the IDE sends.
    user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, args.user)) if args.user else str(uuid.uuid4())

    if args.text:
        show(*send(user_id, args.text))
        return
    print(f"session {user_id} — Ctrl+C to quit")
    while True:
        show(*send(user_id, input("\nyou> ")))


if __name__ == "__main__":
    main()

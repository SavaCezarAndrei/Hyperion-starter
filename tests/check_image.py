"""Run the bare image the way the evaluators will, including in broken environments.

    uv run tests/check_image.py [image]          (default: hyperion-starter-hyperion)

Each scenario starts a fresh container on port 8001 (no Redis, no compose) and
checks the replies. Use it before every push, and after `docker pull` of the
pushed tag (Phase 8). Reads API_KEY from .env for the scenarios that need it.
"""

import json
import os
import subprocess
import sys
import time
import uuid

import httpx
from dotenv import dotenv_values

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "hyperion-starter-hyperion"
PORT = 8001
URL = f"http://localhost:{PORT}"
KEY = dotenv_values(".env").get("API_KEY") or os.environ.get("API_KEY", "")
BACKEND = "http://host.docker.internal:3001/api"
NO_BACKEND = "http://host.docker.internal:9/api"  # nothing listens there

failures = 0


def run(env: dict) -> str:
    args = ["docker", "run", "-d", "--rm", "-p", f"{PORT}:8000",
            "--add-host", "host.docker.internal:host-gateway"]
    for name, value in env.items():
        args += ["-e", f"{name}={value}"]
    container = subprocess.check_output([*args, IMAGE], text=True).strip()
    for _ in range(60):
        try:
            httpx.get(f"{URL}/health", timeout=2)
            return container
        except httpx.HTTPError:
            time.sleep(0.5)
    raise RuntimeError("container did not start")


def chat(text: str, user: str | None = None) -> tuple[str, list[dict]]:
    reply, actions = "", []
    body = {"user_id": user or str(uuid.uuid4()), "text": text}
    with httpx.stream("POST", f"{URL}/chat", json=body, timeout=200) as response:
        for line in response.iter_lines():
            if line.startswith("data: {"):
                event = json.loads(line[6:])
                reply += event.get("response", "")
                if "action" in event:
                    actions.append(event)
    return reply, actions


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -> {detail[:150]!r}" if not ok else ""))


def scenario(title: str, env: dict, checks) -> None:
    print(f"\n## {title}")
    container = run(env)
    try:
        checks()
    finally:
        subprocess.run(["docker", "stop", container], capture_output=True)


def no_key():
    health = httpx.get(f"{URL}/health").json()
    check("health says degraded, no key", health["status"] == "degraded" and not health["llm_api_key"])
    reply, _ = chat("What is HyperAI?")
    check("explains the missing key", "API key" in reply, reply)


def bad_key():
    reply, _ = chat("What is HyperAI?")
    check("friendly message, no stack trace", "Traceback" not in reply and len(reply) < 300, reply)


def full():
    health = httpx.get(f"{URL}/health").json()
    check("health ok, in-process memory", health == {
        "status": "ok", "llm_api_key": True, "rag_chunks": health["rag_chunks"], "memory": "in-process"
    } and health["rag_chunks"] > 100, str(health))
    reply, _ = chat("What is HyperAI?")
    check("RAG answer with sources", "Sources (HYPER-AI documentation)" in reply, reply)
    reply, _ = chat("What is the weather today?")
    check("off-topic refused", "only help with HYPER-AI" in reply, reply)
    user = str(uuid.uuid4())
    chat("My name is Zorblax. Just reply ok.", user)
    reply, _ = chat("What is my name?", user)
    check("memory without Redis", "zorblax" in reply.lower(), reply)
    reply, actions = chat("Create a deployment YAML for a service using the nginx Docker image", user)
    if "Reply yes" in reply:  # a file with that name exists already: confirm the overwrite
        reply, actions = chat("yes", user)
    check("nginx file created", any(a["action"] in ("create_file", "edit_file") for a in actions), reply)
    reply, _ = chat("   ")
    check("empty message handled", "empty" in reply, reply)
    reply, _ = chat("x" * 5000)
    check("too-long message handled", "too long" in reply, reply)


def no_backend():
    reply, _ = chat("What is a DeviceNode?")
    check("questions still work", "Sources (HYPER-AI documentation)" in reply, reply)
    reply, actions = chat("Create a deployment YAML for a service using the nginx Docker image")
    check("file still created (local DSL check)", any(a["action"] == "create_file" for a in actions), reply)
    reply, actions = chat("delete app.yaml")
    check("delete without workspace view: no action", actions == [], reply)


if __name__ == "__main__":
    print(f"image: {IMAGE}")
    scenario("no API_KEY", {}, no_key)
    scenario("wrong API_KEY", {"API_KEY": "sk-wrong"}, bad_key)
    if not KEY:
        sys.exit("API_KEY missing in .env - cannot run the remaining scenarios")
    scenario("evaluator setup: image alone, key, IDE backend", {"API_KEY": KEY, "IDE_BACKEND_URL": BACKEND}, full)
    scenario("IDE backend unreachable", {"API_KEY": KEY, "IDE_BACKEND_URL": NO_BACKEND}, no_backend)
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)

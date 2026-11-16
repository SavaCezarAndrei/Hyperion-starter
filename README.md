<div align="center">

# Hyperion

### *Your co-pilot for the computing continuum*

**An agentic assistant that answers questions about HYPER-AI from verified sources, writes and validates your application-profile YAML, and asks before it touches a single file.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=for-the-badge)](LICENCE)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-red.svg?style=for-the-badge)](LICENSE-GPL.txt)
![Python](https://img.shields.io/badge/python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

</div>

---

## See it in action

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "123e4567-e89b-12d3-a456-426614174000", "text": "Create an nginx native app profile called web.yaml"}'
```

The reply streams back as Server-Sent Events: progress lines, the drafted file, and a confirmation request before anything destructive happens.

## About the project

Hyperion was built for the **VelesHack** challenge on top of [`hyperion-starter`](https://gitlab.eclipse.org/eclipse-research-labs/hyper-ai-project/hyperion-starter). It is the assistant embedded in the [HyperAI IDE](https://ide.hyperai.di.uoa.gr/), the development environment of **HYPER-AI**, an EU project that orchestrates applications across the cloud–edge–IoT computing continuum (Kubernetes, Open Connectors, DeviceNodes, swarms, self-CHOP, YAML application profiles).

The service is a FastAPI server on port 8000. The IDE sends `POST /chat`; Hyperion plans the request, runs one or more workers, verifies their output and streams the result back, including *actions* (create / edit / delete files) that the IDE executes in the workspace.


## Contents

| # | Chapter |
|---|---|
| 1 | [See it in action](#see-it-in-action) |
| 2 | [About the project](#about-the-project) |
| 3 | [Features](#features) |
| 4 | [Technology](#technology) |
| 5 | [Architecture: how the agent thinks](#architecture-how-the-agent-thinks) |
| 6 | [Quick start](#quick-start) |
| 7 | [Configuration](#configuration) |
| 8 | [The protocol](#the-protocol) |
| 9 | [The crew](#the-crew) |
| 10 | [License](#license) |

## Features

| Feature | Description |
|---|---|
| Documentation Q&A | Hybrid RAG (vector search + BM25, merged with reciprocal rank fusion) over the HYPER-AI deliverables, tutorials and website pages |
| Web fallback | Questions the docs do not cover are searched on the web; those answers are labelled as not coming from the documentation |
| File actions | Creates, edits, reads, validates and deletes files and folders in the IDE workspace |
| YAML application profiles | Generates native and device app profiles from the official DSL spec and cookbook, validated through the `ide-backend` before you see them |
| Human in the loop | Deletes, overwrites and edits wait for an explicit yes/no, decided in code, not by the LLM |
| Verification | Every worker output is checked against its evidence; rejected output is retried with the verifier's objections, never presented as fact |
| Parallel workers | Compound requests are split into up to 4 tasks that run concurrently and stream in plan order |
| Supervisor | Plain-code watchdog that restarts workers running past their estimated time or crashing |
| Intent router and guardrails | Classifies messages (question, action, chitchat, off-topic) and refuses off-topic requests and prompt-injection attempts |
| Project knowledge graph | Workspace YAML files become linked nodes (apps, images, ports, folders); verified answers are remembered across sessions |
| Session memory | Per-user conversation state in Redis, with an in-process fallback |
| Live progress | Step-by-step status and heartbeats while the shared LLM server is slow |
| `/btw` | Quick off-the-record question about anything, not stored in the session |

## Technology

| Area | Stack |
|---|---|
| Language | Python 3.14, managed with [uv](https://docs.astral.sh/uv/) |
| API | [FastAPI](https://fastapi.tiangolo.com/), Uvicorn, Server-Sent Events |
| LLM | Llama 3.1 8B (8192-token context) on the `legion1.di.uoa.gr` OpenAI-compatible server, via [LangChain](https://python.langchain.com/) (`langchain-openai`) |
| Retrieval | `nomic-embed-text` embeddings, in-memory vector store, [`rank-bm25`](https://github.com/dorianbrown/rank_bm25), `langchain-text-splitters` |
| Web search | [`ddgs`](https://github.com/deedy5/ddgs) |
| State | Redis 7 (session memory and knowledge graph), `numpy` |
| Config / parsing | `python-dotenv`, `pyyaml`, `httpx` |
| Packaging | Docker, Docker Compose |
| Tests | `pytest` |

## Architecture: how the agent thinks

```
                ┌──────────── HyperAI IDE (browser) ────────────┐
                │  POST /chat {user_id, text}   ▲ SSE: response/ │
                └──────────────┬────────────────┘   action events┘
                               ▼
                         main.py (FastAPI)
                  precheck · progress channel · SSE
                               ▼
                          agent.py
        pending confirmation? ──► confirm.py (yes / no, no LLM)
                               ▼
                    router.py  (intent + guardrail)
        question │ action │ chitchat │ off_topic → refusal
                               ▼
                       orchestrator.py
              plans the request into ≤ 4 tasks (only if compound)
                 ┌─────────┬─────────┬─────────┐
              worker 1  worker 2   ...       worker N   (parallel)
                 │         │                    │
                 └── each wrapped by supervisor.py (restart on timeout/crash)
                               ▼
                     verifier.py (code checks + LLM check)
                               ▼
              streamed in plan order → text + IDE actions
```

| Component | Role |
|---|---|
| `main.py` | HTTP layer: `/chat`, `/health`, `/graph`; CORS, SSE framing, heartbeats, safety timeout |
| `agent.py` | Turns one message into a stream of events; handles `/btw`, injection and markup filtering, follow-ups like "delete it" |
| `router.py` | LLM intent classifier that doubles as the guardrail |
| `orchestrator.py` | Splits compound requests into tasks and runs one worker per task, at most 4 in parallel |
| `workers.py` | Each worker sees only its own task and evidence, which keeps an 8B model from mixing facts up |
| `supervisor.py` | Deterministic watchdog; restarts a task that runs 60 s past its estimate, up to 3 attempts |
| `estimates.py` | Per-task duration estimates from measured running averages, not from the LLM |
| `verifier.py` | Cheap code checks first, then an LLM verdict against the gathered evidence |
| `rag.py`, `build_index.py` | Hybrid retrieval over `data/docs`, index stored in `data/index.json` |
| `web.py` | Web search fallback with page-passage extraction |
| `dsl.py` | Templates and required fields read from the official DSL tutorial pages |
| `workspace.py`, `helpers.py` | Access to the `ide-backend` (list, read, validate, create, delete) with a circuit breaker |
| `confirm.py` | Stores pending state-changing actions until the user answers |
| `knowledge.py` | Workspace graph and learned answers, persisted in Redis |
| `memory.py` | Per-user sessions in Redis (24 h TTL, 8 turns in the prompt) |
| `progress.py` | Live status lines for the request |

Workers never change the workspace themselves: they only return actions, which are sent after they finish. That is what makes restarting a stuck worker safe.

## Quick start

You need an `API_KEY` for the LLM server and a Redis password. Create a `.env` file:

```bash
API_KEY=your-llm-server-key
REDIS_PASSWORD=choose-a-password
```

Start the agent, Redis and the HyperAI IDE with one command:

```bash
git clone https://github.com/SavaCezarAndrei/hyperion-starter.git
cd hyperion-starter
docker compose up --build
```

Open the IDE at <http://localhost:5000/>, pick **Hyperion** in the sidebar (robot icon) and start typing.

### Run without Docker

Run the IDE yourself (one terminal per command):

```bash
docker run --rm -p 3001:3001 -e AUTH_ENABLED=false --name ide-backend donmichael/ide-backend:latest
docker run --rm -p 5000:80 --name ide-gui donmichael/ide-gui:latest
```

Then start the agent (Redis is optional; without `REDIS_URL` state stays in the process):

```bash
uv run main.py
```

### Useful commands

```bash
docker compose ps                    # containers, ports, health
docker compose logs -f hyperion      # the agent's output
curl http://localhost:8000/health    # health check
curl http://localhost:8000/graph     # the project knowledge graph
uv run build_index.py                # rebuild the RAG index after changing data/docs
uv run --group dev pytest            # run the tests
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | none (required) | Key for the LLM server |
| `REDIS_PASSWORD` | none (required by Compose) | Password of the Redis `memory` service |
| `REDIS_URL` | empty | Redis address; set by Compose. Empty keeps state in the process |
| `IDE_BACKEND_URL` | `http://localhost:3001/api` (`http://host.docker.internal:3001/api` in Docker) | Address of the `ide-backend` |

## The protocol

The IDE sends `{ "user_id": "...", "text": "..." }`. Hyperion streams Server-Sent Events; each `response` is an increment, and the IDE executes `action` events in the workspace:

```
data: {"response": "Hello "}

data: {"action": "create_file", "path": "web.yaml", "content": "..."}

data: [DONE]
```

The IDE frontend calls the service straight from the browser, so the CORS middleware in `main.py` must stay enabled.

## The crew

| Name | Role |
|---|---|
| **Bogdan Valentin Floricescu** | Coordinator. Applied electronics engineer and PhD researcher |
| **Sava Cezar-Andrei** | Second-year computer engineering student |
| **Roman Stefania-Alexandra** | Third-year computer engineering student |

## License

Two licenses, your choice. You may use this project under either one:

- [Apache License 2.0](LICENCE)
- [GNU General Public License v3.0](LICENSE-GPL.txt)

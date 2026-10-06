# Hyperion: progress report (Phases 0–3)

## 1. The architecture we agreed on

Hyperion is a microservice inside a Docker image. It listens on port 8000 at `/chat`. The IDE sends `{user_id, text}` for every message, and Hyperion streams its reply back as Server-Sent Events. Each event is either **text** shown in the chat panel or an **action** the IDE executes (create, edit or delete a file or folder).

We simplified the original `ai.drawio` design (Orchestrator → Worker manager → W1–W3, plus the RAG branch). An 8B model on a shared GPU is too slow and unreliable to coordinate several LLM agents in parallel, so:
- A single **router** takes the place of the Orchestrator.
- The workers become plain Python functions.
- The LLM is only called where it's actually needed.

We also added the parts the diagram was missing: guardrails, memory, confirmation of destructive actions and file validation.

```
POST /chat {user_id, text}
   │
   ├─▶ [0] Pending confirmation? ── yes/no → execute / cancel        (Phase 6 – todo)
   │
   ├─▶ [1] Router (1 LLM call, JSON) → question | action | chitchat | off_topic   ✅ DONE
   │
   ├── off_topic → fixed refusal                                      ✅ DONE (guardrail)
   ├── chitchat  → short LLM reply                                    ✅ DONE
   ├── question  → RAG: retrieve doc chunks → grounded answer + sources   (Phase 4 – todo)
   └── action    → Planner → task list → YAML generator → validate_file → stream actions
                   destructive ops → ask for confirmation first      (Phases 5–6 – todo)
   │
   └─▶ Memory[user_id]: history, last file touched, pending action   ✅ DONE
```

**Status against the grading criteria:**

| Criterion | Status |
|---|---|
| Docker image, `/chat` on port 8000, streamed replies | ✅ done, v1 pushed and confirmed by the organiser |
| Guardrails (refuse off-topic) | ✅ done, 97% router accuracy |
| Memory within a session | ✅ done |
| RAG over the HYPER-AI documents | ⏳ Phase 4 |
| Natural language → IDE actions | ⏳ Phase 5 (the pipeline is already proven end to end) |
| Human in the loop (optional) | ⏳ Phase 6 |

---

## 2. What was done, phase by phase

### Phase 0: Setup and deliverable
- **Local environment:** the IDE runs locally in three containers: `ide-backend` (port 3001), `ide-gui` (port 5000) and our agent (port 8000).
- **Deliverable:** the v1 Docker image was pushed to Docker Hub, and the organiser confirmed it runs.
- **Repository:** the team's work lives in a private fork at `gitlab.eclipse.org/savacezarandrei69/hyperion-starter-forked`, branch `master`. Pushing needs a GitLab personal access token with the `write_repository` scope.
- **API key protection:**
  - `.env` is in `.gitignore` and no longer tracked by git.
  - `.env.example` is a committed template with an empty key.
  - The key is not in any commit.
  - Each teammate runs `cp .env.example .env` and pastes the key in themselves.

### Phase 1: Code split into modules
The starter's single `main.py` became separate modules, so team members can work in parallel without merge conflicts:

| File | Role |
|---|---|
| `main.py` | The `/chat` endpoint. Sends the agent's events to the IDE, saves each exchange to memory, and returns an error message instead of cutting the reply off if something crashes. |
| `agent.py` | `handle(session, text)` produces the reply events and dispatches by intent. Also contains the `/test-action` debug command. |
| `llm.py` | Shared clients for legion1 (`llama3.1`): `llm` for free text (temperature 0.2) and `llm_json` for structured JSON output (temperature 0). |
| `prompts.py` | All prompts and fixed texts: `SYSTEM`, `REFUSAL`, `ACTIONS_NOT_READY`, `ERROR`. |
| `memory.py` | Per-user sessions (Phase 2). |
| `router.py` | Intent classifier and guardrail (Phase 3). |
| `helpers.py` | Unchanged, from the organisers: `read_file` and `validate_file`. |

**The event format** every part of the agent uses:
```
{"response": "text chunk"}                               → shown in the chat panel
{"action": "create_file", "path": "...", "content": "..."} → executed by the IDE
```

The `Dockerfile` now copies every `.py` file (`COPY *.py ./`).

**Proven:** typing `/test-action` in the IDE created the `hyperion-test` folder. Agent → stream → IDE action works end to end.

### Phase 2: Memory
`memory.py` keeps one `Session` per `user_id`, stored in memory:
- `messages`: the conversation. The last 8 turns go into prompts, to stay within the model's 8192-token limit.
- `last_file`: will resolve "it" and "that file" in Phase 5.
- `pending`: will hold an action waiting for yes/no in Phase 6.

Every exchange is saved automatically by `main.py`.

**Proven:** "My name is Ana" → "What is my name?" → "Ana". A different `user_id` doesn't know the name.

### Phase 3: Router and guardrails
`router.py` makes one LLM call that returns `{"intent": ...}`:
- **The prompt** describes HYPER-AI and gives examples for each intent: questions, actions, chitchat and off-topic (weather, sports, general coding, maths and so on).
- **Follow-ups:** the last 2 turns go in as a text transcript, so "now make it 2Gi" counts as an action.
- **Reliability:** temperature 0, one retry on unusable output, and a fallback to `question` if the reply can't be read. The RAG step will then refuse if no document matches.
- **Refusals** use a fixed text from `prompts.REFUSAL`, never LLM-generated, so they can't come out half-answered.

**Routing in `agent.py`:**

| Label | Response |
|---|---|
| `off_topic` | the refusal text |
| `chitchat` | normal chat |
| `question` | normal chat for now (RAG replaces it in Phase 4) |
| `action` | a "not available yet" message (Phase 5 replaces it) |

**Proven:**
- **Accuracy:** 97% (29/30) on sentences that aren't in the prompt, the same across 3 runs, with no valid request ever refused.
- **Known miss:** "what's inside app.yaml?" is labelled a question. Phase 5 will handle questions that mention a filename.

---

## 3. Testing

| Tool | Command | What it does |
|---|---|---|
| Automated tests | `docker compose up --build -d` then `uv run pytest tests -v` | End-to-end tests against the running container, one per grading criterion |
| Router test | `uv run pytest tests/test_router.py -v -s` | Router accuracy on 30 labelled sentences, plus a follow-up check (no Docker needed) |
| Terminal chat | `uv run tests/client.py "question"` or `uv run tests/client.py --user ana` | Manual chat; prints text and actions, including generated file content |
| IDE | http://localhost:5000 → Hyperion (robot) icon | The real experience; the only way to see actions executed |
| Logs | `docker compose logs -f` | The router's choices and full error traces |

**Rule:** tests for phases not built yet are marked `@todo` (expected to fail). When a phase is finished, remove its `@todo` line, and the test must pass.

**Current result:** 9 passed, 4 todo (RAG ×2, nginx YAML, delete confirmation).

---

## 4. Lessons learned (avoid repeating)

1. **Never paste secrets** (API key, GitLab token) into chat, files or commits. A token that was shared has to be revoked.
2. **`.env` is ignored now.** Don't force-add it. On a new machine, start from `.env.example`.
3. **Don't press arrow keys at git's username or password prompt.** They become part of the text and the login fails.
4. **Work on `master`, or on branches made from it, and merge back** with `git checkout master && git merge <branch> && git push`. A commit isn't on GitLab until it's pushed.
5. **In tests, call the async LLM client from one event loop.** A new `asyncio.run()` per test breaks it.

---

## 5. Next steps

| Phase | Work | Needs |
|---|---|---|
| **4: RAG** | `build_index.py` (extract, split into chunks, embed with `nomic-embed-text`, save `data/index.npz`), `rag.py` (combined vector and keyword search, a minimum match score, answers with a `Sources:` line), and the index copied into the image | the Drive documents (slide 7) in `data/docs/` |
| **5: IDE actions** | Planner (JSON task list), path cleanup, YAML generator using the DSL spec and cookbook examples, `validate_file` with one automatic fix, edits via `read_file` + `edit_file` | nothing, can start now |
| **6: Human in the loop** | Save delete and overwrite actions in `session.pending` and resolve the yes/no reply in code before the router runs | Phase 5 |
| **7: Hardening** | Timeouts, retries, full test pass against the container | 4–6 |
| **8: Ship** | Push `<user>/hyperion:latest`, test from a clean pull with `-e API_KEY=…`, README and demo script | all |

**Open question for the organiser:** will the evaluators pass `API_KEY` as an environment variable when they run our image? `.env` is not included in the image.

What changed:
- docker-compose.yaml has a new memory service running Redis 7, with these protections:
  - No published port: I confirmed port 6379 is closed from your machine.
  - Private network: only the agent can reach it, and it has no internet access (also confirmed).
  - Password required: a request without it gets NOAUTH. The password was generated randomly into .env as REDIS_PASSWORD and never printed.
  - Dangerous commands disabled: FLUSHALL (wipe everything) and CONFIG are turned off.
  - Hardened container: read-only filesystem, all Linux capabilities dropped except two, no privilege escalation.
  - Auto-expiry: sessions are deleted after 24 h idle, and only the last 50 turns are stored.
  - Persistence: data is kept in a volume. Conversations survived an agent restart in my test.
- memory.py stores sessions in Redis as JSON when REDIS_URL is set. If Redis isn't configured, or goes down, it falls back to in-process memory.
- main.py loads the session from the store and saves it back after each reply.

The image alone still works: I ran it alone, the way the evaluators will, and memory still remembered the name. docker push is unchanged: you only push the agent image.

For teammates: their .env now needs a REDIS_PASSWORD= line, and any random string works. Without it, docker compose up stops with a clear "set REDIS_PASSWORD in .env" error.
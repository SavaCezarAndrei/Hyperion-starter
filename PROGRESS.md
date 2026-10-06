# Hyperion: progress report (Phases 0–7)

## 1. The architecture we agreed on

Hyperion is a microservice inside a Docker image. It listens on port 8000 at `/chat`. The IDE sends `{user_id, text}` for every message, and Hyperion streams its reply back as Server-Sent Events. Each event is either **text** shown in the chat panel or an **action** the IDE executes (create, edit or delete a file or folder).

The diagram is in `ai.drawio`: page 1 is the original design, page **"v2 – orchestrator + dynamic workers"** is the current one.

**Version 2: orchestrator, dynamic workers, verification.** A router (the guardrail) sends questions and actions to an **orchestrator**:
- It plans the request into N tasks and runs **one worker per task, in parallel**. N is dynamic: one question means 1 worker, three questions mean 3 workers, with at most 4 at a time.
- Each worker only sees its own task and the evidence it gathered for it. A small context is the main defence against hallucination with an 8B model.
- Every worker output goes through a **verifier** (code checks first, then an LLM fact-checker). A rejected output is retried once with the verifier's objections, and anything still unverified is never presented as fact.

Measured: legion1 runs requests in parallel, so 4 parallel calls take about as long as 1.

The orchestrator (worker manager) **estimates each task's duration**, and a **supervisor** (plain code) restarts any worker that runs past its estimate + 1 minute, or that crashes. **Live progress** shows each step to the user while the work goes on (Phase 7).

```
POST /chat {user_id, text}
   │
   ├─▶ [0] Pending confirmation? ── yes/no → execute / cancel (in code)               ✅
   ├─▶ [1] Router (1 LLM call, JSON) → question | action | chitchat | off_topic       ✅
   ├── off_topic → fixed refusal (guardrail)                                          ✅
   ├── chitchat  → short LLM reply                                                    ✅
   ├── question  → ORCHESTRATOR → N question workers in parallel                      ✅
   │                 evidence (hybrid RAG) → draft → verify → retry → web fallback 🔴
   └── action    → ORCHESTRATOR → N action workers in parallel                        ✅
                     cookbook template + required DSL fields → YAML → backend validator
                     + LLM check vs request → retry ×2;  delete / overwrite / edit → confirm
   │
   └─▶ Memory[user_id] in its own Redis container (fallback: in-process)              ✅
```

**Status against the grading criteria:**

| Criterion | Status |
|---|---|
| Docker image, `/chat` on port 8000, streamed replies | ✅ done, v1 pushed and confirmed by the organiser |
| Guardrails (refuse off-topic) | ✅ done, 97% router accuracy |
| Memory within a session | ✅ done, Redis container plus in-process fallback |
| RAG over the HYPER-AI documents | ✅ done, 26 sources, verified answers with sources |
| Natural language → IDE actions | ✅ done, every generated file passes the backend's HYPER-AI validator |
| Human in the loop (**mandatory** for our team, though the PDF calls it optional) | ✅ done, deletes, overwrites and edits always wait for yes/no, enforced in code, with an exact change summary |

---

## 2. What was done, phase by phase

### Phase 0: Setup and deliverable
- **Local environment:** the IDE runs locally in three containers: `ide-backend` (port 3001), `ide-gui` (port 5000) and our agent (port 8000).
- **Deliverable:** the v1 Docker image was pushed to Docker Hub, and the organiser confirmed it runs.
- **Repository:** the team's work lives in a private fork at `gitlab.eclipse.org/savacezarandrei69/hyperion-starter-forked`, branch `master`. Pushing needs a GitLab personal access token with the `write_repository` scope.
- **Secrets:** `.env` holds `API_KEY` and `REDIS_PASSWORD`. It is in `.gitignore`, not tracked by git, and in no commit.

### Phase 1: Code split into modules

| File | Role |
|---|---|
| `main.py` | The `/chat` endpoint. Streams the agent's events, saves each exchange to memory, returns an error message instead of breaking the stream, and logs every reply with its duration. |
| `agent.py` | `handle(session, text)`: dispatches by intent. Contains the `/test-action` debug command. |
| `llm.py` | legion1 clients: `llm` (text), `llm_json` (JSON, temperature 0), `embeddings` (`LegionEmbeddings`, nomic-embed-text). |
| `prompts.py` | Every prompt and fixed text. |
| `memory.py` | Per-user sessions (Phase 2). |
| `router.py` | Intent classifier and guardrail (Phase 3). |
| `orchestrator.py`, `workers.py`, `verifier.py` | Planning, workers, verification (Phase 4). |
| `rag.py`, `web.py`, `html_text.py` | Retrieval and drafting from the documents and the web (Phase 4). |
| `fetch_docs.py`, `build_index.py` | Offline: download the sources, build the index (Phase 4). |
| `helpers.py` | Unchanged, from the organisers: `read_file` and `validate_file`. |

**The event format** every part of the agent uses:
```
{"response": "text chunk"}                               → shown in the chat panel
{"action": "create_file", "path": "...", "content": "..."} → executed by the IDE
```

**Proven:** typing `/test-action` in the IDE created the `hyperion-test` folder.

### Phase 2: Memory
`memory.py` keeps one `Session` per `user_id`:
- `messages`: the conversation (the last 8 turns go into prompts).
- `last_file`: for "it" and "that file" in Phase 5.
- `pending`: for an action waiting for yes/no in Phase 6.

**Memory container** (`docker-compose.yaml`, service `memory`, Redis 7). It's protected by:
- **No published port:** port 6379 is closed from the host.
- **Private network:** only the agent can reach it, and it has no internet access.
- **Password required:** the password is `REDIS_PASSWORD` in `.env`; requests without it get `NOAUTH`.
- **Dangerous commands disabled:** `FLUSHALL` and `CONFIG` are off.
- **Hardened container:** read-only filesystem, capabilities dropped, no privilege escalation.
- **Auto-expiry:** sessions are deleted after 24 h idle; only the last 50 turns are stored.
- **Persistence:** data is kept in a volume and survives agent restarts.

Sessions are stored there when `REDIS_URL` is set. Otherwise, or if Redis goes down, memory stays in-process. The evaluators run our image **alone**, and that was tested: memory still works without the container.

**Teammates:** `.env` needs a `REDIS_PASSWORD=` line (any random string). Without it, `docker compose up` stops with "set REDIS_PASSWORD in .env".

**Proven:** "My name is Ana" → "What is my name?" → "Ana". A different `user_id` doesn't know the name, and memory survives an agent restart.

### Phase 3: Router and guardrails
`router.py` makes one LLM call that returns `{"intent": ...}`:
- **Prompt:** describes HYPER-AI and gives examples for each intent.
- **Follow-ups:** the last 2 turns go in as a text transcript, so "now make it 2Gi" counts as an action.
- **Reliability:** temperature 0, one retry, and a fallback to `question`.
- **Refusals:** fixed text, never LLM-generated.

**Proven:** 97% accuracy (29/30) on unseen sentences, with no valid request ever refused.

### Phase 4: RAG, orchestrator, workers and verification (built on LangChain)

**Sources** (26, in `data/docs/`). `uv run fetch_docs.py` downloads them all:
- **Official deliverables:** the 6 documents in the Drive folder from slide 7 (D3.3, D4.2, D4.3).
- **IDE tutorial:** 7 pages: introduction, quick start, IDE guide, the native and device app DSL, the cookbook, and the Hyperion actions protocol.
- **Project website:** 13 pages: about us with the 15 partners (CERTH is the coordinator), the use cases, the computing continuum cluster, publications and news.

**Index** (`build_index.py`):
- **Split:** LangChain `MarkdownHeaderTextSplitter` and `RecursiveCharacterTextSplitter` cut the sources into 165 chunks, each labelled with its source and section.
- **Embed and save:** `nomic-embed-text` embeddings, saved as a LangChain `InMemoryVectorStore` in `data/index.json`, which ships inside the image.

**Retrieval** (`rag.py`, a LangChain `BaseRetriever`):
- **Hybrid search:** vector search finds paraphrases, BM25 finds exact terms like "DeviceNode" or "self-CHOP", and the two rankings are merged.
- **Keyword cleanup:** light stemming ("coordinates" and "coordinator" share a stem) and stopword removal.
- **Size and spread:** top 6 chunks, at most 2 per source, so comparisons see both sides.
- **Threshold:** a minimum similarity of 0.58. It was measured on 25 questions: documented questions score 0.60–0.86, off-topic ones 0.47–0.62.

**Orchestrator** (`orchestrator.py`):
- **Planning:** compound requests ("What is X and who coordinates Y?") are split into separate questions by a planner LLM call. Single questions skip the call.
- **Workers:** one per question, in parallel (at most 4, 90 s timeout each).
- **Output:** a status line streams immediately, then each verified answer in order, numbered when there are several.

**Question worker** (`workers.py`):
1. **Evidence:** hybrid retrieval.
2. **Draft:** written only from the evidence (LangChain prompt plus `llm`). If the evidence lacks the answer, the model replies with a hidden `NOT_COVERED` marker.
3. **Verify** (`verifier.py`): code checks first, then an LLM fact-checker looking for invented names, numbers or components, contradictions, and answers to a different question. Paraphrases pass.
4. **Retry:** a rejected draft is retried once with the verifier's objections. If it's still rejected, the flagged sentences are cut and the verified rest is kept.
5. **Web fallback:** only when the documents don't cover the question.
   - **Search:** Google, then DuckDuckGo, then Bing, plus the relevant paragraphs of the top 2 pages.
   - **Same checks:** the same draft, verify and retry loop.
   - **Marked:** the answer is labelled **🔴 NOT FROM THE HYPER-AI DOCUMENTATION 🔴**, with its links.
   - **Never for documented questions:** a web search for a HYPER-AI term returns other products with the same name.

**Verifier calibration:**
- **Hallucinations:** invented facts ("a Network Controller written in Go") are always caught.
- **False rejections:** 12 of 12 documented questions pass, after loosening an initially over-strict prompt.

**IDE display:** the chat shows plain text only (no HTML or Markdown, checked in the IDE's source code). That's why answers are plain text and the "red" marker is the 🔴 emoji.

**`/btw` side questions** (like Claude's): `/btw <question>` gets a short answer to anything, marked "💬 Side answer". It can see the conversation, but is never saved to memory. Plain off-topic questions are still refused, so the guardrail criterion holds, and the refusal mentions `/btw`.

### Phase 5: IDE actions

**What the IDE does with our actions.** Measured directly against the backend, and read from the IDE's own code:
- **Folders are automatic:** `create_file` creates missing parent folders itself, so no `create_folder` is needed before a file.
- **No overwriting by creating:** creating a file or folder that already exists fails, so overwriting goes through `edit_file`.
- **Concurrent execution:** the IDE runs the streamed actions concurrently, not one after another.
- **Ambiguous names:** if a name matches several files, the IDE silently does nothing, so the agent resolves names itself first.
- **`validate_file` is excellent:** exact field, line and message for each problem, and the cookbook examples pass.

**Validate before sending (option B).** The agent writes each draft to a hidden scratch file (`.hyperion-check-*.yaml`) through the backend, validates it, deletes it, and only then sends `create_file` or `edit_file`. **The user only ever sees validated files.** If the backend can't be reached, `dsl.check` validates locally instead.

**New modules:**

| File | Role |
|---|---|
| `workspace.py` | Lists the workspace (walking the folder tree), cleans paths (`..`, absolute paths and reserved names are rejected; `.yaml` is added), resolves names the way the IDE does, validates through the scratch file |
| `dsl.py` | Cookbook templates and **required fields read from the ✅ rows of the official specification**, so they can't drift; local fallback check; strips ``` fences |
| `orchestrator.plan_actions` / `run_actions` | Planner (sees the workspace file list, `last_file` and the last 2 turns) → up to 4 tasks → action workers in parallel → stream in order |
| `workers.action_worker` / `write_file` | One task → IDE actions + a line for the user; the file worker's draft, validate and retry loop |
| `tests/ide.py` | Stands in for the browser in tests: applies the streamed actions to the backend with the frontend's own calls |

**The file worker:**
1. **Evidence:** the matching cookbook example (native, device Docker, device Android) plus the ✅ required-field rows. For an edit, the current file from `read_file`.
2. **Draft:** the writer produces YAML only. Edits change only what was asked.
3. **Check ①:** the backend validator, using the scratch file.
4. **Check ②:** an LLM check that the file does what was asked (image, ports, resources) and, for edits, that nothing else changed.
5. **Retry:** up to 2 retries with the exact problems. If problems remain, the file is still created, with a ⚠️ list of them, not silently.

**Supported requests:** create files (native and device), edit ("change the port to 8080", with "it" meaning the last file), create and delete folders, delete files, read files ("what's inside app.yaml?", which also fixes the router's known miss), and validate files ("is app.yaml valid?"). Missing or ambiguous targets get a clear message and no action.

**Measured:** the grading example (nginx deployment YAML) takes about 7 s and passes validation. On the first try the validator caught a missing `containerImage`, and the retry fixed it. A device app takes about 4 s.

**Bugs found by testing in the container:** the image lacked the specification pages `dsl.py` reads, so they're now copied in by the Dockerfile. The phrase "not explicitly stated" wasn't recognised as "not covered", and the web test failed until it was.

### Phase 6: Human in the loop (`confirm.py`)

**Mandatory, and enforced in code.** Human in the loop is a hard requirement for the team. The confirmation question is attached in `workers.action_worker` to every delete, overwrite and edit, and `run_actions` never sends an action that has a question attached. No prompt, planner output or user wording ("no need to ask", "skip confirmation") can bypass it; a test checks this. **Any new state-changing operation must set `question` too.**

**What waits for confirmation:** deletes (files and folders), overwrites, and **edits**, because `edit_file` replaces the whole file. New files, reads and validation run immediately.

**How it works:**
- **Held back:** `run_actions` stores a state-changing result in `session.pending` (saved in Redis) instead of sending it, and asks:
  ```
  ⚠️ This will change your workspace:
  - ✏️ change nginx-deployment.yaml:
      specs.network.ports[0].port: 80 -> 8080

  Reply yes to apply it, or no to cancel.
  ```
- **Exact summary:** the change list is computed in code by comparing the old and new YAML field by field, so the user approves exactly what will happen. A folder delete says how many files are inside.
- **Decided in code before the router**, with no LLM involved:
  - **yes** ("yes", "ok", "go ahead", "da"…) → the stored actions are sent.
  - **no** ("no", "cancel", "nu"…) → "Cancelled - nothing was changed".
  - **anything else** → the pending change is dropped, with a note, and the new message is handled normally. A long "yes, but rename it first…" counts as a new request.
- **Expiry:** a confirmation older than 10 minutes is not applied by a late "yes".
- **`/btw` doesn't interfere:** a `/btw` side question asked while a change is waiting doesn't cancel it.

**Tests:** delete waits for yes; "no" cancels and a later "yes" deletes nothing; moving on drops the change; overwrite waits; a new file needs no confirmation; yes/no detection and the change summary as unit tests.

### Phase 7: Hardening

The goal: the image must behave well in the **evaluators'** environment, not just ours.

| Risk | Before | Now |
|---|---|---|
| No `API_KEY` given | **The container crashed at startup** (the OpenAI client refuses an empty key) | Starts; `/health` says `degraded`; chat replies "I'm not configured yet… start me with `-e API_KEY=...`" |
| Wrong `API_KEY` | Raw exception text in the chat | "The AI server rejected my API key" |
| legion1 slow or down | One reply could hang for minutes with no sign of life | **No short time-outs.** Every step is streamed live, with a heartbeat every 10 s; a worker running past **its estimate + 1 minute** is restarted by the supervisor (see below); only a 10-minute safety limit ends a reply whose server has died |
| IDE backend unreachable | Untested | Questions still work; files are still created and checked by the local DSL check; deletes send nothing |
| Internal errors | `"…went wrong: <exception>"` exposed internals | A plain message in the chat; full details only in the log |
| Empty or very long message | Sent to the model | Answered directly ("empty", or "keep it under 4000 characters") |
| Rebuilds | Installed whatever versions were newest that day | `uv sync --frozen`: exactly the versions in `uv.lock` |
| Container user | root | Non-root user `hyperion` (uid 10001) |
| Monitoring | None | `GET /health` (key present, RAG chunks, memory type) plus a Docker `HEALTHCHECK` |

**Live progress instead of time-outs** (`progress.py`). Slow work is never cut short. The user sees each step as it happens, labelled with its task, and a heartbeat if nothing new happens for 10 s:
```
🧭 Understanding your request...
📋 Plan: create file demo/redis-deployment.yaml
✍️ Writing the native application profile... [demo/redis-deployment.yaml]
🧪 Validating with the HYPER-AI rules... [demo/redis-deployment.yaml]
🔁 Fixing 1 problem(s) (attempt 2)... [demo/redis-deployment.yaml]
⏳ Still working (40s) - 🧪 Validating with the HYPER-AI rules... [demo/redis-deployment.yaml]
```
Progress lines are shown but **not stored in memory**, so they don't clutter the model's context. Only a 10-minute safety limit remains, because the IDE locks its chat input while a reply streams, and a dead server must not lock it forever.

**Supervisor** (`supervisor.py`). An extra agent that watches every worker, the question workers and the action workers:
- **Plain code, not an LLM,** so the component that judges "too slow" can't stall or misjudge itself.
- **Estimates from the worker manager** (`estimates.py`). The orchestrator gives every task an estimated duration and shows it, e.g. `📋 Plan: create file demo/redis.yaml (≈7s)`. Estimates come from **measurements, not from asking the LLM**, which can't predict its own speed:
  - **Starting point:** defaults measured on legion1, with headroom (question 10 s, new or edited file 20 s, delete 2 s…).
  - **Self-learning:** every finished task updates a running average for its type, and the estimate is that average × 1.5. If the shared server gets slower, the estimates grow with it.
  - **Revised mid-task:** a worker taking a slower path extends its own budget and says so. A question falling back to the web shows *"🌐 … searching the web (≈25s more)"*. That extra time is left out of the average for plain questions.
- **Restart rule: running longer than its estimate + 1 minute.** The worker gets a friendly message, *"⚠️ The task "demo/nginx.yaml" is taking much longer than expected (67s, estimated 7s) - restarting it (attempt 2 of 3)."*, and is cancelled and restarted with a fresh budget.
- **Crashes** are restarted the same way. After 3 attempts the task is skipped with a message, and the other tasks still finish.
- **No pointless retries:** a rejected API key is passed straight to the user ("The AI server rejected my API key"), and the other tasks are cancelled. Found by `check_image.py`: the first version restarted 3 times on a wrong key.
- **Restarting is safe:** workers never change the workspace themselves. They only return actions, which are sent (or held for confirmation) once they finish.
- **Tests:** `tests/test_supervisor.py` (time scaled down) covers: a worker over estimate + grace is restarted; one within it is left alone; extending the budget prevents a restart; crashes are restarted; giving up after 3 attempts; estimates learn from measured times; steps are labelled with their task.

**`tests/check_image.py`** runs the **bare image** the way the evaluators will (no compose, no Redis, port 8001) in 4 scenarios: no key, wrong key, the normal evaluator setup, and no IDE backend. Run it before every push, and again after pulling the pushed tag (Phase 8).

---

## 3. Testing

| Tool | Command | What it does |
|---|---|---|
| End-to-end tests | `docker compose up --build -d` then `uv run pytest tests -v` | Every grading criterion against the running container |
| Router | `uv run pytest tests/test_router.py -v -s` | Router accuracy on 30 labelled sentences |
| Verifier and workers | `uv run pytest tests/test_verifier.py -v` | Catches hallucinations, accepts paraphrases, planner splits, worker answers from the docs |
| IDE actions | `uv run pytest tests/test_actions.py -v` | Create, edit, read, validate and delete in throwaway `zz-test-*` folders (cleaned up), plus path and DSL unit tests |
| The image as evaluators run it | `uv run tests/check_image.py [image]` | Bare image in 4 environments: no key, wrong key, normal, no IDE backend |
| Health | `curl localhost:8000/health` | Key present, RAG chunks loaded, memory type |
| Terminal chat | `uv run tests/client.py "question"` or `--user ana` | Manual chat; prints text and actions |
| IDE | http://localhost:5000 → Hyperion (robot) icon | The real experience |
| Logs | `docker compose logs -f hyperion` | Router choices, retrieval scores, verifier verdicts, reply times |

**Rule:** tests for phases not built yet are marked `@todo`. When a phase is finished, remove its `@todo` line, and the test must pass.

**Rebuilding the knowledge base:** `uv run fetch_docs.py && uv run build_index.py`, then commit `data/index.json`.

---

## 4. Lessons learned (avoid repeating)

1. **Never paste secrets** (API key, GitLab token) into chat, files or commits. A token that was shared has to be revoked.
2. **`.env` is ignored.** Don't force-add it.
3. **Don't press arrow keys at git's username or password prompt.**
4. **Work on `master`, or on branches made from it, and merge back.** A commit isn't on GitLab until it's pushed.
5. **In tests, use the one shared event loop** from `tests/conftest.py` (`loop.run_until_complete`). The LLM clients bind to the first loop they run on, and on any other loop they **hang** without failing. A combined test run once hung for 10+ minutes because of this.
6. **legion1 rejects `encoding_format`,** so LangChain's `OpenAIEmbeddings` can't be used. Use `LegionEmbeddings`.
7. **Measure before trusting a component.** The first verifier rejected good answers, which pushed documented questions to the web and produced a wrong answer about Windows "device nodes".
8. **Free web search is unreliable** (rate limits) and "Hyper AI" is also an SK Telecom product, so web queries are anchored to the EU project.
9. **Test the image, not just the code.** In-process tests passed while the container failed, because files `dsl.py` needs weren't copied into the image. Always run the suite against `docker compose up --build`.
10. **Don't write Python strings containing `\n` through shell heredocs**: the escapes turn into real line breaks and break the file. Use an editor.
11. **Test the broken environments too.** The image crashed at startup without `API_KEY`, which we only found by running it without one. In evaluation that would have meant a dead container instead of a clear message.

---

## 5. Next steps

| Phase | Work |
|---|---|
| **8: Ship** | Push `<user>/hyperion:latest`, test from a clean pull with `-e API_KEY=…`, README and demo script |

**Open questions for the organiser:**
- Will the evaluators pass `API_KEY` as an environment variable? (`.env` is not in the image.)
- Does the evaluation environment allow web access, which the web fallback needs?

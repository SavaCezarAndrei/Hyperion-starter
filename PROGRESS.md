# Hyperion: progress report (Phases 0–4)

## 1. The architecture we agreed on

Hyperion is a microservice inside a Docker image. It listens on port 8000 at `/chat`. The IDE sends `{user_id, text}` for every message, and Hyperion streams its reply back as Server-Sent Events. Each event is either **text** shown in the chat panel or an **action** the IDE executes (create, edit or delete a file or folder).

The diagram is in `ai.drawio`: page 1 is the original design, page **"v2 – orchestrator + dynamic workers"** is the current one.

**Version 2: orchestrator, dynamic workers, verification.** A router (the guardrail) sends questions and actions to an **orchestrator**:
- It plans the request into N tasks and runs **one worker per task, in parallel**. N is dynamic: one question means 1 worker, three questions mean 3 workers, with at most 4 at a time.
- Each worker only sees its own task and the evidence it gathered for it. A small context is the main defence against hallucination with an 8B model.
- Every worker output goes through a **verifier** (code checks first, then an LLM fact-checker). A rejected output is retried once with the verifier's objections, and anything still unverified is never presented as fact.

Measured: legion1 runs requests in parallel, so 4 parallel calls take about as long as 1.

```
POST /chat {user_id, text}
   │
   ├─▶ [0] Pending confirmation? ── yes/no → execute / cancel              (Phase 6 – todo)
   ├─▶ [1] Router (1 LLM call, JSON) → question | action | chitchat | off_topic       ✅
   ├── off_topic → fixed refusal (guardrail)                                          ✅
   ├── chitchat  → short LLM reply                                                    ✅
   ├── question  → ORCHESTRATOR → N question workers in parallel                      ✅
   │                 evidence (hybrid RAG) → draft → verify → retry → web fallback 🔴
   └── action    → ORCHESTRATOR → N file workers                         (Phases 5–6 – todo)
                     DSL spec + cookbook → YAML → verify (parse, fields, validate_file)
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
| Natural language → IDE actions | ⏳ Phase 5 (the pipeline is already proven end to end) |
| Human in the loop (optional) | ⏳ Phase 6 |

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

---

## 3. Testing

| Tool | Command | What it does |
|---|---|---|
| End-to-end tests | `docker compose up --build -d` then `uv run pytest tests -v` | Every grading criterion against the running container |
| Router | `uv run pytest tests/test_router.py -v -s` | Router accuracy on 30 labelled sentences |
| Verifier and workers | `uv run pytest tests/test_verifier.py -v` | Catches hallucinations, accepts paraphrases, planner splits, worker answers from the docs |
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

---

## 5. Next steps

| Phase | Work |
|---|---|
| **5: IDE actions** | Orchestrator → file workers: evidence (DSL spec + cookbook via RAG), YAML draft, verification (YAML parses, required DSL fields, `validate_file`), retry with errors, path cleanup, edits via `read_file` + `edit_file` |
| **6: Human in the loop** | Save delete and overwrite actions in `session.pending`; resolve yes/no in code before the router |
| **7: Hardening** | Timeouts, retries, full test pass against the container |
| **8: Ship** | Push `<user>/hyperion:latest`, test from a clean pull with `-e API_KEY=…`, README and demo script |

**Open questions for the organiser:**
- Will the evaluators pass `API_KEY` as an environment variable? (`.env` is not in the image.)
- Does the evaluation environment allow web access, which the web fallback needs?

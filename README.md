# ARGUS

**ARGUS** is a self-hosted AI support assistant for any software product. It answers user questions by combining:

- **RAG over your documentation.** Drop in PDFs, and ARGUS retrieves the most relevant passages for every question.
- **Live database context (optional).** Keyword-triggered, read-only SQL queries pull recent errors, job runs or audit events from your own PostgreSQL database.
- **Any OpenAI-compatible LLM.** Use vLLM, Ollama, LM Studio, OpenAI, or anything else that speaks `/v1/chat/completions`.

Responses stream token-by-token to a single-page web UI. The UI has a 3D animated background, a collapsible "thinking steps" view, text-to-speech, voice input and a conversation history.

```
Browser ──SSE──▶ Flask (app.py)
                   │ 1. Embed question → search ChromaDB (your docs)
                   │ 2. Keyword-matched SQL → PostgreSQL (optional)
                   │ 3. Build prompt (system + docs + DB rows + history)
                   │ 4. Stream from your LLM ──▶ tokens back to browser
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full technical details.

---

## Quick start

### 1. Prerequisites

- Python 3.10+
- An OpenAI-compatible LLM endpoint. For example, run locally with [Ollama](https://ollama.com):
  `ollama pull llama3.1` (served at `http://localhost:11434/v1`)
- *(Optional)* PostgreSQL, if you want live DB context

### 2. Install

```bash
git clone https://github.com/<your-username>/ARGUS.git
cd ARGUS
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~80 MB).

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env`. At minimum, set:

| Variable | Example |
|---|---|
| `PRODUCT_NAME` | `Acme Cloud` |
| `ESCALATION_CONTACT` | `support@acme.example` |
| `LLM_BASE_URL` | `http://localhost:11434/v1` (Ollama) |
| `LLM_MODEL` | `llama3.1` |
| `LLM_API_KEY` | your key, or `EMPTY` for local servers without auth |

`.env` is gitignored. Never commit it.

### 4. Add your knowledge base

Put your product documentation in a PDF at `knowledge_base.pdf`. You can also set `KB_PDF_PATH` to a different file or to a folder of PDFs. Then build the vector store:

```bash
python ingest.py
# or: python ingest.py docs/manual.pdf docs/faq.pdf
# or: python ingest.py docs/
```

Re-run this whenever your docs change. It rebuilds `chroma_db/` from scratch.

### 5. Run

```bash
python app.py
```

Open <http://localhost:5000> in Chrome or Firefox.

---

## Optional: live database context

ARGUS can look at real data while answering, for example "why did my export fail last night?" Each **source** in `db_sources.json` is a SQL query that runs when any of its keywords appear in the user's message. The returned rows are injected into the prompt.

**Try it with the bundled sample schema:**

```bash
createdb argus
psql argus -f examples/schema.sql        # creates app.error_logs, app.job_runs, app.audit_logs + seed rows
cp db_sources.example.json db_sources.json
# set DB_HOST / DB_NAME / DB_USER / DB_PASSWORD in .env
python app.py
```

Ask *"the nightly job failed, what happened?"*. ARGUS will pull the failed `nightly-export` run.

**Adapting to your own schema:** edit `db_sources.json`:

```json
{
  "sources": [
    {
      "name": "app_errors",
      "label": "application errors",
      "keywords": ["error", "fail", "broken"],
      "query": "SELECT id, code, description, created_at FROM my_schema.errors ORDER BY created_at DESC LIMIT 10"
    }
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Key under which the rows are given to the LLM |
| `label` | no | Human-readable name shown in the UI's "thinking steps" |
| `keywords` | yes | Case-insensitive substrings that trigger the query |
| `query` | yes | SQL to run. It takes no parameters, so keep a `LIMIT` on it |
| `query_if_empty` | no | Runs if `query` returns no rows (e.g. "failures first, otherwise recent runs") |
| `only_if_no_match` | no | Fallback: only runs when no regular source matched |

> **Security:** connect with a **read-only** database role (see the end of `examples/schema.sql`). Only select columns you are happy to send to your LLM provider. `db_sources.json` is gitignored because it describes your schema.

If `db_sources.json` doesn't exist, the DB feature is switched off and ARGUS runs on RAG alone.

---

## Customisation

| What | Where |
|---|---|
| Assistant / product name | `ASSISTANT_NAME`, `PRODUCT_NAME` in `.env`. The UI reads them from `/api/config` |
| Personality, rules, scope | Set `SYSTEM_PROMPT_FILE` to your own template. The placeholders `{assistant_name}`, `{product_name}` and `{escalation_contact}` are filled in. The default prompt is `DEFAULT_SYSTEM_PROMPT` in `app.py` |
| Logo | Replace `frontend/logo.svg` (52×30 ratio) |
| Avatar | Replace `frontend/avatar.svg`, or point `#argusImg` in `frontend/index.html` at a PNG |
| Welcome-screen quick questions | `QUICK_QUESTIONS` near the top of the `<script>` in `frontend/index.html` |
| Retrieval tuning | `RAG_TOP_K`, `RAG_MIN_SIMILARITY`, `EMBEDDING_MODEL` in `.env`. If you change the embedding model, re-run `ingest.py` |

---

## Configuration reference

All settings are environment variables, loaded from `.env`.

| Variable | Default | Description |
|---|---|---|
| `ASSISTANT_NAME` | `ARGUS` | Assistant's display name |
| `PRODUCT_NAME` | `your product` | Product the assistant supports |
| `ESCALATION_CONTACT` | `the support team` | Who to escalate to when the docs don't cover an issue |
| `SYSTEM_PROMPT_FILE` | *(unset)* | Path to a custom system-prompt template |
| `LLM_BASE_URL` | *(unset)* | Full OpenAI-compatible base URL. Overrides IP/port |
| `LLM_SERVER_IP` / `LLM_SERVER_PORT` | `127.0.0.1` / `8000` | Used as `http://IP:PORT/v1` when `LLM_BASE_URL` is unset |
| `LLM_API_KEY` | `EMPTY` | API key for the LLM endpoint |
| `LLM_MODEL` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Model name |
| `LLM_MAX_TOKENS` / `LLM_TEMPERATURE` | `1024` / `0.7` | Generation parameters |
| `KB_PDF_PATH` | `knowledge_base.pdf` | PDF file or folder for `ingest.py` |
| `KB_COLLECTION` | `argus_kb` | ChromaDB collection name |
| `CHROMA_DIR` | `chroma_db` | Vector-store directory |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `RAG_TOP_K` | `7` | Chunks retrieved per question |
| `RAG_MIN_SIMILARITY` | `0.28` | Chunks below this cosine similarity are dropped |
| `DB_SOURCES_FILE` | `db_sources.json` | DB source definitions |
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` | `localhost` / `5432` / `argus` / `argus` / *(empty)* | PostgreSQL connection |
| `APP_HOST` / `APP_PORT` | `127.0.0.1` / `5000` | Bind address. Use `0.0.0.0` only behind a reverse proxy |
| `FLASK_DEBUG` | `false` | Never enable on a network-reachable host |
| `LOG_FILE` | `argus.log` | Rotating log file (5 MB × 5) |

---

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | `{"session_id": "...", "message": "..."}` → SSE stream of `step`, `delta`, `done` and `error` events |
| `GET` | `/api/config` | `{"assistant_name", "product_name"}` for the UI |
| `GET` | `/api/health` | DB / RAG / model status |
| `GET` | `/api/sessions` | In-memory sessions with a preview of each |
| `DELETE` | `/api/sessions/<id>` | Delete a session |

---

## Before deploying publicly

ARGUS is designed as a self-hosted or internal tool. If you expose it to the internet:

- **Add authentication.** There is none built in, and `/api/sessions` lists every user's conversations.
- **Move sessions out of memory.** They are lost on restart and shared across all users of the process.
- **Sanitise rendered output.** LLM responses are rendered as Markdown into the page. Add [DOMPurify](https://github.com/cure53/DOMPurify) around `marked.parse(...)` in `frontend/index.html`, because your docs or DB rows could carry prompt-injected HTML.
- **Restrict CORS.** `CORS(app)` allows all origins.
- **Remember what gets logged.** The log file records user messages and response previews.
- **Run behind a production server.** Use something like `gunicorn -w 1 -k gthread --threads 8 app:app` behind a TLS reverse proxy. Keep a single worker, because sessions live in process memory.

## License

[MIT](LICENSE)

# ARGUS — Architecture

This document covers how ARGUS works internally. For setup and configuration, see the [README](../README.md).

**ARGUS** is an AI support assistant for any software product. It answers user questions by combining a RAG pipeline over your product documentation with optional live database context, then streams responses from any OpenAI-compatible LLM.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Architecture Overview](#2-architecture-overview)
3. [Technology Stack](#3-technology-stack)
4. [Backend — app.py](#4-backend--apppy)
5. [Ingestion Pipeline — ingest.py](#5-ingestion-pipeline--ingestpy)
6. [RAG Pipeline](#6-rag-pipeline)
7. [Database Context System](#7-database-context-system)
8. [LLM Integration](#8-llm-integration)
9. [Session Management](#9-session-management)
10. [Logging](#10-logging)
11. [Frontend — index.html](#11-frontend--indexhtml)
12. [SSE Streaming Protocol](#12-sse-streaming-protocol)
13. [API Endpoints](#13-api-endpoints)

---

## 1. Project Structure

```
ARGUS/
├── app.py                    # Flask backend — all server logic
├── ingest.py                 # PDF → ChromaDB ingestion script
├── requirements.txt          # Python dependencies
├── .env.example              # Config template (copy to .env)
├── db_sources.example.json   # DB context sources template (copy to db_sources.json)
├── examples/schema.sql       # Sample PostgreSQL schema + seed data
├── knowledge_base.pdf        # Your docs (gitignored; path set by KB_PDF_PATH)
├── argus.log                 # Rotating application log (auto-created)
├── chroma_db/                # Persistent ChromaDB vector store (created by ingest.py)
└── frontend/
    ├── index.html            # Single-page frontend (all HTML + CSS + JS)
    ├── avatar.svg            # Placeholder avatar
    └── logo.svg              # Placeholder logo
```

---

## 2. Architecture Overview

```
User Browser
     │
     │  HTTP / SSE
     ▼
┌─────────────────────────────────────────────────────┐
│                   Flask (app.py)                    │
│                                                     │
│  POST /api/chat                                     │
│       │                                             │
│  ┌────▼────────────────────────────────────────┐   │
│  │              generate() stream              │   │
│  │                                             │   │
│  │  Step 1: RAG Retrieval                      │   │
│  │    ├─ Embed query (all-MiniLM-L6-v2)        │   │
│  │    └─ Search ChromaDB → top-k chunks        │   │
│  │                                             │   │
│  │  Step 2: DB Context (optional)              │   │
│  │    └─ Keyword-triggered SQL queries         │   │
│  │       from db_sources.json → PostgreSQL     │   │
│  │                                             │   │
│  │  Step 3: Prompt Assembly                    │   │
│  │    └─ System prompt + RAG + DB + history    │   │
│  │                                             │   │
│  │  Step 4: LLM Streaming                      │   │
│  │    └─ OpenAI-compatible LLM API             │   │
│  │       delta tokens → SSE → browser          │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
         │                        │
    ChromaDB                 PostgreSQL
  (local disk)             (optional)
```

Every message follows a **4-step pipeline** inside a single SSE stream: the frontend receives step status events first, then the actual response token-by-token.

---

## 3. Technology Stack

### Backend
| Library | Version | Purpose |
|---|---|---|
| `flask` | ≥3.0.0 | HTTP server, SSE streaming, static file serving |
| `flask-cors` | ≥4.0.0 | Cross-origin headers for local dev |
| `openai` | ≥1.30.0 | OpenAI-compatible client for the LLM endpoint |
| `psycopg2-binary` | ≥2.9.9 | PostgreSQL driver for live DB context queries |
| `sentence-transformers` | ≥2.7.0 | Local embedding model (`all-MiniLM-L6-v2`) |
| `chromadb` | ≥0.5.0 | Local vector store for KB chunks |
| `pdfplumber` | ≥0.10.0 | PDF text extraction (used in ingest.py only) |
| `python-dotenv` | ≥1.0.0 | Loads `.env` into the environment |

### Frontend (CDN, no build step)
| Library | Purpose |
|---|---|
| `Three.js` v0.160.1 | Neural mesh 3D animation (WebGL) |
| `marked.js` v12.0.2 | Markdown → HTML rendering for assistant responses |
| `highlight.js` | Syntax highlighting inside code blocks |
| Web Speech API | Text-to-speech (`speechSynthesis`) + voice input (`SpeechRecognition`) |
| HTML5 Canvas 2D | Diagonal sine wave background animation + speaking waveform bars |

### Infrastructure
| Component | Details |
|---|---|
| LLM | Any OpenAI-compatible server (vLLM, Ollama, LM Studio, OpenAI, …), set by `LLM_BASE_URL` or `LLM_SERVER_IP`/`LLM_SERVER_PORT` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` — 80MB, runs on CPU locally |
| Vector DB | ChromaDB in embedded mode — no server, writes to `chroma_db/` on disk |
| App DB | Optional PostgreSQL, queried per `db_sources.json` |

---

## 4. Backend — app.py

### Startup sequence
1. `.env` is loaded, then logging is configured (rotating file handler + console handler)
2. `init_rag()` is called at module load — loads the embedding model and opens the ChromaDB collection
3. The LLM client is initialised against the configured endpoint
4. The system prompt is rendered (default or `SYSTEM_PROMPT_FILE`) and `db_sources.json` is loaded if present
5. Flask app starts on `APP_HOST:APP_PORT` (default `127.0.0.1:5000`)

### Request lifecycle for `POST /api/chat`
The entire processing happens inside a generator function `generate()` so that step events can be streamed to the browser before the LLM response begins:

```
generate()
  │
  ├─ yield step_event("Searching knowledge base...")
  ├─ retrieve_rag_context(user_message)
  ├─ yield step_event("Found N chunk(s)...")
  │
  ├─ yield step_event("Fetching live system data...")
  ├─ fetch_relevant_db_context(user_message)
  ├─ yield step_event("Fetched live data: ...")
  │
  ├─ assemble system_content = SYSTEM_PROMPT + rag_context + db_context
  │
  ├─ yield step_event("Generating response...")
  ├─ llm_client.chat.completions.create(stream=True)
  └─ for each token: yield SSE delta event
```

---

## 5. Ingestion Pipeline — ingest.py

Run **once** before starting `app.py`. Re-run if your documents change. Accepts PDF paths or folders as arguments, falling back to `KB_PDF_PATH`.

```bash
python3 ingest.py
```

### Steps

**Step 1 — PDF extraction**
`pdfplumber` reads each page and extracts raw text, storing `(page_number, text)` pairs. Empty pages are skipped.

**Step 2 — Chunking**
```
CHUNK_SIZE    = 600 characters  (~100–120 tokens)
CHUNK_OVERLAP = 80  characters
```
Each page's text is split on paragraph boundaries (`\n\n`). Paragraphs are accumulated into a buffer until `CHUNK_SIZE` is reached, then saved as a chunk. The last 80 characters of each chunk are carried into the next (overlap) to avoid losing context at boundaries. Long single paragraphs are hard-split at `CHUNK_SIZE`.

**Step 3 — Embedding**
Each chunk is embedded using `all-MiniLM-L6-v2`, producing a **384-dimensional float vector**.

**Step 4 — Storage**
Embeddings + documents + metadata are stored in a ChromaDB collection named `argus_kb` (`KB_COLLECTION`) with cosine similarity space (`hnsw:space: cosine`). Each record has:
- `id`: `chunk_0` … `chunk_N`
- `document`: raw chunk text
- `embedding`: 384-dim vector
- `metadata`: `{"page": <page_number>, "source": <pdf filename>}`

---

## 6. RAG Pipeline

### Configuration
```python
RAG_TOP_K           = 7     # candidates retrieved from ChromaDB
RAG_MIN_SIMILARITY  = 0.28  # minimum cosine similarity to be included
```

### Retrieval flow
1. Query text is embedded with the same `all-MiniLM-L6-v2` model
2. ChromaDB HNSW index returns top-7 nearest chunks by cosine distance
3. Each chunk's similarity is computed as `1 - cosine_distance`
4. Chunks below `0.28` similarity are dropped
5. Kept chunks are formatted as:
   ```
   [Knowledge Base — Page N]
   <chunk text>

   ---

   [Knowledge Base — Page M]
   <chunk text>
   ```
6. This string is appended to the system prompt under the heading `## <PRODUCT_NAME> Knowledge Base (retrieved — use this as your primary reference)`

### Why 0.28 threshold?
Cosine similarity for `all-MiniLM-L6-v2` on domain-specific text typically sits between 0.25–0.45 for relevant matches. Below 0.28 the retrieved text tends to be semantically unrelated and adds noise rather than signal.

---

## 7. Database Context System

DB context is optional. Sources are defined in `db_sources.json` (template: `db_sources.example.json`); if the file is absent the step is skipped entirely and no DB connection is opened.

### Matching
For each request the lowercased user message is checked against each source's `keywords` (substring match):

1. Every regular source with a keyword hit runs its `query`.
2. If a source's `query` returns no rows and it defines `query_if_empty`, that runs instead (e.g. "failures first, otherwise recent runs").
3. Only if **no** regular source matched, sources marked `only_if_no_match` are checked the same way — a generic "show me recent logs" fallback.

A DB connection is opened only when at least one source matched. Each query failure is logged and rolled back without affecting the other sources.

### Injection into prompt
Results are keyed by source `name`, serialised as JSON (`default=str` to handle datetime objects) and appended to the system prompt under `## Live Database Context (from the client's <PRODUCT_NAME> instance)`. The source `label` is shown in the UI's thinking steps.

---

## 8. LLM Integration

```text
Endpoint: LLM_BASE_URL, or http://<LLM_SERVER_IP>:<LLM_SERVER_PORT>/v1
Model   : LLM_MODEL
API key : LLM_API_KEY ("EMPTY" for local servers without auth)
```

Parameters per request:
```python
max_tokens  = LLM_MAX_TOKENS   # default 1024
temperature = LLM_TEMPERATURE  # default 0.7
stream      = True
```

The `openai` Python SDK is used with a custom `base_url`, making the integration compatible with any OpenAI-compatible inference server (vLLM, LM Studio, Ollama, etc.).

### System prompt design
The default prompt (`DEFAULT_SYSTEM_PROMPT` in `app.py`) defines the assistant's personality, instructs it to answer only from retrieved context (never from general knowledge), forbids mentioning page numbers or the retrieval system, keeps it in scope for `PRODUCT_NAME`, and specifies response formatting (bold terms, numbered steps, no code blocks). Override it with `SYSTEM_PROMPT_FILE`; `{assistant_name}`, `{product_name}` and `{escalation_contact}` are substituted.

---

## 9. Session Management

Sessions are stored in-memory as a Python dict:
```python
sessions: dict[str, list[dict]] = {}
# { "session_id": [{"role": "user", "content": "..."}, ...] }
```

- Session ID is a random ID generated client-side on page load (not persisted — a reload starts a new session)
- Conversation history (last 40 messages) is sent to the LLM on every request
- History is trimmed to the last 40 messages to avoid exceeding context window limits
- Sessions are lost on server restart and are shared by everyone using the process (in-memory only, no auth)

---

## 10. Logging

Configured with Python's `logging` module using a `RotatingFileHandler`:

```
Log file    : argus.log (LOG_FILE)
Max size    : 5 MB per file
Backup count: 5 files (argus.log, argus.log.1, … argus.log.5)
Encoding    : UTF-8
```

Both the log file and console receive all messages. Every chat request logs:
- User message + session ID + history length
- RAG: query, similarity scores, kept/dropped chunks
- DB: which tables were queried and how many rows returned
- Prompt sizes (base, RAG contribution, DB contribution)
- LLM response length and preview

---

## 11. Frontend — index.html

A single self-contained HTML file. No build step, no npm, no framework.

### Visual components

**Branding**
- On load the page fetches `/api/config` and fills every `.js-assistant` / `.js-product` element, the page title and the input placeholder
- Welcome-screen quick questions come from the `QUICK_QUESTIONS` array

**Top bar**
- Logo (`logo.svg`)
- Assistant name, product badge, and a "DB Connected" badge when `/api/health` reports the DB as connected

**Left panel — avatar area**
- `avatar.svg` rendered inside a circular clip (`border-radius: 50%`, `overflow: hidden`)
- Image zoomed with `transform: scale(1.44)` and `transform-origin: center 45%`
- Two animated ring borders (`#argusRing`, `#argusRing2`)
- Speaking waveform bars (`#waveCanvas`) overlay the avatar while text-to-speech is playing

**Three.js neural mesh**
- `IcosahedronGeometry(1.8, 3)` — 642 vertices
- `brain.scale.set(1.12, 0.90, 0.82)` — uniform oval deformation (symmetric, no silhouette change during rotation)
- Vertex colours: purple (`rgb(0.65, 0.38, 1.0)`) → cyan (`rgb(0.12, 0.88, 1.0)`) based on normalised Y position
- `Points` layer: node size 0.060, opacity 0.98
- `LineSegments` layer: edges between vertices within distance threshold, opacity 0.38
- Rotation: Y-axis only at `+0.0018 rad/frame`
- Renderer: `antialias: false`, `powerPreference: 'high-performance'`, `pixelRatio: min(dpr, 1)`

**Diagonal wave background**
4 sine wave layers drawn on `#waveBg` canvas, rotated `Math.PI/6` (30°) for diagonal effect:
```javascript
{ amp:42, freq:0.0055, speed:0.55, yFrac:0.52, alpha:0.28, lw:1.6 }
{ amp:58, freq:0.0040, speed:0.35, yFrac:0.56, alpha:0.14, lw:2.0 }
{ amp:36, freq:0.0065, speed:0.65, yFrac:0.47, alpha:0.18, lw:1.3 }
{ amp:50, freq:0.0045, speed:0.28, yFrac:0.60, alpha:0.11, lw:1.8 }
```
Waves are redrawn every 4th animation frame to reduce CPU load.

**Right panel — conversation feed**
- `#msgFeed`: scrollable (`overflow-y: auto`) message list
- User messages: right-aligned purple gradient cards
- Assistant messages: left-aligned glassmorphism cards with `backdrop-filter: blur()`
- Markdown rendered via `marked.js`; code blocks syntax-highlighted via `highlight.js`
- Quick-question chips shown when no conversation is active

**Thinking steps card (inside the assistant card)**
When a response is in progress, a collapsible "Processing" block appears at the top of the assistant card showing each pipeline step in real time:
- Cyan pulsing dot while processing → turns green on completion
- Steps animate in with a spinning `⟳` icon → `✓` when done
- On first response token: steps body collapses via CSS transition (`max-height: 0`)
- Drop-down arrow (`▾`) lets user re-expand the steps at any time

**Input bar**
- Auto-resizing `<textarea>` — grows with content, scrolls at max height
- Send button (arrow icon)
- Voice input button — `SpeechRecognition` API, toggles mic on/off

**Voice features**
- **Text-to-speech**: `window.speechSynthesis` reads responses aloud. Note: blocked by Brave browser's fingerprinting protection — works in Chrome/Firefox.
- **Speech-to-text**: `SpeechRecognition` / `webkitSpeechRecognition` transcribes microphone input and fills the message input box.

**History sidebar**
- Lists this page's sessions (first message preview)
- Click to switch the active session ID (messages already in the feed are not reloaded)
- Delete button per session

### Performance optimisations
- DOM references for the RAF loop cached outside `loop()` (no `getElementById` inside the animation loop)
- Wave canvas redrawn every 4th frame only
- WebGL renderer: `antialias: false`, pixel ratio capped at 1
- `powerPreference: 'high-performance'` hint to GPU

---

## 12. SSE Streaming Protocol

All chat responses use **Server-Sent Events** over a single HTTP connection. The frontend reads the stream with `response.body.getReader()`.

### Event types

```jsonc
// Pipeline step update (before LLM response starts)
{ "type": "step", "text": "Searching knowledge base..." }

// LLM response token
{ "type": "delta", "content": "The ", "session_id": "ss_..." }

// Stream complete
{ "type": "done", "session_id": "ss_..." }

// Error
{ "type": "error", "content": "I ran into a connection issue..." }
```

### Frontend handling
1. `step` events → append a step line to the thinking block inside the assistant card
2. First `delta` event → `finaliseSteps()` collapses the thinking block, then tokens stream into `mcard-body`
3. End of stream → triggers `speakText()` for TTS playback
4. `error` event → displays error message in the card

---

## 13. API Endpoints

### `POST /api/chat`
Main chat endpoint. Returns an SSE stream.

**Request body:**
```json
{
  "session_id": "ss_abc123",
  "message": "How do I reset a user's password?"
}
```

**Response:** `text/event-stream` — sequence of SSE events (see §12)

---

### `GET /api/health`
Health check. Returns system status.

**Response:**
```jsonc
{
  "status": "ok",
  "db": "connected",          // or "unavailable" / "disabled"
  "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
  "rag": "ready",
  "rag_chunks": 312
}
```

---

### `GET /api/sessions`
Lists all active in-memory sessions.

**Response:**
```json
[
  { "id": "ss_abc123", "preview": "What are the actions of...", "message_count": 4 }
]
```

---

### `DELETE /api/sessions/<session_id>`
Deletes a session from memory.

**Response:** `{ "success": true }`

---

### `GET /api/config`
Branding values for the UI.

**Response:** `{ "assistant_name": "ARGUS", "product_name": "Acme Cloud" }`

---

### `GET /`
Serves `frontend/index.html`.

import json
import logging
import os
import traceback
import uuid
from logging.handlers import RotatingFileHandler

import chromadb
import psycopg2
import psycopg2.extras
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from dotenv import load_dotenv
from flask_cors import CORS
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = os.getenv("LOG_FILE", "argus.log")

logger = logging.getLogger("argus")
logger.setLevel(logging.DEBUG)

# File handler — rotates at 5 MB, keeps last 5 files
_fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

# Console handler — mirrors everything to terminal as before
_ch = logging.StreamHandler()
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(logging.Formatter("%(message)s"))   # keep terminal clean (no timestamp prefix)

logger.addHandler(_fh)
logger.addHandler(_ch)

def log(msg: str, level: str = "info"):
    getattr(logger, level)(msg)

# ── RAG config & startup ───────────────────────────────────────────────────────
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("KB_COLLECTION", "argus_kb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "7"))
RAG_MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.28"))   # chunks below this score are discarded as irrelevant

_rag_model: SentenceTransformer | None = None
_rag_collection = None


def init_rag():
    global _rag_model, _rag_collection
    try:
        _rag_model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _rag_collection = client.get_collection(COLLECTION_NAME)
        log(f"[RAG] Ready — {_rag_collection.count()} chunks loaded from ChromaDB")
    except Exception as exc:
        log(f"[RAG] Init failed: {exc}. Run 'python ingest.py' first.")


def retrieve_rag_context(query: str, top_k: int = RAG_TOP_K) -> str:
    """Embed query, search ChromaDB, return formatted context string."""
    if not _rag_model or not _rag_collection:
        log("[RAG] ⚠️  Model or collection not loaded — skipping retrieval")
        return ""
    try:
        log(f"\n{'='*60}")
        log(f"[RAG] 🔍 Query: \"{query}\"")
        log(f"[RAG] Embedding query with {EMBEDDING_MODEL} ...")

        embedding = _rag_model.encode([query]).tolist()

        log(f"[RAG] Searching ChromaDB for top {top_k} chunks ...")
        results = _rag_collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]   # cosine distance (lower = more similar)

        log(f"[RAG] ✅ Retrieved {len(docs)} chunks (filtering below similarity={RAG_MIN_SIMILARITY}):\n")
        parts = []
        dropped = 0
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            similarity = 1 - dist   # cosine distance → similarity score
            preview = doc.replace("\n", " ")[:120]
            if similarity < RAG_MIN_SIMILARITY:
                log(f"  [{i}] Page {meta['page']} | similarity={similarity:.3f} ❌ DROPPED (below threshold)")
                dropped += 1
                continue
            log(f"  [{i}] Page {meta['page']} | similarity={similarity:.3f} ✅ KEPT")
            log(f"       \"{preview}...\"")
            parts.append(f"[Knowledge Base — Page {meta['page']}]\n{doc}")

        if dropped:
            log(f"\n[RAG] {dropped} chunk(s) dropped, {len(parts)} chunk(s) used")
        if not parts:
            log(f"[RAG] ⚠️  All chunks below threshold — no KB context injected")
        log(f"{'='*60}\n")
        return "\n\n---\n\n".join(parts)

    except Exception as exc:
        log(f"[RAG] ❌ Retrieval error: {exc}")
        return ""


init_rag()

# ── LLM config ──────────────────────────────────────────────────────────────
openai_api_key = os.getenv("LLM_API_KEY", "EMPTY")
open_api_ip = os.getenv("LLM_SERVER_IP", "127.0.0.1")
open_api_port = os.getenv("LLM_SERVER_PORT", "8000")
# LLM_BASE_URL wins if set (e.g. https://api.openai.com/v1 or a remote vLLM behind TLS)
openai_api_base = os.getenv("LLM_BASE_URL") or f"http://{open_api_ip}:{open_api_port}/v1"
open_api_model = os.getenv("LLM_MODEL", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

llm_client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

# ── In-memory session store (replace with DB persistence in production) ──────
sessions: dict[str, list[dict]] = {}

# ── Branding ──────────────────────────────────────────────────────────────────
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "ARGUS")
PRODUCT_NAME = os.getenv("PRODUCT_NAME", "your product")
ESCALATION_CONTACT = os.getenv("ESCALATION_CONTACT", "the support team")

# ── System prompt ─────────────────────────────────────────────────────────────
# Placeholders {assistant_name}, {product_name} and {escalation_contact} are
# filled from the env vars above. Point SYSTEM_PROMPT_FILE at your own template
# to replace the default prompt entirely.
DEFAULT_SYSTEM_PROMPT = """You are {assistant_name}, a senior support consultant for {product_name}.

## Your Personality
You're warm, empathetic, and genuinely invested in solving the client's problem. You talk like a real, experienced human — not a robotic assistant. You're patient, thorough, and methodical. When a client is frustrated, you acknowledge their frustration before jumping to solutions. When an issue is resolved, you celebrate with them. You use light professional humour when the moment allows, but you're always focused on resolution.

Do NOT start every reply with "Certainly!" or "Of course!" — vary your openers naturally, the way a real person would.

## Knowledge & Accuracy
Your answers must be grounded exclusively in:
1. The **{product_name} Knowledge Base** passages injected below (labelled "Knowledge Base — retrieved")
2. The **live database context** injected below (labelled "Live Database Context"), when present

Do NOT answer from general knowledge or assumptions about how {product_name} works. If the retrieved passages do not contain enough information to answer confidently, say so honestly and suggest the client contact {escalation_contact} for clarification. Never invent feature names, file names, configuration values, or steps that are not present in the retrieved context.

Never reference the knowledge base, page numbers, or retrieval system in your responses — the client should not know how you work internally. Answer as a human expert would, directly and naturally.

## How You Work
1. **Understand first** — Ask clarifying questions before jumping to fixes. What's the exact error? When did it start? What changed recently?
2. **Use the knowledge base** — Base every technical answer on the retrieved passages. Never mention the knowledge base, page numbers, or that you are retrieving context — just answer naturally as if you already know this.
3. **Step-by-step guidance** — Always number your troubleshooting steps clearly.
4. **Confirm resolution** — End every resolution by asking if the issue is fully sorted.
5. **Escalation awareness** — If the retrieved context doesn't cover the issue, say so clearly and advise escalating to {escalation_contact}.
6. **Proactive follow-up** — Suggest related checks after the main issue is resolved, but only if supported by the knowledge base.

## Response Style
- Use **bold** for important terms, steps, or error names
- Use numbered lists for multi-step instructions
- Keep responses concise unless detail is needed — don't pad
- Acknowledge the client's effort: "Good call checking that first", "That's actually a really common one"
- If the knowledge base doesn't cover something, say so honestly — never guess
- **Never include code blocks, code snippets, scripts, or commands in your responses.** If a configuration value or command needs to be mentioned, describe it in plain text only.

## Scope
You only answer questions related to **{product_name}** — its features, configuration, troubleshooting, integrations, and usage. If a user asks about anything outside of {product_name} (e.g. general programming, unrelated software, personal questions, or any other topic), politely decline and redirect them:

"I'm here specifically to help with {product_name}. For other questions, I'd suggest reaching out to the appropriate resource. Is there anything about {product_name} I can help you with?"

Do not attempt to partially answer out-of-scope questions. Decline clearly, briefly, and without judgement.

## Database Context
When live database records are injected into this conversation, treat them as real data from the client's {product_name} instance. Reference specific IDs, timestamps, and error messages in your analysis.

You are the face of {product_name} support. Every interaction is a chance to build trust. Be accurate above all else.
"""


def load_system_prompt() -> str:
    template = DEFAULT_SYSTEM_PROMPT
    prompt_file = os.getenv("SYSTEM_PROMPT_FILE")
    if prompt_file:
        try:
            with open(prompt_file, encoding="utf-8") as f:
                template = f.read()
            log(f"[PROMPT] Loaded system prompt from {prompt_file}")
        except OSError as exc:
            log(f"[PROMPT] Could not read {prompt_file}: {exc} — using default prompt", "warning")
    # str.replace (not str.format) so literal braces in a custom prompt are safe
    for key, value in {
        "{assistant_name}": ASSISTANT_NAME,
        "{product_name}": PRODUCT_NAME,
        "{escalation_contact}": ESCALATION_CONTACT,
    }.items():
        template = template.replace(key, value)
    return template


SYSTEM_PROMPT = load_system_prompt()


# ── DB context sources ────────────────────────────────────────────────────────
# Which tables to query, and when, is defined in a JSON file (see
# db_sources.example.json) so no schema is hardcoded here. If the file is
# missing, live DB context is simply disabled and the app runs on RAG alone.
DB_SOURCES_FILE = os.getenv("DB_SOURCES_FILE", "db_sources.json")


def load_db_sources(path: str = DB_SOURCES_FILE) -> list[dict]:
    if not os.path.exists(path):
        log(f"[DB] No {path} found — live DB context disabled")
        return []
    try:
        with open(path, encoding="utf-8") as f:
            sources = json.load(f).get("sources", [])
    except (OSError, ValueError) as exc:
        log(f"[DB] Could not parse {path}: {exc} — live DB context disabled", "warning")
        return []
    valid = []
    for src in sources:
        if not src.get("name") or not src.get("query") or not src.get("keywords"):
            log(f"[DB] Skipping source without name/query/keywords: {src}", "warning")
            continue
        src["keywords"] = [k.lower() for k in src["keywords"]]
        valid.append(src)
    log(f"[DB] Loaded {len(valid)} source(s) from {path}")
    return valid


DB_SOURCES = load_db_sources()
DB_SOURCE_LABELS = {src["name"]: src.get("label", src["name"]) for src in DB_SOURCES}


# ── Database helpers ──────────────────────────────────────────────────────────
def get_db_connection():
    try:
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME", "argus"),
            user=os.getenv("DB_USER", "argus"),
            password=os.getenv("DB_PASSWORD", ""),
        )
    except Exception as exc:
        log(f"[DB] Connection failed: {exc}")
        return None


def _run_source_query(cursor, conn, src: dict) -> list[dict]:
    try:
        cursor.execute(src["query"])
        rows = cursor.fetchall()
        if not rows and src.get("query_if_empty"):
            # e.g. "show failures first; if there are none, show recent runs"
            cursor.execute(src["query_if_empty"])
            rows = cursor.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log(f"[DB] {src['name']} query failed: {e}")
        conn.rollback()
        return []


def fetch_relevant_db_context(message: str) -> dict:
    """Run the configured DB sources whose keywords appear in the user's message."""
    if not DB_SOURCES:
        return {}

    msg_lower = message.lower()
    primary = [s for s in DB_SOURCES if not s.get("only_if_no_match")]
    fallback = [s for s in DB_SOURCES if s.get("only_if_no_match")]

    matched = [s for s in primary if any(w in msg_lower for w in s["keywords"])]
    if not matched:
        matched = [s for s in fallback if any(w in msg_lower for w in s["keywords"])]
    if not matched:
        return {}

    conn = get_db_connection()
    if not conn:
        return {}

    context: dict = {}
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        for src in matched:
            rows = _run_source_query(cursor, conn, src)
            if rows:
                context[src["name"]] = rows
                log(f"[DB] {src['name']} → {len(rows)} rows")
        return context
    except Exception as exc:
        log(f"[DB] context query failed: {exc}")
        return {}
    finally:
        conn.close()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    session_id: str = data.get("session_id") or str(uuid.uuid4())
    user_message: str = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    if session_id not in sessions:
        sessions[session_id] = []

    log(f"\n{'#'*60}")
    log(f"[CHAT] 📩 User message : \"{user_message}\"")
    log(f"[CHAT] Session        : {session_id}")
    log(f"[CHAT] History length : {len(sessions[session_id])} messages")

    def step_event(text: str) -> str:
        """Yield a step SSE event to show in the frontend thinking card."""
        return f"data: {json.dumps({'type': 'step', 'text': text})}\n\n"

    def generate():
        full_response = ""
        token_count = 0
        try:
            # ── Step 1: RAG retrieval ─────────────────────────────────────────
            log(f"\n[STEP 1] RAG — Knowledge Base retrieval")
            yield step_event("Searching knowledge base...")
            rag_context = retrieve_rag_context(user_message)
            if rag_context:
                chunk_count = rag_context.count("\n\n---\n\n") + 1
                yield step_event(f"Found {chunk_count} relevant chunk(s) from knowledge base")
            else:
                log("[STEP 1] ⚠️  No RAG context retrieved (collection empty or model not loaded)")
                yield step_event("No knowledge base match found")

            # ── Step 2: DB context ────────────────────────────────────────────
            log(f"[STEP 2] DB — fetching relevant live context ...")
            yield step_event("Fetching live system data...")
            db_context = fetch_relevant_db_context(user_message)
            if db_context:
                labels = [DB_SOURCE_LABELS.get(k, k) for k in db_context.keys()]
                yield step_event(f"Fetched live data: {', '.join(labels)}")
                log(f"[STEP 2] ✅ DB context keys: {list(db_context.keys())}")
            else:
                log(f"[STEP 2] ℹ️  No DB context matched (DB unavailable or no keyword hit)")
                yield step_event("No live system data retrieved")

            # ── Step 3: Build prompt ──────────────────────────────────────────
            log(f"\n[STEP 3] Building prompt ...")
            system_content = SYSTEM_PROMPT
            if rag_context:
                system_content += (
                    f"\n\n## {PRODUCT_NAME} Knowledge Base (retrieved — use this as your primary reference)\n\n"
                    + rag_context
                )
            if db_context:
                system_content += (
                    f"\n\n## Live Database Context (from the client's {PRODUCT_NAME} instance)\n"
                    + json.dumps(db_context, default=str, indent=2)
                )

            messages = [{"role": "system", "content": system_content}]
            messages.extend(sessions[session_id])
            messages.append({"role": "user", "content": user_message})

            rag_chars = len(rag_context) if rag_context else 0
            db_chars  = len(json.dumps(db_context, default=str)) if db_context else 0
            log(f"[STEP 3] System prompt size : {len(system_content)} chars")
            log(f"          ├─ Base prompt    : {len(SYSTEM_PROMPT)} chars")
            log(f"          ├─ RAG context    : {rag_chars} chars")
            log(f"          └─ DB context     : {db_chars} chars")
            log(f"[STEP 3] Total messages to LLM : {len(messages)} (incl. system)")

            # ── Step 4: LLM streaming ─────────────────────────────────────────
            log(f"\n[STEP 4] Sending to LLM ({open_api_model}) — streaming ...")
            yield step_event("Generating response...")

            stream = llm_client.chat.completions.create(
                model=open_api_model,
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_response += delta
                    token_count += 1
                    yield f"data: {json.dumps({'type': 'delta', 'content': delta, 'session_id': session_id})}\n\n"

            # Persist to session history
            sessions[session_id].append({"role": "user", "content": user_message})
            sessions[session_id].append({"role": "assistant", "content": full_response})
            if len(sessions[session_id]) > 40:
                sessions[session_id] = sessions[session_id][-40:]

            log(f"[STEP 4] ✅ LLM done — {len(full_response)} chars, ~{token_count} chunks streamed")
            log(f"[CHAT] 💬 Response preview: \"{full_response[:120].replace(chr(10), ' ')}...\"")
            log(f"{'#'*60}\n")

            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"

        except Exception:
            log(traceback.format_exc())
            log(f"[STEP 4] ❌ LLM streaming failed")
            yield f"data: {json.dumps({'type': 'error', 'content': 'I ran into a connection issue. Please try again in a moment.'})}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    result = []
    for sid, msgs in sessions.items():
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        if first_user:
            result.append(
                {
                    "id": sid,
                    "preview": first_user[:70] + ("…" if len(first_user) > 70 else ""),
                    "message_count": len(msgs),
                }
            )
    return jsonify(result)


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return jsonify({"success": True})


@app.route("/api/config", methods=["GET"])
def ui_config():
    """Branding values the frontend renders (assistant name, product name)."""
    return jsonify({"assistant_name": ASSISTANT_NAME, "product_name": PRODUCT_NAME})


@app.route("/api/health", methods=["GET"])
def health():
    db_ok = False
    try:
        conn = get_db_connection() if DB_SOURCES else None
        if conn:
            conn.close()
            db_ok = True
    except Exception:
        pass
    rag_ok = _rag_collection is not None
    rag_chunks = _rag_collection.count() if rag_ok else 0
    return jsonify({
        "status": "ok",
        "db": "connected" if db_ok else ("disabled" if not DB_SOURCES else "unavailable"),
        "model": open_api_model,
        "rag": "ready" if rag_ok else "not_ingested",
        "rag_chunks": rag_chunks,
    })


if __name__ == "__main__":
    app.run(
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes"),
    )

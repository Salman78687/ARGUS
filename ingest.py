"""
ingest.py — One-time script to parse your knowledge-base PDF(s),
chunk them, embed with sentence-transformers, and store in ChromaDB.

Run once (or whenever the documents change):
    python ingest.py                     # uses KB_PDF_PATH from .env
    python ingest.py docs/manual.pdf     # or pass one or more PDFs / folders
"""

import glob
import os
import re
import sys

import chromadb
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

PDF_PATH = os.getenv("KB_PDF_PATH", "knowledge_base.pdf")   # a PDF file or a folder of PDFs
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("KB_COLLECTION", "argus_kb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 600   # characters — ~100-120 tokens, good semantic unit
CHUNK_OVERLAP = 80  # characters of overlap between consecutive chunks


# ── PDF text extraction ───────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return list of (page_number, text) for every non-empty page."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append((i + 1, text.strip()))
    return pages


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split page text into overlapping chunks.
    Tries to break on paragraph boundaries first; falls back to hard split.
    Returns list of {"text": str, "page": int}.
    """
    chunks = []

    for page_num, text in pages:
        # Normalise whitespace while preserving paragraph breaks
        text = re.sub(r"\n{3,}", "\n\n", text)
        paragraphs = re.split(r"\n\n+", text)

        buffer = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(buffer) + len(para) + 1 <= chunk_size:
                buffer = (buffer + " " + para).strip() if buffer else para
            else:
                if buffer:
                    chunks.append({"text": buffer, "page": page_num})
                    # carry overlap into next chunk
                    buffer = buffer[-overlap:].strip() + " " + para
                    buffer = buffer.strip()
                else:
                    buffer = para

                # If buffer is already over chunk_size (long paragraph), hard-split it
                while len(buffer) > chunk_size:
                    chunks.append({"text": buffer[:chunk_size], "page": page_num})
                    buffer = buffer[chunk_size - overlap :]

        if buffer:
            chunks.append({"text": buffer, "page": page_num})

    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def resolve_pdfs(paths: list[str]) -> list[str]:
    """Expand folders into the PDFs they contain."""
    pdfs = []
    for path in paths:
        if os.path.isdir(path):
            pdfs.extend(sorted(glob.glob(os.path.join(path, "**", "*.pdf"), recursive=True)))
        elif os.path.exists(path):
            pdfs.append(path)
        else:
            raise FileNotFoundError(f"PDF not found: {path}")
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {', '.join(paths)}")
    return pdfs


def main():
    pdfs = resolve_pdfs(sys.argv[1:] or [PDF_PATH])

    chunks = []
    print(f"[1/4] Extracting text from {len(pdfs)} PDF(s) ...")
    for pdf in pdfs:
        pages = extract_pages(pdf)
        print(f"      → {pdf}: {len(pages)} pages extracted")

        print("[2/4] Chunking ...")
        pdf_chunks = chunk_pages(pages)
        for c in pdf_chunks:
            c["source"] = os.path.basename(pdf)
        chunks.extend(pdf_chunks)
        print(f"      → {len(pdf_chunks)} chunks created")

    print(f"[3/4] Loading embedding model ({EMBEDDING_MODEL}) ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("[4/4] Embedding & storing in ChromaDB ...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("      (deleted existing collection)")
    except Exception:
        pass

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"page": c["page"], "source": c["source"]} for c in chunks],
    )

    print(f"\nDone! {len(chunks)} chunks stored in '{CHROMA_DIR}/'")
    print("You can now start app.py — RAG will be active automatically.")


if __name__ == "__main__":
    main()

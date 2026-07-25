"""
embedder.py
------------
Turns text chunks into dense embedding vectors, with disk caching so
the (slow, one-time) embedding step does NOT re-run on every server
restart — only when the underlying data or the model actually changed.

Config (read from backend/.env via python-dotenv, loaded in main.py):

    RETRIEVAL_BACKEND     "tfidf" (default) or "embeddings"
                          Which engine rag.py uses to search.

    EMBEDDING_MODEL       sentence-transformers model name.
                          default: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                          (multilingual, works reasonably on Arabic; needs
                          internet the FIRST time only, to download weights).

    EMBEDDING_CACHE_DIR   folder where the cached vectors (.npy) live.
                          default: backend/app/.cache

    FORCE_REEMBED         "true" / "1" -> ignore any existing cache and
                          rebuild the embeddings from scratch. Set this
                          to true once after you edit all_data.txt if you
                          ever want to force a rebuild manually, then set
                          it back to false (the cache would auto-detect
                          the data change anyway — this flag is only for
                          "I want to force it regardless").

How the cache invalidation works:
    A fingerprint is computed from ALL chunk texts + the embedding model
    name (sha256, truncated). If a cached file with that exact
    fingerprint exists, it's loaded straight from disk (fast, no model
    calls). If the data file changes even by one character, or the
    model name changes, the fingerprint changes automatically and a
    fresh embedding pass runs and gets cached under the new fingerprint.
    Old cache files are left on disk (harmless) unless you clean them up.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "tfidf").strip().lower()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
CACHE_DIR = Path(
    os.getenv("EMBEDDING_CACHE_DIR", str(Path(__file__).resolve().parent / ".cache"))
)
FORCE_REEMBED = os.getenv("FORCE_REEMBED", "false").strip().lower() in ("1", "true", "yes")

# Ensure the cache folder exists before writing embedding files.
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Lazy-loaded SentenceTransformer model instance.
_model = None  # only created when the embeddings backend is used


def _data_fingerprint(texts: List[str], model_name: str) -> str:
    # Create a stable fingerprint for the current data and model.
    # If the text or model changes, the fingerprint changes too.
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    h.update(model_name.encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_paths(fingerprint: str):
    # Map the fingerprint to cache file paths for vectors and metadata.
    return (
        CACHE_DIR / f"embeddings_{fingerprint}.npy",
        CACHE_DIR / f"embeddings_{fingerprint}.meta.json",
    )


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # heavy import, only when needed

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _encode_with_model(texts: List[str]) -> np.ndarray:
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,  # so cosine similarity == dot product
    )
    return np.asarray(vectors, dtype="float32")


def embed_texts(texts: List[str], _encode_fn=None) -> np.ndarray:
    """Return an (n_texts, dim) embedding matrix.

    Loads from the on-disk cache when the data+model fingerprint
    matches a previous run; otherwise computes fresh embeddings and
    writes them to the cache.

    _encode_fn: internal hook used only by tests to swap in a fake,
    fast encoder instead of downloading a real model — not meant to be
    passed by application code.
    """
    encode_fn = _encode_fn or _encode_with_model
    fingerprint = _data_fingerprint(texts, EMBEDDING_MODEL)
    npy_path, meta_path = _cache_paths(fingerprint)

    if not FORCE_REEMBED and npy_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("model") == EMBEDDING_MODEL and meta.get("count") == len(texts):
                return np.load(npy_path)
        except Exception:
            pass  # corrupted/partial cache -> fall through and rebuild

    # Compute and cache embeddings when no valid cache exists.
    matrix = encode_fn(texts)
    np.save(npy_path, matrix)
    meta_path.write_text(
        json.dumps(
            {"model": EMBEDDING_MODEL, "count": len(texts), "fingerprint": fingerprint}
        ),
        encoding="utf-8",
    )
    return matrix


def embed_query(query: str) -> np.ndarray:
    # Embed a single query string with the same model and normalization.
    # This is used at runtime to compare user questions against cached chunks.
    model = _get_model()
    vec = model.encode([query], normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")

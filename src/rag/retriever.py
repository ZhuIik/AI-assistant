import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from src.config import EMBED_MODEL

INDEX_DIR = Path("data/datasets/rag_index")

_embedder = None
_store = None
_lexical_meta: Optional[list[dict]] = None
_lexical_doc_freq: Optional[Counter] = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", str(text).lower())


def _load_vector_stack():
    global _embedder, _store
    if _embedder is not None and _store is not None:
        return True

    manifest_path = INDEX_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("vector_built"):
            return False

    try:
        from .embedder import Embedder
        from .vector_store import VectorStore
    except ModuleNotFoundError:
        return False

    index_file = INDEX_DIR / "index.faiss"
    if not index_file.exists():
        return False

    _embedder = Embedder(EMBED_MODEL)
    _store = VectorStore.load(INDEX_DIR)
    return True


def _load_lexical_meta():
    global _lexical_meta, _lexical_doc_freq
    if _lexical_meta is not None and _lexical_doc_freq is not None:
        return

    meta_path = INDEX_DIR / "meta.json"
    if not meta_path.exists():
        _lexical_meta = []
        _lexical_doc_freq = Counter()
        return

    _lexical_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    doc_freq = Counter()
    for row in _lexical_meta:
        doc_freq.update(set(_tokenize(row.get("search_text") or row.get("text") or "")))
    _lexical_doc_freq = doc_freq


def _lexical_score(row: dict, query_terms: Counter, total_docs: int) -> float:
    if not query_terms or not total_docs or not _lexical_doc_freq:
        return 0.0

    text_terms = Counter(_tokenize(row.get("search_text") or row.get("text") or ""))
    score = 0.0
    for term, query_count in query_terms.items():
        term_count = text_terms.get(term, 0)
        if not term_count:
            continue
        idf = math.log((total_docs + 1) / (_lexical_doc_freq[term] + 1)) + 1
        score += query_count * (1 + math.log(term_count)) * idf
    return score


def _lexical_search(question: str, top_k: int) -> list[dict]:
    _load_lexical_meta()
    if not _lexical_meta or not _lexical_doc_freq:
        return []

    query_terms = Counter(_tokenize(question))
    if not query_terms:
        return _lexical_meta[:top_k]

    total_docs = len(_lexical_meta)
    scored = []
    for row in _lexical_meta:
        score = _lexical_score(row, query_terms, total_docs)
        if score:
            scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:top_k]]


def _hybrid_rerank(question: str, vector_rows: list[dict], top_k: int) -> list[dict]:
    lexical_rows = _lexical_search(question, top_k=max(top_k * 5, 20))
    _load_lexical_meta()

    query_terms = Counter(_tokenize(question))
    total_docs = len(_lexical_meta or [])
    merged: dict[str, tuple[dict, float]] = {}

    for rank, row in enumerate(vector_rows, start=1):
        row_id = row.get("id") or f"vector-{rank}"
        merged[row_id] = (row, 1.0 / rank)

    for rank, row in enumerate(lexical_rows, start=1):
        row_id = row.get("id") or f"lexical-{rank}"
        existing_row, existing_bonus = merged.get(row_id, (row, 0.0))
        merged[row_id] = (existing_row, existing_bonus + 2.0 / rank)

    scored = []
    for row, rank_bonus in merged.values():
        lexical = _lexical_score(row, query_terms, total_docs)
        scored.append((lexical + rank_bonus, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:top_k]]


def get_relevant_chunks(question: str, top_k: int = 5) -> list[dict]:
    """
    Return lecture chunks with text and citation metadata.

    Uses FAISS candidates when available, then reranks them together with
    lexical matches. This keeps semantic search, but prevents short unrelated
    chunks from beating exact Russian lecture terms.
    """
    if _load_vector_stack():
        import numpy as np
        import requests

        try:
            q_emb = _embedder.encode([question])
        except requests.RequestException:
            return _lexical_search(question, top_k=top_k)

        q_emb = np.array(q_emb)
        vector_rows = _store.search(q_emb, top_k=max(top_k * 5, 20))
        return _hybrid_rerank(question, vector_rows, top_k=top_k)

    return _lexical_search(question, top_k=top_k)

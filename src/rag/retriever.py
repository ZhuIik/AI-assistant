import json
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from src.config import EMBED_MODEL

INDEX_DIR = Path("data/datasets/rag_index")

_embedder = None
_store = None
_lexical_meta: Optional[list[dict]] = None
_bm25: Optional[BM25Okapi] = None


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


def _row_text(row: dict) -> str:
    return row.get("search_text") or row.get("text") or ""


def _load_lexical_meta():
    global _lexical_meta, _bm25
    if _lexical_meta is not None:
        return

    meta_path = INDEX_DIR / "meta.json"
    if not meta_path.exists():
        _lexical_meta = []
        _bm25 = None
        return

    _lexical_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tokenized_corpus = [_tokenize(_row_text(row)) for row in _lexical_meta]
    _bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None


def _bm25_score_map(question: str) -> dict[str, float]:
    """BM25 score per row id for the whole corpus, or {} if there's nothing to score."""
    _load_lexical_meta()
    if not _lexical_meta or _bm25 is None:
        return {}

    query_terms = _tokenize(question)
    if not query_terms:
        return {}

    scores = _bm25.get_scores(query_terms)
    return {
        (row.get("id") or f"row-{index}"): float(score)
        for index, (row, score) in enumerate(zip(_lexical_meta, scores))
    }


def _lexical_search(question: str, top_k: int) -> list[dict]:
    _load_lexical_meta()
    if not _lexical_meta:
        return []

    query_terms = _tokenize(question)
    if not query_terms:
        return _lexical_meta[:top_k]

    scores = _bm25_score_map(question)
    scored = [
        (scores.get(row.get("id") or f"row-{index}", 0.0), row)
        for index, row in enumerate(_lexical_meta)
    ]
    scored = [(score, row) for score, row in scored if score > 0]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:top_k]]


def _hybrid_rerank(question: str, vector_rows: list[dict], top_k: int) -> list[dict]:
    lexical_rows = _lexical_search(question, top_k=max(top_k * 5, 20))
    bm25_scores = _bm25_score_map(question)

    merged: dict[str, tuple[dict, float]] = {}

    for rank, row in enumerate(vector_rows, start=1):
        row_id = row.get("id") or f"vector-{rank}"
        merged[row_id] = (row, 1.0 / rank)

    for rank, row in enumerate(lexical_rows, start=1):
        row_id = row.get("id") or f"lexical-{rank}"
        existing_row, existing_bonus = merged.get(row_id, (row, 0.0))
        merged[row_id] = (existing_row, existing_bonus + 2.0 / rank)

    scored = []
    for row_id, (row, rank_bonus) in merged.items():
        lexical = bm25_scores.get(row_id, 0.0)
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

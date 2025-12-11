# src/rag/retriever.py
from pathlib import Path
import numpy as np
from typing import Optional

from .embedder import Embedder
from .vector_store import VectorStore

INDEX_DIR = Path("data/datasets/rag_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embedder: Optional[Embedder] = None
_store: Optional[VectorStore] = None



def _lazy_init():
    global _embedder, _store
    if _embedder is None:
        _embedder = Embedder(EMBEDDING_MODEL)
    if _store is None:
        _store = VectorStore.load(INDEX_DIR)


def get_relevant_chunks(question: str, top_k: int = 5) -> list[dict]:
    """
    Возвращает top_k чанков вида:
    { "text": ..., "source": ..., "id": ... }
    """
    _lazy_init()

    q_emb = _embedder.encode([question])
    q_emb = np.array(q_emb)

    results = _store.search(q_emb, top_k=top_k)
    return results

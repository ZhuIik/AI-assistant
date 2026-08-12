import numpy as np
import requests

from src.config import EMBED_MODEL, LM_STUDIO_BASE_URL

_BATCH_SIZE = 64

# Some embedding models are trained with asymmetric query/document instruction
# prefixes (E5/BGE-style) and lose retrieval quality without them. Keyed by a
# lowercase substring of the LM Studio model id; (query_prefix, document_prefix).
# Deliberately excludes the current production model (nomic-embed-text) —
# nomic's card documents the same convention, but the prod index/reference
# metrics were built without it, so adding it here would silently change
# already-deployed retrieval behavior without a matching index rebuild.
_PREFIXES: dict[str, tuple[str, str]] = {
    "berta": ("search_query: ", "search_document: "),
    "qwen3-embedding": (
        "Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: ",
        "",
    ),
}


def _prefix_for(model_name: str, kind: str) -> str:
    lowered = model_name.lower()
    for key, (query_prefix, document_prefix) in _PREFIXES.items():
        if key in lowered:
            return query_prefix if kind == "query" else document_prefix
    return ""


class Embedder:
    """Embeds text via LM Studio's local OpenAI-compatible /v1/embeddings endpoint.

    Default model (nomic-embed-text) is multilingual, so it works for both
    Russian and English lecture content, unlike the English-centric MiniLM
    model used previously.
    """

    def __init__(self, model_name: str = EMBED_MODEL, base_url: str = LM_STUDIO_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def encode(self, texts: list[str], kind: str = "document") -> np.ndarray:
        prefix = _prefix_for(self.model_name, kind)
        if prefix:
            texts = [prefix + text for text in texts]

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start:start + _BATCH_SIZE]
            response = requests.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model_name, "input": batch},
                timeout=120,
            )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in data)
        return np.array(vectors, dtype="float32")

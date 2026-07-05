import numpy as np
import requests

from src.config import EMBED_MODEL, LM_STUDIO_BASE_URL

_BATCH_SIZE = 64


class Embedder:
    """Embeds text via LM Studio's local OpenAI-compatible /v1/embeddings endpoint.

    Default model (nomic-embed-text) is multilingual, so it works for both
    Russian and English lecture content, unlike the English-centric MiniLM
    model used previously.
    """

    def __init__(self, model_name: str = EMBED_MODEL, base_url: str = LM_STUDIO_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def encode(self, texts: list[str]) -> np.ndarray:
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

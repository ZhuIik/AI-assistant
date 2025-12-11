from pathlib import Path
import faiss
import numpy as np
import json

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.meta: list[dict] = []

    def add(self, embeddings: np.ndarray, metadatas: list[dict]):
        self.index.add(embeddings.astype("float32"))
        self.meta.extend(metadatas)

    def save(self, dir_path: str):
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        (p / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, dir_path: str):
        p = Path(dir_path)
        index = faiss.read_index(str(p / "index.faiss"))
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        store = cls(index.d)
        store.index = index
        store.meta = meta
        return store

    def search(self, query_emb: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_emb.astype("float32"), top_k)
        results = []
        for idx in I[0]:
            results.append(self.meta[idx])
        return results
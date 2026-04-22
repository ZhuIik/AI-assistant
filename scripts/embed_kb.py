# scripts/embed_kb.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from rag.loader import load_documents
from rag.chunker import split_into_chunks
from rag.embedder import Embedder
from rag.vector_store import VectorStore


def main():
    docs = load_documents("data/cleaned/")
    print(f"Loaded docs: {len(docs)}")

    chunks = []
    metadatas = []
    for doc in docs:
        for i, chunk in enumerate(split_into_chunks(doc["text"])):
            chunks.append(chunk)
            metadatas.append({
                "id": f'{doc["id"]}_{i}',
                "text": chunk,
                "source": doc["source"],
            })

    embedder = Embedder()
    embeddings = embedder.encode(chunks)
    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, metadatas)
    store.save("data/datasets/rag_index")

    print("Index built and saved.")

if __name__ == "__main__":
    main()

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Пути
KB_PATH = "../data/knowledge_base/kb.jsonl"
OUT_DIR = "../embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

# Загружаем модель эмбеддингов (лёгкая, но точная)
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
meta = []

print("[i] Загружаем базу знаний...")
with open(KB_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        # Можно использовать summary, если хочешь компактнее
        text = item.get("summary") or item.get("text")
        texts.append(text)
        meta.append(item)

print(f"[i] Всего фрагментов: {len(texts)}")

# Векторизация
print("[i] Создание эмбеддингов...")
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# Сохраняем FAISS индекс
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
faiss.write_index(index, os.path.join(OUT_DIR, "faiss_index.bin"))

# Сохраняем метаданные
np.save(os.path.join(OUT_DIR, "meta.npy"), np.array(meta, dtype=object))

print("✅ Векторизация завершена!")
print(f"Индекс сохранён в {OUT_DIR}/faiss_index.bin")

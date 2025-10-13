import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# Пути
INDEX_PATH = "../embeddings/faiss_index.bin"
META_PATH = "../embeddings/meta.npy"

# Загружаем индекс и метаданные
index = faiss.read_index(INDEX_PATH)
meta = np.load(META_PATH, allow_pickle=True)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

MODEL = "mistral:instruct"

def retrieve_context(question, k=5):
    """Находит k самых похожих фрагментов и возвращает текст + источник"""
    q_emb = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    contexts, sources = [], set()

    for idx in I[0]:
        item = meta[idx]
        text = item.get("summary") or item.get("text")
        lecture = item.get("lecture", "unknown").replace(".txt", "")
        contexts.append(text)
        sources.add(lecture)
    return "\n".join(contexts), sorted(sources)

def ask(question):
    """Формирует ответ от модели на основе найденных фрагментов"""
    context, sources = retrieve_context(question)
    prompt = f"""Ты — академический ассистент.
Используй только представленные материалы лекций, чтобы ответить на вопрос.
Отвечай по-русски, чётко, понятно и академично.

Контекст:
{context}

Вопрос: {question}

Ответь, опираясь исключительно на приведённый контекст."""
    
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"]
    return answer, sources

if __name__ == "__main__":
    print("🎓 Lecture RAG Assistant готов к работе!")
    while True:
        q = input("\nЗадай вопрос (Enter — выход): ").strip()
        if not q:
            break
        answer, sources = ask(q)
        print("\n🧠 Ответ:\n")
        print(answer)
        print("\n📚 Источники:", ", ".join(sources))

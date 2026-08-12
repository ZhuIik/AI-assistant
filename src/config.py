import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv(path=None, *args, **kwargs):
        if path is None:
            path = Path(".env")
        path = Path(path)
        if not path.exists():
            return False
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return True


# Корень проекта: Ai-assistant/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

# LM Studio (локальный OpenAI-совместимый сервер) — включается в Настройки → Local Server
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")

# Идентификатор модели, загруженной в LM Studio (gemma-4-E4B-it-GGUF)
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "google/gemma-4-e4b")

# Модель для офлайн-разметки аннотаций (scripts/prefill_annotations.py). Может быть
# легче/быстрее основной чат-модели — по умолчанию совпадает с LM_STUDIO_MODEL.
ANNOTATE_MODEL = os.getenv("ANNOTATE_MODEL", LM_STUDIO_MODEL)

# Модель-судья для RAGAS (scripts/eval_rag.py). По умолчанию совпадает с LM_STUDIO_MODEL
# (как было раньше) — если явно не задать другую, поведение не меняется. Задать отдельную
# модель нужно, чтобы проверить self-preference bias: судья и генератор — одна и та же
# модель гораздо охотнее оправдывает свой же стиль ответа (см. EVAL_LOG.md).
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LM_STUDIO_MODEL)

# Embedding-модель для RAG-индекса (src/rag/embedder.py), тоже через LM Studio.
# Multilingual, работает и с русским, и с английским текстом.
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")

# Директория с FAISS-индексом/метаданными (src/rag/retriever.py). Переопределяется
# при прогоне eval на тестовом индексе (например data/datasets/rag_index_by_sentence),
# чтобы не трогать продакшн-индекс.
RAG_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", str(PROJECT_ROOT / "data" / "datasets" / "rag_index")))

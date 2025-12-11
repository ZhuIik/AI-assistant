from src.rag.retriever import get_relevant_chunks
from .model import load_pipeline
import time
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты ассистент по материалам курса и лекций. "
    "Если в контексте есть нужная информация, опирайся в первую очередь на него. "
    "Если в контексте нет прямого ответа, можешь использовать свои знания и отвечать как обычно. "
    "Отвечай развёрнуто, простым языком, как преподаватель, который объясняет тему студенту. "
    "Обязательно приводите 1–2 коротких примера, которые помогают понять идею на практике. "
    "Не повторяй текст контекста и технические метки, дай только нормальный ответ на вопрос."
)


def build_rag_prompt(user_question: str, top_k: int = 5) -> str:
    chunks = get_relevant_chunks(user_question, top_k=top_k)

    # контекст можем использовать даже если он шумный —
    # но просим модель НЕ повторять его явно
    if chunks:
        context_text = "\n\n".join(ch.get("text", "") for ch in chunks)
    else:
        context_text = "(контекст не найден или недостаточен)"

    prompt = f"""{SYSTEM_PROMPT}

Контекст (это подсказка, не зачитывай его в ответе:

{context_text}

)

Вопрос:
{user_question}

Сначала кратко сформулируй основную идею в 2–3 предложениях.
Затем подробно объясни шаг за шагом.
В конце приведи 1–2 простых примера (из бизнеса, жизни или учебных кейсов), которые иллюстрируют идею.

Ответ:
"""
    return prompt


# Лениво грузим пайплайн — только при первом вызове
_pipe = None


def get_pipeline():
    global _pipe
    if _pipe is None:
        try:
            logger.info("Загружаю модельный пайплайн (может занять время)...")
            start = time.time()
            _pipe = load_pipeline()
            logger.info("Пайплайн загружен за %.1f сек", time.time() - start)
        except Exception as e:
            logger.exception("Ошибка при загрузке пайплайна")
            raise RuntimeError(f"Не удалось загрузить модельный пайплайн: {e}") from e
    return _pipe


def chat_once(prompt: str) -> str:
    """Единичный запрос к модели с RAG-контекстом."""
    full_prompt = build_rag_prompt(prompt, top_k=5)

    pipe = get_pipeline()
    raw = pipe(
        full_prompt,
        max_new_tokens=1200,
        do_sample=True,
        temperature=0.7,
    )[0]["generated_text"]

    # попробуем вытащить только то, что ИДЁТ ПОСЛЕ "Ответ:"
    marker = "Ответ:"
    if marker in raw:
        answer = raw.split(marker, 1)[1].strip()
    else:
        # fallback: если модель не повторила промпт, берём всё
        answer = raw.strip()

    return answer

import json
import re

import requests

from src.config import ANNOTATE_MODEL, LM_STUDIO_BASE_URL

SYSTEM_PROMPT = (
    "Ты помощник, который размечает фрагменты транскриптов лекций для поисковой базы. "
    "Отвечай только JSON, без пояснений и без markdown-разметки."
)

USER_PROMPT_TEMPLATE = """Ниже - фрагмент транскрипта лекции. Он получен автоматическим распознаванием речи \
и может содержать ошибки распознавания (спутанные по звучанию слова, лишние знаки препинания).

Текст фрагмента:
{text}

Выполни:
1. Если видишь явную ошибку распознавания (слово не имеет смысла в контексте и похоже по звучанию на другое) \
- точечно исправь только её. Не меняй смысл, не сокращай и не пересказывай текст своими словами.
2. Определи тему фрагмента (topic) - короткая фраза, 3-8 слов.
3. Напиши краткое summary (1-2 предложения) о том, что реально обсуждается в тексте.
4. Выдели до 5 ключевых слов/терминов из текста (keywords).
5. Если в тексте реально заданы вопросы - выпиши их дословно (questions), максимум 5. Если вопросов нет - пустой список.
6. Определи interaction_type: один из "lecture", "qa", "example", "exercise".
7. Определи cognitive_level по таксономии Блума: один из "remember", "understand", "apply", "analyze", "evaluate", "create".

Не выдумывай факты, термины и вопросы, которых нет в тексте.

Верни только JSON в точности такого вида, без пояснений и без markdown:
{{"cleaned_text": "...", "topic": "...", "summary": "...", "keywords": ["..."], "questions": ["..."], "interaction_type": "...", "cognitive_level": "..."}}
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_REQUIRED_FIELDS = {
    "cleaned_text", "topic", "summary", "keywords", "questions",
    "interaction_type", "cognitive_level",
}


class AnnotationError(RuntimeError):
    """Raised when the local LLM is unreachable or returns an unusable response."""


def annotate_fragment(text: str, timeout: float = 300.0) -> dict:
    """Ask the local LLM to lightly clean ASR text and derive topic/summary/keywords/questions.

    Raises AnnotationError on any failure so callers can fall back to the heuristic path.
    """
    try:
        response = requests.post(
            f"{LM_STUDIO_BASE_URL.rstrip('/')}/chat/completions",
            json={
                "model": ANNOTATE_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
                ],
                "temperature": 0.1,
                # Generous budget: some locally served models emit hidden "thinking"
                # tokens (reasoning_content) before the actual JSON answer, and a low
                # limit can cut the response off before any real content is produced.
                "max_tokens": 3000,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError) as exc:
        raise AnnotationError(f"LLM request failed: {exc}") from exc

    match = _JSON_BLOCK_RE.search(content)
    if not match:
        raise AnnotationError(f"LLM did not return JSON: {content[:200]!r}")

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"Could not parse LLM JSON: {exc}") from exc

    missing = _REQUIRED_FIELDS - data.keys()
    if missing:
        raise AnnotationError(f"LLM JSON missing fields: {missing}")

    return data

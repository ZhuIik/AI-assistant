import ast
import logging
import operator
import re
import time

from src.rag.retriever import get_relevant_chunks
from src.logging_config import configure_logging
from src.session_history import add_exchange, get_recent_history


configure_logging()
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Ты учебный ассистент по материалам курса и лекций.
Отвечай на русском языке как внимательный преподаватель: спокойно, понятно и по шагам.
Главная цель ответа: помочь студенту понять тему, а не просто выдать формальное определение.

Правила ответа:
- Сначала коротко объясни суть вопроса своими словами.
- Затем раскрой смысл: зачем это важно, как это работает и на что обратить внимание.
- Если вопрос неоднозначный или не хватает данных, задай 1-2 конкретных уточняющих вопроса.
- Опирайся на найденные фрагменты лекций как на источник фактов.
- Не выдумывай факты, формулы, даты, требования и термины, которых нет в контексте.
- Не переписывай транскрипт целиком.
- Не придумывай иллюстративные примеры, кейсы, компании или цифры, которых нет в найденном
  контексте. Если в контексте уже есть подходящий пример - можешь его пересказать своими словами.

Не добавляй раздел "Ссылки на лекции": он будет добавлен автоматически.
Не используй внутренние номера фрагментов вроде [1] в итоговом ответе.
Если найденный контекст не дает прямого ответа, честно скажи это в начале и дальше
дай осторожное пояснение только на основе найденных фрагментов.
"""

STOPWORDS = {
    "что", "такое", "как", "это", "про", "для", "или", "при", "где", "когда",
    "почему", "зачем", "какой", "какая", "какие", "какое", "расскажи",
    "объясни", "поясни", "лекции", "лекция",
}

FAST_CHAT_REPLIES = {
    "привет": "Привет! Задай вопрос по лекциям или по учебной теме, и я объясню как преподаватель.",
    "здравствуй": "Здравствуйте! Напишите вопрос, а я помогу разобраться по шагам.",
    "здравствуйте": "Здравствуйте! Напишите вопрос, а я помогу разобраться по шагам.",
    "спасибо": "Пожалуйста! Если хочешь, можем разобрать следующий вопрос.",
}

COURSE_DESCRIPTION = (
    "Этот предмет связан с практическим предпринимательством: как придумать и описать предпринимательскую идею, "
    "понять целевую аудиторию, продумать продажи, ресурсы, затраты, инвестиции и собрать это в понятный проект "
    "или бизнес-план. Если говорить проще, курс учит не только считать отдельные показатели, а связывать идею, "
    "рынок, деньги и организацию работы в одну логичную картину."
)

BOT_CAPABILITIES = (
    "Я помогаю разбираться с материалами курса: объясняю темы простыми словами, показываю логику расчетов, "
    "привожу учебные примеры и подсказываю, где в лекциях об этом говорилось. Если вопрос слишком общий, "
    "я могу уточнить, что именно нужно разобрать. Можно спрашивать, например: "
    "\"объясни точку безубыточности\", \"как описать одну продажу\", \"что важно в бизнес-плане\"."
)

_ALLOWED_MATH_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
)
_MATH_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _format_clock(total_seconds: float) -> str:
    seconds = max(0, int(float(total_seconds)))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_time_value(value) -> str:
    if value is None or value == "":
        return ""

    if isinstance(value, (int, float)):
        return _format_clock(value)

    text = str(value).strip()
    if not text:
        return ""

    if re.fullmatch(r"\d+(\.\d+)?", text):
        return _format_clock(float(text))

    match = re.match(r"^(\d{1,2}):(\d{1,2}):(\d{1,2})(?:[.,]\d+)?$", text)
    if match:
        hours, minutes, seconds = (int(part) for part in match.groups())
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return re.sub(r"(\d{1,2}:\d{2}:\d{2})[.,]\d+", r"\1", text)


def _format_timecode(chunk: dict) -> str:
    start = _format_time_value(chunk.get("start_sec") or chunk.get("start"))
    end = _format_time_value(chunk.get("end_sec") or chunk.get("end"))
    if start and end:
        return f"{start}-{end}"

    raw_timecode = chunk.get("timecode") or ""
    if "-" not in raw_timecode:
        return _format_time_value(raw_timecode)

    raw_start, raw_end = raw_timecode.split("-", 1)
    start = _format_time_value(raw_start)
    end = _format_time_value(raw_end)
    return f"{start}-{end}" if start and end else _format_time_value(raw_timecode)


def _format_source_name(value) -> str:
    text = str(value or "").strip()
    if not text:
        return "Источник"

    text = re.sub(r"\.(annotations|segments|draft)\.jsonl$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.(txt|jsonl|json)$", "", text, flags=re.IGNORECASE)
    match = re.fullmatch(r"lecture[_\s-]*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if match:
        return f"Лекция {match.group(1)}"

    return text.replace("_", " ").strip()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", str(text).lower()))


def _content_terms(text: str) -> set[str]:
    return {term for term in _tokenize(text) if len(term) > 2 and term not in STOPWORDS}


def _normalize_timecodes_in_text(text: str) -> str:
    return re.sub(r"(\d{1,2}:\d{2}:\d{2})[.,]\d+", r"\1", text)


def _snippet_around_terms(text: str, query_terms: set[str] | None = None, max_len: int = 520) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip(" .:-")
    text = _normalize_timecodes_in_text(text)
    if not text:
        return "фрагмент лекции по теме вопроса"

    if query_terms:
        lowered = text.lower()
        matches = [
            lowered.find(term)
            for term in sorted(query_terms, key=len, reverse=True)
            if lowered.find(term) >= 0
        ]
        if matches:
            pos = min(matches)
            start = max(0, pos - max_len // 3)
            end = min(len(text), pos + max_len * 2 // 3)
            if start > 0:
                next_space = text.find(" ", start)
                if 0 <= next_space < pos:
                    start = next_space + 1
            if end < len(text):
                prev_space = text.rfind(" ", pos, end)
                if prev_space > pos:
                    end = prev_space
            snippet = text[start:end].strip(" .,;:-")
            if start > 0:
                snippet = "... " + snippet
            if end < len(text):
                snippet = snippet + " ..."
            return snippet

    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0].strip() + " ..."


def _short_summary(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip(" .:-")
    text = _normalize_timecodes_in_text(text)
    if not text:
        return "фрагмент по теме вопроса."
    if len(text) <= 140:
        return text.rstrip(" .") + "."
    return text[:140].rsplit(" ", 1)[0].rstrip(" .") + "."


def _format_chunk(chunk: dict, index: int) -> str:
    lecture = _format_source_name(chunk.get("lecture_id") or chunk.get("source"))
    timecode = _format_timecode(chunk)
    topic = chunk.get("topic") or ""
    summary = chunk.get("summary") or ""
    questions = chunk.get("questions") or []
    text = chunk.get("text") or ""

    header = f"[{index}] {lecture}"
    if timecode:
        header += f" {timecode}"
    if topic:
        header += f" | {topic}"

    return "\n".join(
        part
        for part in [
            header,
            f"Summary: {summary}" if summary else "",
            f"Related questions: {'; '.join(questions)}" if questions else "",
            f"Transcript: {text}" if text else "",
        ]
        if part
    )


def _fallback_source_description(chunk: dict, query_terms: set[str] | None = None) -> str:
    text = chunk.get("text") or ""
    summary = chunk.get("summary") or ""
    topic = chunk.get("topic") or ""

    if summary and not re.search(
        r"связанн|учебн|материал|покажу|конечно|потому|возникает",
        summary.lower(),
    ):
        return _short_summary(summary)

    if topic and not re.search(r"обсуждение учебного материала", topic.lower()):
        return _short_summary(topic)

    return _short_summary(_snippet_around_terms(text or summary or topic, query_terms, max_len=220))


def _source_summary_prompt(chunks: list[dict], user_question: str) -> str:
    query_terms = _content_terms(user_question)
    blocks = []
    for index, chunk in enumerate(chunks[:5], start=1):
        lecture = _format_source_name(chunk.get("lecture_id") or chunk.get("source"))
        timecode = _format_timecode(chunk)
        text = _snippet_around_terms(chunk.get("text") or chunk.get("summary") or "", query_terms)
        blocks.append(f"{index}. {lecture} {timecode}\n{text}")

    return (
        "Сформулируй краткое саммари для каждой ссылки на лекцию.\n"
        "Пиши по-русски, по одной строке на каждый фрагмент.\n"
        "Не добавляй новых фактов. Не цитируй длинно. Каждая строка: 8-16 слов, "
        "что именно обсуждается во фрагменте.\n\n"
        f"Вопрос пользователя: {user_question}\n\n"
        "Фрагменты:\n"
        + "\n\n".join(blocks)
        + "\n\nСаммари:"
    )


def _parse_source_summaries(raw: str, expected: int) -> list[str]:
    if "Саммари:" in raw:
        raw = raw.split("Саммари:", 1)[1]
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[\).:-])\s*", "", line).strip()
        line = line.strip(" .")
        if line:
            lines.append(line + ".")
        if len(lines) >= expected:
            break
    return lines


def _generate_source_summaries(chunks: list[dict], user_question: str) -> list[str]:
    if not chunks:
        return []

    fallback = [
        _fallback_source_description(chunk, _content_terms(user_question))
        for chunk in chunks[:5]
    ]

    try:
        raw = get_pipeline()(  # type: ignore
            _source_summary_prompt(chunks, user_question),
            max_new_tokens=360,
            do_sample=False,
            temperature=0.1,
        )[0]["generated_text"]
    except Exception:
        return fallback

    summaries = _parse_source_summaries(raw, expected=len(fallback))
    if len(summaries) < len(fallback):
        summaries.extend(fallback[len(summaries):])
    return [_normalize_timecodes_in_text(item) for item in summaries[:len(fallback)]]


def _format_sources(chunks: list[dict], summaries: list[str] | None = None, user_question: str = "") -> str:
    lines = ["Ссылки на лекции"]
    query_terms = _content_terms(user_question)
    for index, chunk in enumerate(chunks[:5]):
        lecture = _format_source_name(chunk.get("lecture_id") or chunk.get("source"))
        timecode = _format_timecode(chunk)
        if summaries and index < len(summaries):
            description = _short_summary(summaries[index])
        else:
            description = _fallback_source_description(chunk, query_terms)
        lines.append(f"{lecture} {timecode} - {description}".strip())
    return "\n".join(lines)


def _strip_model_sources(answer: str) -> str:
    return re.split(r"\n\s*Ссылки на лекции\s*\n", answer, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def _format_history_for_prompt(history: list[dict]) -> str:
    if not history:
        return "(предыдущего контекста нет)"

    lines = []
    for item in history:
        role = "Пользователь" if item.get("role") == "user" else "Ассистент"
        content = re.sub(r"\s+", " ", item.get("content") or "").strip()
        if len(content) > 600:
            content = content[:600].rsplit(" ", 1)[0] + " ..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_rag_prompt(user_question: str, top_k: int = 5, history: list[dict] | None = None) -> str:
    chunks = get_relevant_chunks(user_question, top_k=top_k)

    if chunks:
        context_text = "\n\n".join(_format_chunk(ch, i) for i, ch in enumerate(chunks, start=1))
    else:
        context_text = "(контекст не найден или недостаточен)"

    return f"""{SYSTEM_PROMPT}

Недавний контекст диалога:
{_format_history_for_prompt(history or [])}

Контекст из аннотаций лекций:

{context_text}

Вопрос:
{user_question}

Сформируй ответ так:
Дай подробный основной ответ без заголовка: 2-4 содержательных абзаца, объясняющих суть, логику и важные детали, опираясь только на приведенный контекст.
Не добавляй раздел "Пример" и не выдумывай иллюстративные кейсы, компании или цифры, которых нет в контексте.
Не добавляй ссылки и таймкоды: они будут добавлены автоматически.

Ответ:
"""


_pipe = None


def get_pipeline():
    global _pipe
    if _pipe is None:
        try:
            from .model import load_pipeline

            _pipe = load_pipeline()
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить модельный пайплайн: {e}") from e
    return _pipe


def _extractive_answer(user_question: str, chunks: list[dict], reason: str) -> str:
    if not chunks:
        return (
            "Я не нашел подходящие фрагменты в индексе лекций, поэтому не могу надежно ответить по материалам курса.\n\n"
            "Пример\n"
            "Надежный пример не привожу: нет найденного фрагмента, на который можно опереться.\n\n"
            f"Модель генерации сейчас недоступна: {reason}\n\n"
            "Ссылки на лекции\n"
            "Релевантные таймкоды не найдены."
        )

    main = chunks[0]
    summary = main.get("summary") or main.get("text") or "Подходящий фрагмент найден, но в нем нет краткого описания."
    text = _snippet_around_terms(main.get("text") or summary, _content_terms(user_question), max_len=520)

    answer = "\n\n".join(
        [
            summary,
            "Модель генерации сейчас недоступна, поэтому я даю извлеченный ответ по ближайшим найденным фрагментам.",
            "Пример\n" + text,
            _format_sources(chunks, user_question=user_question),
        ]
    )
    return _normalize_timecodes_in_text(answer)


def _is_simple_math_question(text: str) -> bool:
    expression = text.strip().rstrip("?").replace(",", ".")
    if not re.fullmatch(r"[0-9\s+\-*/().]+", expression):
        return False
    return bool(re.search(r"\d\s*[+\-*/]\s*\d", expression))


def _eval_math_node(node):
    if not isinstance(node, _ALLOWED_MATH_NODES):
        raise ValueError("unsupported expression")
    if isinstance(node, ast.Expression):
        return _eval_math_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("unsupported constant")
    if isinstance(node, ast.UnaryOp):
        value = _eval_math_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        left = _eval_math_node(node.left)
        right = _eval_math_node(node.right)
        op_type = type(node.op)
        if op_type not in _MATH_OPERATORS:
            raise ValueError("unsupported operator")
        return _MATH_OPERATORS[op_type](left, right)
    raise ValueError("unsupported expression")


def _answer_simple_math(text: str) -> str:
    expression = text.strip().rstrip("?").replace(",", ".")
    result = _eval_math_node(ast.parse(expression, mode="eval"))
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"


def _answer_fast_chat(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", text.strip().lower()).strip("!.? ")
    direct_reply = FAST_CHAT_REPLIES.get(normalized)
    if direct_reply:
        return direct_reply

    if _looks_like_capabilities_question(normalized):
        return BOT_CAPABILITIES

    if _looks_like_course_question(normalized):
        return COURSE_DESCRIPTION

    return None


def _looks_like_capabilities_question(text: str) -> bool:
    return bool(
        re.search(r"\b(что|чем|как)\b", text)
        and re.search(r"\b(умеешь|умеет\w*|можешь|может|помогаешь|помочь|делаешь)\b", text)
    ) or text in {
        "что ты умеешь",
        "что умеешь",
        "что умеет бот",
        "что ты можешь",
        "чем ты можешь помочь",
        "помощь",
        "help",
    }


def _looks_like_course_question(text: str) -> bool:
    return bool(
        re.search(r"\b(про что|о чем|что за|какой|какая)\b", text)
        and re.search(r"\b(предмет|курс|дисциплина|лекции)\b", text)
    ) or text in {
        "про что предмет",
        "о чем предмет",
        "что это за предмет",
        "про что этот предмет",
        "о чем этот курс",
        "что изучаем",
    }


def _fast_answer(prompt: str) -> tuple[str, str] | None:
    chat_reply = _answer_fast_chat(prompt)
    if chat_reply:
        return "fast_chat", chat_reply

    if _is_simple_math_question(prompt):
        try:
            return "fast_math", _answer_simple_math(prompt)
        except ZeroDivisionError:
            return "fast_math", "На ноль делить нельзя: такое выражение не имеет обычного числового результата."
        except Exception:
            return None

    return None


def chat_once(prompt: str, session_id: str | None = None, history: list[dict] | None = None) -> str:
    """Single model call with optional session history and lecture RAG context."""
    start_time = time.perf_counter()
    prompt = str(prompt or "").strip()
    session_key = str(session_id).strip() if session_id is not None else None
    history = history if history is not None else get_recent_history(session_key)

    fast = _fast_answer(prompt)
    if fast:
        mode, answer = fast
        if session_key:
            add_exchange(session_key, prompt, answer)
        logger.info(
            "chat_answer mode=%s session_id=%s rag_used=false elapsed_ms=%d",
            mode,
            session_key or "-",
            int((time.perf_counter() - start_time) * 1000),
        )
        return answer

    chunks = get_relevant_chunks(prompt, top_k=5)
    full_prompt = build_rag_prompt(prompt, top_k=5, history=history)

    try:
        raw = get_pipeline()(  # type: ignore
            full_prompt,
            max_new_tokens=1600,
            do_sample=True,
            temperature=0.7,
        )[0]["generated_text"]
    except Exception as exc:
        answer = _extractive_answer(prompt, chunks, str(exc))
        if session_key:
            add_exchange(session_key, prompt, answer)
        logger.exception(
            "chat_answer mode=rag_fallback session_id=%s chunks=%d elapsed_ms=%d",
            session_key or "-",
            len(chunks),
            int((time.perf_counter() - start_time) * 1000),
        )
        return answer

    marker = "Ответ:"
    answer = raw.split(marker, 1)[1].strip() if marker in raw else raw.strip()
    answer = _strip_model_sources(_normalize_timecodes_in_text(answer))
    if chunks:
        summaries = _generate_source_summaries(chunks, prompt)
        answer = f"{answer}\n\n{_format_sources(chunks, summaries=summaries, user_question=prompt)}"

    answer = answer.strip()
    if session_key:
        add_exchange(session_key, prompt, answer)
    logger.info(
        "chat_answer mode=rag session_id=%s chunks=%d elapsed_ms=%d",
        session_key or "-",
        len(chunks),
        int((time.perf_counter() - start_time) * 1000),
    )
    return answer

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS_DIR = ROOT / "data" / "annotations"

RUSSIAN_STOPWORDS = {
    "а", "без", "более", "бы", "был", "была", "были", "было", "быть", "в", "вам", "вас",
    "весь", "во", "вот", "все", "всё", "вы", "где", "да", "даже", "для", "до", "его", "ее",
    "её", "если", "есть", "еще", "ещё", "же", "за", "здесь", "и", "из", "или", "им", "их",
    "к", "как", "какая", "какие", "какой", "когда", "кто", "ли", "либо", "мне", "может",
    "можно", "мы", "на", "над", "надо", "нам", "нас", "не", "него", "нее", "неё", "нет",
    "ни", "но", "ну", "о", "об", "однако", "он", "она", "они", "оно", "опять", "от", "по",
    "под", "при", "про", "просто", "раз", "с", "сам", "сама", "сами", "само", "себя", "сейчас",
    "сказал", "сказать", "со", "совсем", "так", "также", "такой", "там", "тебя", "тем", "то",
    "тогда", "того", "тоже", "только", "том", "тот", "тут", "ты", "у", "уже", "хорошо", "чего",
    "чей", "чем", "что", "чтобы", "эта", "эти", "это", "я",
    "типа", "вроде", "короче", "вообще", "прям", "самый", "самое", "самая",
    "этот", "этакий", "сюда", "туда", "тут", "здесь", "этого", "этом", "этой",
    "какой-то", "что-то", "кто-то", "грубо", "говоря", "ладно", "угу", "ага",
    "окей", "давайте", "пожалуйста",
}

TOPIC_PATTERNS = [
    {
        "label": "Системная постановка задачи управления",
        "stems": ("задач", "управлен", "систем", "цель", "критер", "эффектив"),
        "keywords": ["задача управления", "система", "цель", "критерии"],
        "question": "Как в этом фрагменте формулируется задача управления?",
        "summary": "В этом фрагменте обсуждается системная постановка задачи управления, ее цель и критерии.",
    },
    {
        "label": "Основной бизнес-процесс",
        "stems": ("бизнес", "процесс", "основн", "операц", "bpmn"),
        "keywords": ["бизнес-процесс", "операции", "BPMN"],
        "question": "Какой основной бизнес-процесс рассматривается в этом фрагменте?",
        "summary": "В этом фрагменте разбирается основной бизнес-процесс и его шаги.",
    },
    {
        "label": "Проект и требования к работе",
        "stems": ("проект", "работ", "требован", "раздел", "выпускн", "квалификац"),
        "keywords": ["проект", "требования", "структура работы"],
        "question": "Какие требования к проекту или работе обсуждаются в этом фрагменте?",
        "summary": "В этом фрагменте обсуждаются требования к проекту, структуре работы или содержанию раздела.",
    },
    {
        "label": "Разбор примера или кейса",
        "stems": ("пример", "например", "образц", "случа", "кейс"),
        "keywords": ["пример", "кейс", "разбор"],
        "question": "Какой пример или кейс рассматривается в этом фрагменте?",
        "summary": "В этом фрагменте преподаватель или студенты разбирают пример и обсуждают его детали.",
    },
    {
        "label": "Организационные вопросы занятия",
        "stems": ("следующ", "пара", "оценк", "готов", "выступ", "групп", "команд", "занят"),
        "keywords": ["занятие", "организация", "выступление", "оценка"],
        "question": "Какие организационные вопросы занятия обсуждаются в этом фрагменте?",
        "summary": "В этом фрагменте обсуждаются организационные вопросы занятия, выступлений или оценивания.",
    },
    {
        "label": "Продажи и бизнес-план",
        "stems": ("продаж", "затрат", "доход", "бизнес", "клиент", "логист", "склад", "поставщик", "товар"),
        "keywords": ["продажи", "бизнес-план", "затраты", "доход"],
        "question": "Как в этом фрагменте обсуждаются продажи, затраты или бизнес-план?",
        "summary": "В этом фрагменте обсуждаются продажи, бизнес-план и связанные с ними затраты или доходы.",
    },
    {
        "label": "Параметры агрегатов и продукции",
        "stems": ("параметр", "агрегат", "продукц", "маршрут", "рольганг", "печ", "температур", "заготовк"),
        "keywords": ["параметры", "агрегаты", "продукция", "температура"],
        "question": "Какие параметры агрегатов или продукции описываются в этом фрагменте?",
        "summary": "В этом фрагменте рассматриваются параметры агрегатов, продукции и их связь с технологическим процессом.",
    },
    {
        "label": "Пример из энергетики",
        "stems": ("электростанц", "энергет", "газ", "мощност", "диспетчер", "котл"),
        "keywords": ["энергетика", "электростанция", "мощность"],
        "question": "Какой пример из энергетики приводится в этом фрагменте?",
        "summary": "В этом фрагменте приводится пример из энергетики для пояснения обсуждаемой идеи.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefill annotation fields from draft annotation blocks."
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_ANNOTATIONS_DIR,
        help="Directory with *.draft.jsonl files.",
    )
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=5,
        help="Maximum number of keywords to keep.",
    )
    parser.add_argument(
        "--lecture-id",
        action="append",
        dest="lecture_ids",
        help="Process only the specified lecture ID. Can be repeated.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip annotation files that already exist.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def lecture_id_from_draft_path(path: Path) -> str:
    return path.name.removesuffix(".draft.jsonl")


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[А-Яа-яA-Za-z0-9\-]+", text.lower())


def stem_like(token: str) -> str:
    token = token.lower()
    suffixes = (
        "иями", "ями", "ами", "ого", "ему", "ому", "ыми", "ими", "иях", "ах", "ях",
        "ия", "ья", "ие", "ые", "ой", "ий", "ый", "ое", "ее", "ая", "яя", "ам", "ям",
        "ом", "ем", "ах", "ях", "ов", "ев", "ей", "иям", "ием", "ию", "ью", "ию",
        "а", "я", "ы", "и", "о", "е", "у", "ю",
    )
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def detect_topic_config(text: str) -> Optional[dict[str, Any]]:
    lowered = text.lower()
    stemmed_tokens = {stem_like(token) for token in tokenize(text)}
    best_match = None
    best_score = 0

    for config in TOPIC_PATTERNS:
        score = sum(
            1 for stem in config["stems"]
            if stem in lowered or any(token.startswith(stem) for token in stemmed_tokens)
        )
        if score >= 2 and score > best_score:
            best_match = config
            best_score = score

    return best_match


def extract_keywords(text: str, max_keywords: int, topic_config: Optional[dict[str, Any]]) -> list[str]:
    if topic_config:
        return topic_config["keywords"][:max_keywords]

    tokens = tokenize(text)
    filtered = []
    for token in tokens:
        if len(token) < 5:
            continue
        if token in RUSSIAN_STOPWORDS:
            continue
        if token.isdigit():
            continue
        if not re.fullmatch(r"[а-яa-z\-]+", token):
            continue
        filtered.append(token)

    counts = Counter(filtered)
    keywords = [word for word, count in counts.most_common(max_keywords) if count >= 1]
    return keywords


def infer_topic(text: str, keywords: list[str], topic_config: Optional[dict[str, Any]]) -> str:
    if topic_config:
        return topic_config["label"]
    if keywords:
        return "Обсуждение учебного материала"
    return "Фрагмент лекции"


def looks_like_real_question(text: str) -> bool:
    tokens = tokenize(text)
    meaningful = [t for t in tokens if t not in RUSSIAN_STOPWORDS and len(t) >= 4]
    if not (6 <= len(text) <= 200):
        return False
    if len(tokens) <= 1:
        return False
    return len(meaningful) >= 1


def extract_questions(text: str, topic: str, topic_config: Optional[dict[str, Any]]) -> list[str]:
    raw_parts = re.split(r"(?<=[\?\!\.])\s+", text)
    found = []
    for part in raw_parts:
        cleaned = normalize_text(part)
        if "?" in cleaned and looks_like_real_question(cleaned):
            found.append(cleaned)

    # Keep only the real questions that were actually spoken in the fragment.
    unique_questions = []
    seen = set()
    for question in found:
        if question not in seen:
            seen.add(question)
            unique_questions.append(question)
    return unique_questions[:5]


def infer_interaction_type(text: str) -> str:
    lowered = text.lower()
    if "например" in lowered or "пример" in lowered:
        return "example"
    if any(word in lowered for word in ("сделайте", "посчитайте", "откройте", "задание", "упражнен")):
        return "exercise"
    if "?" in text:
        return "qa"
    return "lecture"


def infer_cognitive_level(text: str, interaction_type: str) -> str:
    lowered = text.lower()

    if any(word in lowered for word in ("разработ", "сформулиру", "построй", "созда")):
        return "create"
    if any(word in lowered for word in ("оцен", "выберите", "сравните", "лучше", "хуже")):
        return "evaluate"
    if any(word in lowered for word in ("анализ", "разбер", "структур", "связ", "причин")):
        return "analyze"
    if interaction_type == "exercise" or any(word in lowered for word in ("примен", "рассчита", "использу")):
        return "apply"
    if any(word in lowered for word in ("почему", "объясн", "суть", "значит", "понят")):
        return "understand"
    return "remember" if interaction_type == "qa" else "understand"


def infer_difficulty(text: str, keywords: list[str], cognitive_level: str) -> float:
    token_count = len(tokenize(text))
    keyword_factor = min(len(keywords) / 5.0, 1.0)
    length_factor = min(token_count / 120.0, 1.0)
    cognitive_factor = {
        "remember": 0.20,
        "understand": 0.35,
        "apply": 0.55,
        "analyze": 0.70,
        "evaluate": 0.82,
        "create": 0.90,
    }.get(cognitive_level, 0.40)

    score = 0.45 * cognitive_factor + 0.30 * keyword_factor + 0.25 * length_factor
    return round(min(max(score, 0.05), 0.95), 2)


def infer_actor_label(item: dict[str, Any]) -> str:
    speaker = item.get("speaker")
    speakers_raw = item.get("speakers_raw") or []
    if speaker == "professor":
        return "Преподаватель"
    if speaker == "student":
        return "Студент"
    if len(speakers_raw) > 1:
        return "Участники обсуждения"
    return "В этом фрагменте"


def infer_summary(
    item: dict[str, Any],
    topic: str,
    interaction_type: str,
    keywords: list[str],
    topic_config: Optional[dict[str, Any]],
) -> str:
    if topic_config:
        summary = topic_config["summary"]
        if interaction_type == "qa":
            return f"{summary} Формат фрагмента ближе к вопросам и ответам."
        return summary

    actor = infer_actor_label(item)
    if topic == "Обсуждение учебного материала" and keywords:
        return f"{actor} обсуждают учебный материал, связанный с: {', '.join(keywords[:3])}."
    if topic == "Фрагмент лекции":
        return f"{actor} обсуждают короткий фрагмент лекции без явно выраженной отдельной темы."
    return f"{actor} обсуждают тему «{topic}»."


def prefill_item(item: dict[str, Any], max_keywords: int) -> dict[str, Any]:
    text = normalize_text(item.get("text", ""))
    topic_config = detect_topic_config(text)
    keywords = extract_keywords(text, max_keywords=max_keywords, topic_config=topic_config)
    topic = infer_topic(text, keywords, topic_config)
    interaction_type = infer_interaction_type(text)
    cognitive_level = infer_cognitive_level(text, interaction_type)
    difficulty = infer_difficulty(text, keywords, cognitive_level)
    questions = extract_questions(text, topic, topic_config)

    enriched = dict(item)
    enriched["topic"] = topic
    enriched["keywords"] = keywords
    enriched["questions"] = questions
    enriched["summary"] = infer_summary(
        item=item,
        topic=topic,
        interaction_type=interaction_type,
        keywords=keywords,
        topic_config=topic_config,
    )
    enriched["cognitive_level"] = cognitive_level
    enriched["interaction_type"] = interaction_type
    enriched["difficulty"] = difficulty
    enriched["embedding"] = item.get("embedding", [])
    return enriched


def main() -> None:
    args = parse_args()
    files = sorted(args.annotations_dir.glob("*.draft.jsonl"))
    if args.lecture_ids:
        allowed = set(args.lecture_ids)
        files = [
            path for path in files
            if lecture_id_from_draft_path(path) in allowed
        ]
    if not files:
        raise SystemExit(f"No draft annotation files found in {args.annotations_dir}")

    total = 0
    for path in files:
        out_path = args.annotations_dir / path.name.replace(".draft.jsonl", ".annotations.jsonl")
        if args.skip_existing and out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue

        items = load_jsonl(path)
        enriched = [prefill_item(item, args.max_keywords) for item in items]
        write_jsonl(out_path, enriched)
        total += len(enriched)
        print(f"[done] {out_path} ({len(enriched)} annotations)")

    print(f"Prefilled annotations built: {total}")


if __name__ == "__main__":
    main()

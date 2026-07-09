"""
Run the fixed question set (data/eval/questions.jsonl) through the current
RAG pipeline and score it with RAGAS. Append one line per run to
data/eval/results.jsonl so different configs (embedder, top_k, model,
chunking, prompt, ...) can be compared over time.

Each line is {"timestamp": ..., "config": {...}, "metrics": {...}}. "config"
always includes embed_model/chat_model/top_k, plus anything passed via
--config as free-form key=value pairs - no fixed schema, so a new axis
(e.g. --config chunking=large) doesn't require migrating old rows.

Usage:
    python scripts/eval_rag.py --tag "nomic-embed-v1.5-topk5"
    python scripts/eval_rag.py --tag "bigger-chunks" --config chunking=large enrichment=keywords

The judge and embeddings used by RAGAS itself point at the same LM Studio
server as the app (LM_STUDIO_BASE_URL / EMBED_MODEL in src/config.py), so no
external API key is needed.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EMBED_MODEL, LM_STUDIO_BASE_URL, LM_STUDIO_MODEL
from src.llm.chat import build_rag_prompt
from src.llm.model import load_pipeline
from src.rag.retriever import get_relevant_chunks

QUESTIONS_PATH = ROOT / "data" / "eval" / "questions.jsonl"
RESULTS_PATH = ROOT / "data" / "eval" / "results.jsonl"

# Matches LM Studio's "max concurrent predictions" setting.
GENERATION_WORKERS = 4


def load_questions(lecture_ids: list[str] | None = None) -> list[dict]:
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if lecture_ids:
        allowed = set(lecture_ids)
        rows = [r for r in rows if r.get("lecture_id") in allowed]
    return rows


def _answer_one(item: dict, top_k: int, pipeline) -> dict:
    question = item["question"]
    chunks = get_relevant_chunks(question, top_k=top_k)

    prompt = build_rag_prompt(question, top_k=top_k)
    raw = pipeline(prompt, max_new_tokens=800, do_sample=False, temperature=0.0)[0]["generated_text"]
    answer = raw.split("Ответ:", 1)[1].strip() if "Ответ:" in raw else raw.strip()

    return {
        "question": question,
        "contexts": [c.get("text") or c.get("summary") or "" for c in chunks],
        "answer": answer,
        "ground_truth": item["reference_answer"],
    }


def run_pipeline(questions: list[dict], top_k: int) -> list[dict]:
    pipeline = load_pipeline()
    # Warm up the retriever's lazily-initialized globals (embedder/index/BM25)
    # once before fanning out, so concurrent threads don't race to load them.
    get_relevant_chunks(questions[0]["question"], top_k=top_k)

    with ThreadPoolExecutor(max_workers=GENERATION_WORKERS) as executor:
        rows = list(executor.map(lambda item: _answer_one(item, top_k, pipeline), questions))
    return rows


def score_with_ragas(rows: list[dict]) -> dict:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    judge = ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio",
        model=LM_STUDIO_MODEL,
        temperature=0.0,
    )
    embeddings = OpenAIEmbeddings(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio",
        model=EMBED_MODEL,
        check_embedding_ctx_length=False,
    )

    dataset = Dataset.from_list(
        [
            {
                "question": r["question"],
                "contexts": r["contexts"],
                "answer": r["answer"],
                "ground_truth": r["ground_truth"],
            }
            for r in rows
        ]
    )

    # LM Studio's "max concurrent predictions" is set to 4 - match it here so
    # ragas' request concurrency doesn't exceed what the server actually
    # parallelizes (ragas' default of 16 queues past its own timeout).
    run_config = RunConfig(timeout=600, max_workers=4)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
        run_config=run_config,
    )
    return result.to_pandas().mean(numeric_only=True).to_dict()


def append_results(tag: str, config: dict, ragas_scores: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": tag,
        "config": config,
        "metrics": {
            "faithfulness": round(ragas_scores.get("faithfulness", float("nan")), 4),
            "answer_relevancy": round(ragas_scores.get("answer_relevancy", float("nan")), 4),
            "context_precision": round(ragas_scores.get("context_precision", float("nan")), 4),
            "context_recall": round(ragas_scores.get("context_recall", float("nan")), 4),
        },
    }

    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n=== Eval result ===")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"\nAppended to {RESULTS_PATH}")


def parse_config_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        overrides[key] = value
    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Label for this run, e.g. 'baseline'")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--config",
        nargs="*",
        default=[],
        help="Extra free-form config as key=value pairs, e.g. --config chunking=large enrichment=keywords",
    )
    parser.add_argument(
        "--lecture-id",
        action="append",
        dest="lecture_ids",
        help="Only run questions tagged with this lecture_id. Can be repeated. "
        "Use when evaluating against a partial index (e.g. a single-lecture pilot index) "
        "so questions about lectures missing from that index don't score falsely low.",
    )
    args = parser.parse_args()

    questions = load_questions(lecture_ids=args.lecture_ids)
    if not questions:
        raise SystemExit(f"No eval questions matched lecture_ids={args.lecture_ids}")
    print(f"Loaded {len(questions)} eval questions" + (f" for {args.lecture_ids}" if args.lecture_ids else ""))

    rows = run_pipeline(questions, top_k=args.top_k)

    ragas_scores = score_with_ragas(rows)

    config = {
        "embed_model": EMBED_MODEL,
        "chat_model": LM_STUDIO_MODEL,
        "top_k": args.top_k,
        "lecture_ids": args.lecture_ids or "all",
        **parse_config_overrides(args.config),
    }
    append_results(args.tag, config, ragas_scores)


if __name__ == "__main__":
    main()

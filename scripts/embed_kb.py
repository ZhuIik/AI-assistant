"""Build the lecture RAG index from reviewed annotation files.

The bot needs lecture/timecode references, so the index is built from
``data/annotations/*.annotations.jsonl`` instead of cleaned plain text.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ANNOTATIONS_DIR = ROOT / "data" / "annotations"
INDEX_DIR = ROOT / "data" / "datasets" / "rag_index"


def _iter_annotations():
    for path in sorted(ANNOTATIONS_DIR.glob("*.annotations.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Bad JSON in {path}:{line_no}: {exc}") from exc

                text = (row.get("text") or "").strip()
                summary = (row.get("summary") or "").strip()
                topic = (row.get("topic") or "").strip()
                questions = row.get("questions") or []
                keywords = row.get("keywords") or []

                searchable = "\n".join(
                    part
                    for part in [
                        f"Topic: {topic}" if topic else "",
                        f"Summary: {summary}" if summary else "",
                        f"Keywords: {', '.join(keywords)}" if keywords else "",
                        f"Questions: {' '.join(questions)}" if questions else "",
                        f"Transcript: {text}" if text else "",
                    ]
                    if part
                )

                if not searchable:
                    continue

                lecture_id = row.get("lecture_id") or path.stem.replace(".annotations", "")
                start = row.get("start") or ""
                end = row.get("end") or ""
                annotation_id = row.get("id") or line_no

                yield searchable, {
                    "id": f"{lecture_id}:{annotation_id}",
                    "lecture_id": lecture_id,
                    "start": start,
                    "end": end,
                    "timecode": f"{start}-{end}" if start and end else "",
                    "topic": topic,
                    "summary": summary,
                    "questions": questions,
                    "keywords": keywords,
                    "text": text,
                    "source": path.name,
                    "segment_ids": row.get("segment_ids") or [],
                    "search_text": searchable,
                }


def main():
    pairs = list(_iter_annotations())
    if not pairs:
        raise RuntimeError(f"No annotations found in {ANNOTATIONS_DIR}")

    chunks = [text for text, _ in pairs]
    metadatas = [meta for _, meta in pairs]
    print(f"Loaded annotated chunks: {len(chunks)}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "meta.json").write_text(
        json.dumps(metadatas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "source": "annotations",
        "chunk_count": len(metadatas),
        "vector_built": False,
    }

    def _fall_back_to_lexical(reason: str) -> None:
        (INDEX_DIR / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"{reason}; saved lexical index metadata only.")
        print(f"Metadata saved to {INDEX_DIR / 'meta.json'}")

    try:
        from src.rag.embedder import Embedder
        from src.rag.vector_store import VectorStore
        import requests
    except ModuleNotFoundError as exc:
        _fall_back_to_lexical(f"Vector dependencies are not installed ({exc.name})")
        return

    embedder = Embedder()
    try:
        embeddings = embedder.encode(chunks)
    except requests.RequestException as exc:
        _fall_back_to_lexical(f"Embedding model unreachable via LM Studio ({exc})")
        return

    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, metadatas)
    store.save(str(INDEX_DIR))
    manifest["vector_built"] = True
    (INDEX_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Index built and saved to {INDEX_DIR}")

if __name__ == "__main__":
    main()

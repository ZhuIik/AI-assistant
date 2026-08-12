"""
Alternative draft-builder: chunk by semantic breaks between sentences,
instead of the fixed min/max character target in
build_annotation_drafts_by_sentence.py.

Why: sentence-based chunking never cuts mid-sentence, but it still cuts
purely on accumulated length - a chunk can span two unrelated topics if the
lecturer switches subject mid-way through filling the 800-1800 char target.
This shows up as a ceiling on context_precision in eval (see EVAL_LOG.md):
retrieval pulls in a chunk because half of it matches the question, but the
other half is noise. Semantic chunking instead embeds each sentence and cuts
where the topic actually shifts (a drop in cosine similarity between
consecutive sentences), so a chunk stays about one thing.

Method: embed every sentence (same embedder as the RAG index - EMBED_MODEL,
so pilot results are representative of production retrieval), compute
cosine similarity between each consecutive pair, and treat a similarity drop
above the --breakpoint-percentile of the lecture's own distribution as a
topic boundary (relative/adaptive per lecture, not a fixed magic number -
same method used by LlamaIndex's SemanticSplitterNodeParser). A breakpoint
only actually cuts the chunk once --min-chars is satisfied; if no semantic
breakpoint shows up before --max-chars, it force-cuts there anyway (same
safety net as the sentence-based chunker, prevents runaway growth on a
topically uniform stretch).

Timecodes come from scripts/chunking/timed_sentences.py, shared with every
other chunking strategy - only the grouping logic below is specific to this
one.

Output format is identical to build_annotation_drafts.py's *.draft.jsonl, so
scripts/prefill_annotations.py works unchanged - just point --drafts-dir at
this script's output instead.

Usage:
    python scripts/build_annotation_drafts_semantic.py --lecture-id Lecture_2
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_annotation_drafts import (
    DEFAULT_SEGMENTS_DIR,
    DEFAULT_SPEAKER_MAP,
    empty_block,
    finalize_block,
    lecture_id_from_segments_path,
    load_segments,
    load_speaker_map,
    merge_blocks,
    register_speaker,
    write_blocks,
)
from scripts.chunking.timed_sentences import format_timecode, split_all_into_timed_sentences
from src.rag.embedder import Embedder

DEFAULT_DRAFTS_DIR = ROOT / "data" / "drafts_semantic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build draft annotation blocks by cutting at semantic breaks between sentences."
    )
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--drafts-dir", type=Path, default=DEFAULT_DRAFTS_DIR)
    parser.add_argument("--speaker-map", type=Path, default=DEFAULT_SPEAKER_MAP)
    parser.add_argument(
        "--min-chars",
        type=int,
        default=800,
        help="Don't act on a semantic breakpoint before the chunk reaches at least this many characters.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1800,
        help="Force-close a chunk once adding the next sentence would exceed this many characters, "
        "even without a semantic breakpoint.",
    )
    parser.add_argument(
        "--breakpoint-percentile",
        type=float,
        default=95.0,
        help="A consecutive-sentence similarity drop above this percentile of the lecture's own "
        "distribution counts as a topic boundary. Computed per lecture (adaptive), not a fixed "
        "similarity value, since raw similarity scale varies by embedder and content.",
    )
    parser.add_argument("--lecture-id", action="append", dest="lecture_ids")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _cosine_similarities(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity between each consecutive pair of rows. Length = len(embeddings) - 1."""
    norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.sum(norms[:-1] * norms[1:], axis=1)


def find_breakpoints(sentence_texts: list[str], breakpoint_percentile: float) -> set[int]:
    """Indices i such that there's a topic break between sentence i and sentence i+1."""
    if len(sentence_texts) < 2:
        return set()

    embedder = Embedder()
    embeddings = embedder.encode(sentence_texts, kind="document")
    similarities = _cosine_similarities(embeddings)
    distances = 1.0 - similarities

    threshold = np.percentile(distances, breakpoint_percentile)
    return {i for i, distance in enumerate(distances) if distance >= threshold}


def build_semantic_blocks(
    segments: list[dict[str, Any]],
    speaker_map: dict[str, dict[str, str]],
    min_chars: int,
    max_chars: int,
    breakpoint_percentile: float,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    lecture_id = segments[0]["lecture_id"]
    timed_sentences = split_all_into_timed_sentences(segments)
    breakpoints = find_breakpoints([s["text"] for s in timed_sentences], breakpoint_percentile)

    raw_blocks: list[dict[str, Any]] = []
    current = empty_block(lecture_id)
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current["segment_ids"]:
            raw_blocks.append(current)
        current = empty_block(lecture_id)
        current_len = 0

    for idx, sentence in enumerate(timed_sentences):
        sentence_text = sentence["text"]
        addition_len = len(sentence_text) + (1 if current_len else 0)

        should_force_cut = current["segment_ids"] and current_len + addition_len > max_chars
        should_semantic_cut = (
            current["segment_ids"] and current_len >= min_chars and (idx - 1) in breakpoints
        )
        if should_force_cut or should_semantic_cut:
            flush()
            addition_len = len(sentence_text)

        if not current["segment_ids"]:
            current["start"] = format_timecode(sentence["start_sec"])
            current["start_sec"] = sentence["start_sec"]

        current["end"] = format_timecode(sentence["end_sec"])
        current["end_sec"] = sentence["end_sec"]
        if sentence["segment_id"] not in current["segment_ids"]:
            current["segment_ids"].append(sentence["segment_id"])
        current["text"] = sentence_text if not current["text"] else f'{current["text"]} {sentence_text}'
        register_speaker(current, sentence["speaker"])
        current_len += addition_len

    flush()

    # Don't leave a short trailing chunk orphaned - fold it into the previous one.
    if len(raw_blocks) >= 2 and len(raw_blocks[-1]["text"]) < min_chars:
        tail = raw_blocks.pop()
        raw_blocks[-1] = merge_blocks(raw_blocks[-1], tail)

    return [finalize_block(block, idx, speaker_map) for idx, block in enumerate(raw_blocks, start=1)]


def main() -> None:
    args = parse_args()
    args.drafts_dir.mkdir(parents=True, exist_ok=True)
    speaker_map = load_speaker_map(args.speaker_map)

    files = sorted(args.segments_dir.glob("*.segments.jsonl"))
    if args.lecture_ids:
        allowed = set(args.lecture_ids)
        files = [path for path in files if lecture_id_from_segments_path(path) in allowed]
    if not files:
        raise SystemExit(f"No segment files found in {args.segments_dir}")

    total_blocks = 0
    for path in files:
        out_path = args.drafts_dir / path.name.replace(".segments.jsonl", ".draft.jsonl")
        if args.skip_existing and out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue

        segments = load_segments(path)
        blocks = build_semantic_blocks(
            segments=segments,
            speaker_map=speaker_map,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            breakpoint_percentile=args.breakpoint_percentile,
        )
        write_blocks(out_path, blocks)
        total_blocks += len(blocks)
        print(f"[done] {out_path} ({len(blocks)} blocks)")

    print(f"Draft annotations built: {total_blocks}")


if __name__ == "__main__":
    main()

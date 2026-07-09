"""
Alternative draft-builder: chunk by sentence boundaries with a min/max
character target, instead of the pause/duration/char-limit heuristic in
build_annotation_drafts.py.

Why: the pause-based chunker can cut a chunk wherever a silence/duration/char
limit happens to fall, which is often mid-thought - the annotation LLM then
has to summarize a decontextualized fragment, producing vague or wrong
summaries. This version never cuts inside a sentence, and enforces a minimum
chunk size so a stray one-word sentence ("Да.", "Ну.") can't become an
isolated chunk on its own - it gets absorbed into a neighboring chunk.

Timecodes are produced by scripts/chunking/timed_sentences.py, shared with
any other chunking strategy (semantic, recursive, ...) we add later - only
the grouping logic below (build_sentence_blocks) is specific to this
strategy.

Output format is identical to build_annotation_drafts.py's *.draft.jsonl, so
scripts/prefill_annotations.py works unchanged - just point --drafts-dir at
this script's output instead.

Usage:
    python scripts/build_annotation_drafts_by_sentence.py --lecture-id Lecture_2
"""

import argparse
import sys
from pathlib import Path
from typing import Any

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

DEFAULT_DRAFTS_DIR = ROOT / "data" / "drafts_by_sentence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build draft annotation blocks by accumulating sentences to a target size."
    )
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--drafts-dir", type=Path, default=DEFAULT_DRAFTS_DIR)
    parser.add_argument("--speaker-map", type=Path, default=DEFAULT_SPEAKER_MAP)
    parser.add_argument(
        "--min-chars",
        type=int,
        default=800,
        help="Don't close a chunk before it reaches at least this many characters "
        "(prevents stray short-sentence chunks).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1800,
        help="Close a chunk once adding the next sentence would exceed this many characters.",
    )
    parser.add_argument("--lecture-id", action="append", dest="lecture_ids")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def build_sentence_blocks(
    segments: list[dict[str, Any]],
    speaker_map: dict[str, dict[str, str]],
    min_chars: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    lecture_id = segments[0]["lecture_id"]
    timed_sentences = split_all_into_timed_sentences(segments)

    raw_blocks: list[dict[str, Any]] = []
    current = empty_block(lecture_id)
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current["segment_ids"]:
            raw_blocks.append(current)
        current = empty_block(lecture_id)
        current_len = 0

    for sentence in timed_sentences:
        sentence_text = sentence["text"]
        addition_len = len(sentence_text) + (1 if current_len else 0)

        if current["segment_ids"] and current_len >= min_chars and current_len + addition_len > max_chars:
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
        blocks = build_sentence_blocks(
            segments=segments,
            speaker_map=speaker_map,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        write_blocks(out_path, blocks)
        total_blocks += len(blocks)
        print(f"[done] {out_path} ({len(blocks)} blocks)")

    print(f"Draft annotations built: {total_blocks}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENTS_DIR = ROOT / "data" / "segments"
DEFAULT_DRAFTS_DIR = ROOT / "data" / "drafts"
DEFAULT_SPEAKER_MAP = ROOT / "data" / "annotations" / "speaker_map.json"
SHORT_TURN_MARKERS = {
    "да", "ага", "угу", "окей", "хорошо", "ладно", "понятно", "ясно",
    "спасибо", "так", "вот", "ну", "нет", "конечно", "отлично",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build draft annotation blocks from lecture segments."
    )
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=DEFAULT_SEGMENTS_DIR,
        help="Directory with *.segments.jsonl files.",
    )
    parser.add_argument(
        "--drafts-dir",
        type=Path,
        default=DEFAULT_DRAFTS_DIR,
        help="Directory to write draft annotation files (intermediate, regenerable).",
    )
    parser.add_argument(
        "--speaker-map",
        type=Path,
        default=DEFAULT_SPEAKER_MAP,
        help="Optional JSON with lecture/speaker -> professor|student mapping.",
    )
    parser.add_argument(
        "--max-gap-sec",
        type=float,
        default=12.0,
        help="Split block when pause between segments is greater than this value.",
    )
    parser.add_argument(
        "--max-block-sec",
        type=float,
        default=180.0,
        help="Split block when total block duration would exceed this value.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2200,
        help="Split block when block text would exceed this many characters.",
    )
    parser.add_argument(
        "--short-turn-max-sec",
        type=float,
        default=4.5,
        help="Treat very short conversational turns under this duration as absorbable.",
    )
    parser.add_argument(
        "--short-turn-max-tokens",
        type=int,
        default=8,
        help="Treat turns with no more than this many tokens as absorbable.",
    )
    parser.add_argument(
        "--merge-short-turn-gap-sec",
        type=float,
        default=6.0,
        help="Merge short standalone blocks into the previous block when the gap is small.",
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
        help="Skip draft files that already exist.",
    )
    return parser.parse_args()


def load_segments(path: Path) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                segments.append(json.loads(line))
    return segments


def load_speaker_map(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def tokenize(text: str) -> list[str]:
    return [token for token in normalize_text(text).lower().replace("?", " ").replace("!", " ").replace(".", " ").replace(",", " ").split() if token]


def looks_like_short_turn(
    text: str,
    duration_sec: float,
    max_sec: float,
    max_tokens: int,
) -> bool:
    tokens = tokenize(text)
    if not tokens:
        return True
    if duration_sec <= max_sec and len(tokens) <= max_tokens:
        return True
    if len(tokens) <= 3 and all(token in SHORT_TURN_MARKERS for token in tokens):
        return True
    return False


def register_speaker(block: dict[str, Any], speaker: Optional[str]) -> None:
    if not speaker:
        return
    block["speaker_counts"][speaker] = block["speaker_counts"].get(speaker, 0) + 1
    if speaker not in block["speakers_raw"]:
        block["speakers_raw"].append(speaker)
    block["speaker_raw"] = max(
        block["speaker_counts"],
        key=lambda name: (block["speaker_counts"][name], -block["speakers_raw"].index(name)),
    )


def append_segment(block: dict[str, Any], segment: dict[str, Any], text: str) -> None:
    if not block["segment_ids"]:
        block["start"] = segment["start"]
        block["start_sec"] = float(segment["start_sec"])

    block["segment_ids"].append(int(segment["segment_id"]))
    block["end"] = segment["end"]
    block["end_sec"] = float(segment["end_sec"])
    block["text"] = text if not block["text"] else f'{block["text"]} {text}'
    register_speaker(block, segment.get("speaker"))


def should_split(
    current: dict[str, Any],
    segment: dict[str, Any],
    max_gap_sec: float,
    max_block_sec: float,
    max_chars: int,
    short_turn_max_sec: float,
    short_turn_max_tokens: int,
) -> bool:
    if not current["segment_ids"]:
        return False

    same_speaker = current["speaker_raw"] == segment.get("speaker")
    gap = float(segment["start_sec"]) - float(current["end_sec"])
    new_duration = float(segment["end_sec"]) - float(current["start_sec"])
    segment_text = normalize_text(segment.get("text", ""))
    new_chars = len(current["text"]) + 1 + len(segment_text)
    short_turn = looks_like_short_turn(
        text=segment_text,
        duration_sec=float(segment["duration_sec"]),
        max_sec=short_turn_max_sec,
        max_tokens=short_turn_max_tokens,
    )

    if gap > max_gap_sec or new_duration > max_block_sec or new_chars > max_chars:
        return True

    if not same_speaker and not short_turn:
        return True

    return False


def empty_block(lecture_id: str) -> dict[str, Any]:
    return {
        "lecture_id": lecture_id,
        "segment_ids": [],
        "start": None,
        "end": None,
        "start_sec": None,
        "end_sec": None,
        "speaker_raw": None,
        "speakers_raw": [],
        "speaker_counts": {},
        "text": "",
    }


def merge_blocks(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = empty_block(left["lecture_id"])
    merged["start"] = left["start"]
    merged["start_sec"] = left["start_sec"]
    merged["end"] = right["end"]
    merged["end_sec"] = right["end_sec"]
    merged["segment_ids"] = left["segment_ids"] + right["segment_ids"]
    merged["text"] = normalize_text(f'{left["text"]} {right["text"]}')
    for speaker in left["speakers_raw"]:
        for _ in range(left["speaker_counts"].get(speaker, 0)):
            register_speaker(merged, speaker)
    for speaker in right["speakers_raw"]:
        for _ in range(right["speaker_counts"].get(speaker, 0)):
            register_speaker(merged, speaker)
    return merged


def merge_short_blocks(
    raw_blocks: list[dict[str, Any]],
    merge_short_turn_gap_sec: float,
    short_turn_max_sec: float,
    short_turn_max_tokens: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for block in raw_blocks:
        if not merged:
            merged.append(block)
            continue

        gap = float(block["start_sec"]) - float(merged[-1]["end_sec"])
        is_short_block = looks_like_short_turn(
            text=block["text"],
            duration_sec=float(block["end_sec"]) - float(block["start_sec"]),
            max_sec=short_turn_max_sec,
            max_tokens=short_turn_max_tokens,
        )

        if gap <= merge_short_turn_gap_sec and is_short_block:
            merged[-1] = merge_blocks(merged[-1], block)
        else:
            merged.append(block)
    return merged


def finalize_block(
    block: dict[str, Any],
    block_id: int,
    speaker_map: dict[str, dict[str, str]],
) -> dict[str, Any]:
    lecture_id = block["lecture_id"]
    raw_speaker = block["speaker_raw"]
    mapped_roles = {
        speaker_map.get(lecture_id, {}).get(speaker_name)
        for speaker_name in block["speakers_raw"]
    }
    mapped_roles.discard(None)
    speaker = next(iter(mapped_roles)) if len(mapped_roles) == 1 else None

    return {
        "id": block_id,
        "lecture_id": lecture_id,
        "segment_ids": block["segment_ids"],
        "topic": "",
        "keywords": [],
        "questions": [],
        "start": block["start"],
        "end": block["end"],
        "summary": "",
        "speaker": speaker,
        "speaker_raw": raw_speaker,
        "speakers_raw": block["speakers_raw"],
        "cognitive_level": "",
        "interaction_type": "",
        "embedding": [],
        "difficulty": None,
        "text": block["text"],
    }


def build_blocks(
    segments: list[dict[str, Any]],
    speaker_map: dict[str, dict[str, str]],
    max_gap_sec: float,
    max_block_sec: float,
    max_chars: int,
    short_turn_max_sec: float,
    short_turn_max_tokens: int,
    merge_short_turn_gap_sec: float,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    lecture_id = segments[0]["lecture_id"]
    raw_blocks: list[dict[str, Any]] = []
    current = empty_block(lecture_id)

    for segment in segments:
        text = normalize_text(segment.get("text", ""))
        if should_split(
            current,
            segment,
            max_gap_sec,
            max_block_sec,
            max_chars,
            short_turn_max_sec,
            short_turn_max_tokens,
        ):
            raw_blocks.append(current)
            current = empty_block(lecture_id)

        append_segment(current, segment, text)

    if current["segment_ids"]:
        raw_blocks.append(current)

    raw_blocks = merge_short_blocks(
        raw_blocks=raw_blocks,
        merge_short_turn_gap_sec=merge_short_turn_gap_sec,
        short_turn_max_sec=short_turn_max_sec,
        short_turn_max_tokens=short_turn_max_tokens,
    )

    return [
        finalize_block(block, idx, speaker_map)
        for idx, block in enumerate(raw_blocks, start=1)
    ]


def write_blocks(path: Path, blocks: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in blocks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def lecture_id_from_segments_path(path: Path) -> str:
    return path.name.removesuffix(".segments.jsonl")


def main() -> None:
    args = parse_args()
    args.drafts_dir.mkdir(parents=True, exist_ok=True)
    speaker_map = load_speaker_map(args.speaker_map)

    files = sorted(args.segments_dir.glob("*.segments.jsonl"))
    if args.lecture_ids:
        allowed = set(args.lecture_ids)
        files = [
            path for path in files
            if lecture_id_from_segments_path(path) in allowed
        ]
    if not files:
        raise SystemExit(f"No segment files found in {args.segments_dir}")

    total_blocks = 0
    for path in files:
        out_path = args.drafts_dir / path.name.replace(".segments.jsonl", ".draft.jsonl")
        if args.skip_existing and out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue

        segments = load_segments(path)
        blocks = build_blocks(
            segments=segments,
            speaker_map=speaker_map,
            max_gap_sec=args.max_gap_sec,
            max_block_sec=args.max_block_sec,
            max_chars=args.max_chars,
            short_turn_max_sec=args.short_turn_max_sec,
            short_turn_max_tokens=args.short_turn_max_tokens,
            merge_short_turn_gap_sec=args.merge_short_turn_gap_sec,
        )
        write_blocks(out_path, blocks)
        total_blocks += len(blocks)
        print(f"[done] {out_path} ({len(blocks)} blocks)")

    print(f"Draft annotations built: {total_blocks}")


if __name__ == "__main__":
    main()

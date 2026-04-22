import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"
SEGMENTS_DIR = ROOT / "data" / "segments"

TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}\.\d{3}|\d+:\d{2}:\d{2}\.\d{3}")
LINE_RE = re.compile(r"^\[(?P<span>[^\]]+)\]\s*(?:(?P<speaker>[^:]+):\s*)?(?P<text>.*)$")


def brief_to_seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 2:
        mins, sec_ms = parts
        hours = 0
    elif len(parts) == 3:
        hours, mins, sec_ms = parts
        hours = int(hours)
    else:
        raise ValueError(f"Unsupported timestamp format: {value}")

    secs, millis = sec_ms.split(".")
    return hours * 3600 + int(mins) * 60 + int(secs) + int(millis) / 1000


def seconds_to_json_time(value: float) -> str:
    hours = int(value // 3600)
    value -= hours * 3600
    mins = int(value // 60)
    value -= mins * 60
    secs = int(value)
    millis = int(round((value - secs) * 1000))
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def parse_transcript_line(line: str):
    match = LINE_RE.match(line.strip())
    if not match:
        return None

    timestamps = TIMESTAMP_RE.findall(match.group("span"))
    if len(timestamps) < 2:
        return None

    start_sec = round(brief_to_seconds(timestamps[0]), 3)
    end_sec = round(brief_to_seconds(timestamps[1]), 3)
    text = " ".join(match.group("text").split())
    speaker = match.group("speaker")
    if speaker:
        speaker = speaker.strip()

    return {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "speaker": speaker or None,
        "text": text,
    }


def build_segments(transcript_path: Path):
    segments = []
    with transcript_path.open("r", encoding="utf-8") as file:
        for line in file:
            parsed = parse_transcript_line(line)
            if parsed is not None:
                segments.append(parsed)
    return segments


def write_segments(lecture_id: str, segments):
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SEGMENTS_DIR / f"{lecture_id}.segments.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for idx, segment in enumerate(segments, start=1):
            item = {
                "segment_id": idx,
                "lecture_id": lecture_id,
                "start": seconds_to_json_time(segment["start_sec"]),
                "end": seconds_to_json_time(segment["end_sec"]),
                "start_sec": segment["start_sec"],
                "end_sec": segment["end_sec"],
                "duration_sec": round(segment["end_sec"] - segment["start_sec"], 3),
                "speaker": segment["speaker"],
                "text": segment["text"],
            }
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    return output_path


def main():
    created = []
    for transcript_path in sorted(TRANSCRIPTS_DIR.glob("*.txt")):
        lecture_id = transcript_path.stem
        output_path = SEGMENTS_DIR / f"{lecture_id}.segments.jsonl"
        if output_path.exists():
            continue

        segments = build_segments(transcript_path)
        if not segments:
            print(f"[skip] {transcript_path.name}: no parsable segments")
            continue

        write_segments(lecture_id, segments)
        created.append(output_path.name)
        print(f"[done] {output_path}")

    print(f"Created: {len(created)}")


if __name__ == "__main__":
    main()

"""
Strip timecodes/speaker tags from data/transcripts/*.txt into plain reading
text, once, so eval question-writing doesn't need to re-clean lectures each
time.

Produces:
  data/eval/clean_transcripts/<lecture>.txt   - one clean file per lecture
  data/eval/clean_transcripts/_all.txt        - all lectures concatenated,
                                                 with a header per lecture
"""

import re
from pathlib import Path

SRC_DIR = Path("data/transcripts")
OUT_DIR = Path("data/eval/clean_transcripts")

LINE_RE = re.compile(r"^\[\d{2}:\d{2}\.\d{3}[-‐-―]\d{2}:\d{2}\.\d{3}\]\s*(SPK\d+):\s*(.*)$")

LECTURE_NUM_RE = re.compile(r"Lecture_(\d+)(?:\.(\d+))?$")


def lecture_sort_key(path: Path):
    match = LECTURE_NUM_RE.match(path.stem)
    if not match:
        return (float("inf"), 0)
    major, minor = match.groups()
    return (int(major), int(minor) if minor else 0)


def clean_lecture(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    paragraphs = []
    current_speaker = None
    buffer = []

    for line in lines:
        match = LINE_RE.match(line.strip())
        if not match:
            continue
        speaker, text = match.groups()
        text = text.strip()
        if not text:
            continue

        if speaker != current_speaker:
            if buffer:
                paragraphs.append(" ".join(buffer))
            buffer = [text]
            current_speaker = speaker
        else:
            buffer.append(text)

    if buffer:
        paragraphs.append(" ".join(buffer))

    return "\n\n".join(paragraphs)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_parts = []

    for path in sorted(SRC_DIR.glob("*.txt"), key=lecture_sort_key):
        lecture_id = path.stem
        cleaned = clean_lecture(path)
        (OUT_DIR / f"{lecture_id}.txt").write_text(cleaned, encoding="utf-8")
        all_parts.append(f"=== {lecture_id} ===\n\n{cleaned}")
        print(f"{lecture_id}: {len(cleaned)} chars")

    (OUT_DIR / "_all.txt").write_text("\n\n\n".join(all_parts), encoding="utf-8")
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()

"""One-shot pipeline: drop video/audio files into a folder and run this script
to transcribe, annotate and add them to the RAG index in a single command.

Wraps the existing step-by-step scripts (convert -> transcribe -> annotate ->
prefill -> index) and only touches the lectures that are actually new, so it
is safe to re-run after adding more files.

Examples:
    python scripts/ingest_lectures.py
    python scripts/ingest_lectures.py --model large-v3 --diarize
    python scripts/ingest_lectures.py data/raw/lectures --language ru --merge
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
RAW_LECTURES_DIR = ROOT / "data" / "raw" / "lectures"
SEGMENTS_DIR = ROOT / "data" / "segments"

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4a", ".mp3", ".flac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe, annotate and index new lecture files in one command."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(RAW_LECTURES_DIR),
        help="Folder with new video/audio files (default: data/raw/lectures).",
    )
    parser.add_argument("--model", default="medium", help="faster-whisper model size.")
    parser.add_argument("--language", default="ru", help="Recognition language.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization.")
    parser.add_argument("--safe_int8", action="store_true", help="Use int8_float16 on GPU.")
    parser.add_argument("--merge", action="store_true", help="Merge nearby same-speaker segments.")
    parser.add_argument("--max-len", type=float, default=45.0)
    parser.add_argument("--max-gap", type=float, default=1.5)
    parser.add_argument("--include-eng", action="store_true", help="Include *_eng files.")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use heuristic annotation rules instead of the local LLM (e.g. LM Studio is not running).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Fragments annotated in parallel via the LLM (match LM Studio's Max Concurrent Predictions).",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def convert_videos_to_wav(folder: Path) -> None:
    """Convert any non-WAV audio/video sitting directly in `folder` to WAV,
    so the transcriber (WAV-only) picks it up."""
    if not folder.is_dir():
        return

    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        wav_path = path.with_suffix(".wav")
        if wav_path.exists():
            continue
        print(f"[convert] {path.name} -> {wav_path.name}")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(path),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(wav_path),
            ],
            check=True,
        )


def existing_lecture_ids() -> set[str]:
    if not SEGMENTS_DIR.exists():
        return set()
    return {p.name.removesuffix(".segments.jsonl") for p in SEGMENTS_DIR.glob("*.segments.jsonl")}


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)

    print(f"== 1/4: converting video/audio to WAV in {input_dir} ==")
    convert_videos_to_wav(input_dir)

    before = existing_lecture_ids()

    print("== 2/4: transcribing new files ==")
    whisper_cmd = [
        sys.executable, str(SCRIPTS_DIR / "OpenAI-Whisper.py"), str(input_dir),
        "--model", args.model,
        "--language", args.language,
        "--device", args.device,
        "--max-len", str(args.max_len),
        "--max-gap", str(args.max_gap),
    ]
    if args.diarize:
        whisper_cmd.append("--diarize")
    if args.safe_int8:
        whisper_cmd.append("--safe_int8")
    if args.merge:
        whisper_cmd.append("--merge")
    if args.include_eng:
        whisper_cmd.append("--include-eng")
    run(whisper_cmd)

    after = existing_lecture_ids()
    new_lecture_ids = sorted(after - before)

    if not new_lecture_ids:
        print("\nNo new lectures were transcribed - nothing to annotate or index.")
        return

    print(f"\nNew lectures: {', '.join(new_lecture_ids)}")

    lecture_id_args: list[str] = []
    for lecture_id in new_lecture_ids:
        lecture_id_args += ["--lecture-id", lecture_id]

    print("== 3/4: building annotations ==")
    run([sys.executable, str(SCRIPTS_DIR / "build_annotation_drafts.py"), *lecture_id_args])

    prefill_cmd = [
        sys.executable, str(SCRIPTS_DIR / "prefill_annotations.py"), *lecture_id_args,
        "--concurrency", str(args.concurrency),
    ]
    if args.no_llm:
        prefill_cmd.append("--no-llm")
    run(prefill_cmd)

    print("== 4/4: rebuilding RAG index ==")
    run([sys.executable, str(SCRIPTS_DIR / "embed_kb.py")])

    print(f"\nDone. Added {len(new_lecture_ids)} lecture(s) to the RAG index: {', '.join(new_lecture_ids)}")


if __name__ == "__main__":
    main()

# scripts/OpenAI-Whisper.py

r"""
CLI wrapper for src/audio/whisper_transcriber.py.

Examples:
python .\scripts\OpenAI-Whisper.py
python .\scripts\OpenAI-Whisper.py .\data\raw\lectures --model large-v3 --language ru --safe_int8 --diarize --merge --max-len 45 --max-gap 1.5
python .\scripts\OpenAI-Whisper.py .\data\raw\lectures\unprocessed

When pointed at data/raw/lectures, the script:
- creates data/raw/lectures/unprocessed and data/raw/lectures/processed
- sorts existing WAV files into those folders
- processes only files from unprocessed
- skips *_eng.wav by default
- moves successfully processed WAV files into processed
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DEFAULT_INPUT = ROOT / "data" / "raw" / "lectures"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from audio.whisper_transcriber import TranscriptionConfig, run_transcription


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> TranscriptionConfig:
    ap = argparse.ArgumentParser(
        description="faster-whisper WAV transcriber with lecture folder organization"
    )
    ap.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="Path to a WAV file or a folder with WAV files. Default: data/raw/lectures",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="medium",
        help="tiny/base/small/medium/large-v3 (default: medium)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default="ru",
        help="Recognition language (default: ru)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use: auto/cuda/cpu (default: auto)",
    )
    ap.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (pyannote)",
    )
    ap.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HUGGINGFACE_TOKEN"),
        help="HuggingFace token",
    )
    ap.add_argument(
        "--safe_int8",
        action="store_true",
        help="Use int8_float16 on GPU to reduce VRAM usage",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="Merge nearby segments of the same speaker",
    )
    ap.add_argument(
        "--max-len",
        type=float,
        default=45.0,
        help="Max merged segment length in seconds",
    )
    ap.add_argument(
        "--max-gap",
        type=float,
        default=1.5,
        help="Max silence gap for merging in seconds",
    )
    ap.add_argument(
        "--no-organize-lectures",
        action="store_true",
        help="Do not auto-create processed/unprocessed folders and do not sort WAV files",
    )
    ap.add_argument(
        "--keep-processed-files",
        action="store_true",
        help="Do not move successfully processed WAV files into the processed folder",
    )
    ap.add_argument(
        "--include-eng",
        action="store_true",
        help="Include *_eng.wav files in batch processing",
    )

    args = ap.parse_args()

    return TranscriptionConfig(
        input_path=args.input_path,
        model=args.model,
        language=args.language,
        device=args.device,
        diarize=args.diarize,
        hf_token=args.hf_token,
        safe_int8=args.safe_int8,
        merge=args.merge,
        max_len=args.max_len,
        max_gap=args.max_gap,
        organize_lectures=not args.no_organize_lectures,
        move_processed=not args.keep_processed_files,
        include_eng=args.include_eng,
    )


def main():
    configure_stdio()
    cfg = parse_args()
    run_transcription(cfg)


if __name__ == "__main__":
    main()

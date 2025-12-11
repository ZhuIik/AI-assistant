# scripts/OpenAI-Whisper.py

"""
CLI-обёртка для транскрибатора из src/audio/whisper_transcriber.py
python .\scripts\OpenAI-Whisper.py .\data\raw\lectures\Lecture_6.1.wav --model large-v3 --language ru --safe_int8 --diarize --merge --max-len 45 --max-gap 1.5
"""

import os
import sys
import argparse
from pathlib import Path

# --- чтобы можно было импортировать src, если запускаем из корня ---
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from audio.whisper_transcriber import TranscriptionConfig, run_transcription


def parse_args() -> TranscriptionConfig:
    ap = argparse.ArgumentParser(
        description="faster-whisper WAV transcriber with diarization and merging"
    )
    ap.add_argument("input_path", type=str, help="Путь к WAV-файлу или папке с WAV")
    ap.add_argument("--model", type=str, default="medium",
                    help="tiny/base/small/medium/large-v3 (default: medium)")
    ap.add_argument("--language", type=str, default="ru",
                    help="Язык распознавания (default: ru)")
    ap.add_argument("--diarize", action="store_true",
                    help="Включить спикеров (pyannote)")
    ap.add_argument("--hf_token", type=str,
                    default=os.getenv("HUGGINGFACE_TOKEN", "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"),
                    help="HuggingFace token")
    ap.add_argument("--safe_int8", action="store_true",
                    help="GPU: compute_type=int8_float16 (экономия VRAM)")
    ap.add_argument("--merge", action="store_true",
                    help="Склеивать сегменты в пределах одного спикера")
    ap.add_argument("--max-len", type=float, default=45.0,
                    help="Макс. длина объединённого сегмента (сек)")
    ap.add_argument("--max-gap", type=float, default=1.5,
                    help="Макс. пауза между сегментами для склейки (сек)")

    args = ap.parse_args()

    return TranscriptionConfig(
        input_path=args.input_path,
        model=args.model,
        language=args.language,
        diarize=args.diarize,
        hf_token=args.hf_token,
        safe_int8=args.safe_int8,
        merge=args.merge,
        max_len=args.max_len,
        max_gap=args.max_gap,
    )


def main():
    cfg = parse_args()
    run_transcription(cfg)


if __name__ == "__main__":
    main()





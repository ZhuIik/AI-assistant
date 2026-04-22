import glob
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Work around duplicate OpenMP runtime initialization on Windows when
# torch / faster-whisper / diarization dependencies load different copies.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch
except Exception:
    torch = None

from faster_whisper import WhisperModel

SUPPORTED_EXTS = (".wav",)
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_LECTURES_DIR = DATA_DIR / "raw" / "lectures"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
SEGMENTS_DIR = DATA_DIR / "segments"
ANNOTATIONS_DIR = DATA_DIR / "annotations"


@dataclass
class TranscriptionConfig:
    input_path: str
    model: str = "medium"
    language: str = "ru"
    device: str = "auto"
    diarize: bool = False
    hf_token: str = ""
    safe_int8: bool = False
    merge: bool = False
    max_len: float = 45.0
    max_gap: float = 1.5
    organize_lectures: bool = True
    move_processed: bool = True
    include_eng: bool = False


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def sec_to_timestamp(t: float) -> str:
    if t is None:
        t = 0.0
    hrs = int(t // 3600)
    t -= 3600 * hrs
    mins = int(t // 60)
    t -= 60 * mins
    secs = int(t)
    ms = int(round((t - secs) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def sec_to_brief(t: float) -> str:
    if t is None:
        t = 0.0
    hrs = int(t // 3600)
    t -= 3600 * hrs
    mins = int(t // 60)
    t -= 60 * mins
    secs = int(t)
    ms = int(round((t - secs) * 1000))
    return f"{hrs}:{mins:02d}:{secs:02d}.{ms:03d}" if hrs > 0 else f"{mins:02d}:{secs:02d}.{ms:03d}"


def sec_to_json_time(t: float) -> str:
    if t is None:
        t = 0.0
    hrs = int(t // 3600)
    t -= 3600 * hrs
    mins = int(t // 60)
    t -= 60 * mins
    secs = int(t)
    ms = int(round((t - secs) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"


def write_txt_segments(segments: List[dict], out_path: str, speaker_by_idx: Optional[List[str]] = None) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = sec_to_brief(seg["start"])
            end = sec_to_brief(seg["end"])
            text = " ".join((seg["text"] or "").split())
            speaker = (speaker_by_idx[i - 1] if speaker_by_idx else "").strip() if speaker_by_idx else ""
            if speaker:
                f.write(f"[{start}-{end}] {speaker}: {text}\n")
            else:
                f.write(f"[{start}-{end}] {text}\n")


def write_jsonl_segments(
    segments: List[dict],
    out_path: str,
    lecture_id: str,
    speaker_by_idx: Optional[List[str]] = None,
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            text = " ".join((seg["text"] or "").split())
            item = {
                "segment_id": i,
                "lecture_id": lecture_id,
                "start": sec_to_json_time(seg["start"]),
                "end": sec_to_json_time(seg["end"]),
                "start_sec": round(float(seg["start"]), 3),
                "end_sec": round(float(seg["end"]), 3),
                "duration_sec": round(float(seg["end"]) - float(seg["start"]), 3),
                "speaker": (speaker_by_idx[i - 1] if speaker_by_idx else None),
                "text": text,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def ensure_output_dirs() -> None:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_lecture_dirs(root_dir: Path) -> Tuple[Path, Path]:
    unprocessed_dir = root_dir / "unprocessed"
    processed_dir = root_dir / "processed"
    unprocessed_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return unprocessed_dir, processed_dir


def has_existing_segmentation(lecture_id: str) -> bool:
    return (SEGMENTS_DIR / f"{lecture_id}.segments.jsonl").exists()


def organize_lecture_directory(root_dir: Path) -> Tuple[Path, Path]:
    unprocessed_dir, processed_dir = ensure_lecture_dirs(root_dir)

    for path in sorted(root_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue

        target_dir = processed_dir if has_existing_segmentation(path.stem) else unprocessed_dir
        target_path = target_dir / path.name

        if target_path.exists():
            print(f"[skip] {path} -> {target_path} (already exists)")
            continue

        shutil.move(str(path), str(target_path))
        print(f"[move] {path} -> {target_path}")

    return unprocessed_dir, processed_dir


def should_process_audio_file(path: Path, include_eng: bool) -> bool:
    if path.suffix.lower() not in SUPPORTED_EXTS:
        return False
    if include_eng:
        return True
    return not path.stem.lower().endswith("_eng")


def resolve_input_files(
    input_path: Path,
    organize_lectures: bool,
    include_eng: bool,
) -> Tuple[List[Path], Optional[Path]]:
    if input_path.is_file():
        processed_dir = None
        if organize_lectures and input_path.parent.name.lower() == "unprocessed":
            _, processed_dir = ensure_lecture_dirs(input_path.parent.parent)
        return [input_path], processed_dir

    if not input_path.is_dir():
        print(f"Path not found: {input_path}")
        raise SystemExit(2)

    processed_dir: Optional[Path] = None

    if organize_lectures and input_path.resolve() == RAW_LECTURES_DIR.resolve():
        input_path, processed_dir = organize_lecture_directory(input_path)
    elif organize_lectures and input_path.name.lower() == "unprocessed":
        _, processed_dir = ensure_lecture_dirs(input_path.parent)
    elif organize_lectures and (
        (input_path / "unprocessed").exists() or (input_path / "processed").exists()
    ):
        input_path, processed_dir = organize_lecture_directory(input_path)

    files = [
        Path(p)
        for p in list_wavs(str(input_path))
        if should_process_audio_file(Path(p), include_eng=include_eng)
    ]
    return files, processed_dir


def move_to_processed(path: Path, processed_dir: Optional[Path]) -> None:
    if processed_dir is None or path.parent.resolve() == processed_dir.resolve():
        return

    target_path = processed_dir / path.name
    if target_path.exists():
        print(f"[skip] {path} -> {target_path} (already exists)")
        return

    shutil.move(str(path), str(target_path))
    print(f"[move] {path} -> {target_path}")


def load_diarization_pipeline(hf_token: Optional[str]):
    if not hf_token:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN is not set. Pass --hf_token or define it in the environment."
        )
    from pyannote.audio import Pipeline

    return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)


def diarize_file(pipe, path: str) -> List[Tuple[float, float, str]]:
    diar = pipe(path)
    out = []
    for seg, _, spk in diar.itertracks(yield_label=True):
        out.append((float(seg.start), float(seg.end), str(spk)))
    return out


def assign_speakers_to_segments(
    whisper_segments: List[dict],
    diar_segments: List[Tuple[float, float, str]],
) -> List[str]:
    speakers = []
    for ws in whisper_segments:
        ws_start, ws_end = float(ws["start"]), float(ws["end"])
        best_spk, best_ovlp = None, 0.0
        for ds_start, ds_end, ds_spk in diar_segments:
            ovlp = max(0.0, min(ws_end, ds_end) - max(ws_start, ds_start))
            if ovlp > best_ovlp:
                best_ovlp, best_spk = ovlp, ds_spk
        speakers.append(best_spk or "SPK")

    mapping, cnt, canon = {}, 1, []
    for spk in speakers:
        if spk not in mapping:
            mapping[spk] = f"SPK{cnt:02d}"
            cnt += 1
        canon.append(mapping[spk])
    return canon


def merge_segments_with_speakers(
    segments: List[dict],
    speakers: Optional[List[str]],
    max_gap: float,
    max_len: float,
) -> Tuple[List[dict], Optional[List[str]]]:
    if not segments:
        return [], speakers

    merged: List[dict] = []
    merged_spk: Optional[List[str]] = [] if speakers else None

    cur = {"start": segments[0]["start"], "end": segments[0]["end"], "text": segments[0]["text"]}
    cur_spk = speakers[0] if speakers else None

    for seg_idx in range(1, len(segments)):
        seg = segments[seg_idx]
        spk = speakers[seg_idx] if speakers else None

        same_spk = (spk == cur_spk) if speakers else True
        gap = seg["start"] - cur["end"]
        new_len = seg["end"] - cur["start"]

        if same_spk and gap <= max_gap and new_len <= max_len:
            cur["end"] = seg["end"]
            cur["text"] = (cur["text"] + " " + seg["text"]).strip()
        else:
            merged.append(cur)
            if merged_spk is not None:
                merged_spk.append(cur_spk)
            cur = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            cur_spk = spk

    merged.append(cur)
    if merged_spk is not None:
        merged_spk.append(cur_spk)
    return merged, merged_spk


def list_wavs(root: str):
    if os.path.isfile(root):
        return [root] if root.lower().endswith(SUPPORTED_EXTS) else []
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(glob.glob(os.path.join(root, f"*{ext}")))
    return sorted(files)


def pick_device_and_compute_type(device: str = "auto", prefer_fp16: bool = True) -> tuple:
    requested = (device or "auto").lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported device: {device}")

    if requested == "cpu":
        return "cpu", "int8"

    cuda_available = torch is not None and torch.cuda.is_available()

    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("Requested device=cuda, but CUDA is not available.")
        return "cuda", ("float16" if prefer_fp16 else "int8_float16")

    if cuda_available:
        return "cuda", ("float16" if prefer_fp16 else "int8_float16")
    return "cpu", "int8"


def transcribe_one(
    model: WhisperModel,
    path: str,
    language: str,
    diarize: bool,
    pipe,
    do_merge: bool,
    max_gap: float,
    max_len: float,
) -> None:
    lecture_id = Path(path).stem
    ensure_output_dirs()
    out_txt = TRANSCRIPTS_DIR / f"{lecture_id}.txt"
    out_jsonl = SEGMENTS_DIR / f"{lecture_id}.segments.jsonl"

    print(f"[run ] {path}")
    seg_it, info = model.transcribe(
        path,
        language=language,
        task="transcribe",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=800),
        beam_size=5,
        best_of=5,
    )

    segments = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in seg_it]

    speaker_by_idx: Optional[List[str]] = None
    if diarize:
        diar_segments = diarize_file(pipe, path)
        speaker_by_idx = assign_speakers_to_segments(segments, diar_segments)

    if do_merge and max_len > 0:
        segments, speaker_by_idx = merge_segments_with_speakers(
            segments,
            speaker_by_idx,
            max_gap=max_gap,
            max_len=max_len,
        )

    write_txt_segments(segments, str(out_txt), speaker_by_idx)
    write_jsonl_segments(segments, str(out_jsonl), lecture_id, speaker_by_idx)

    print(f"[done] {out_txt}")
    print(f"[done] {out_jsonl}")


def run_transcription(cfg: TranscriptionConfig):
    configure_stdio()

    device, compute_type = pick_device_and_compute_type(
        device=cfg.device,
        prefer_fp16=not cfg.safe_int8,
    )
    print(f"Device: {device} | compute_type: {compute_type}")
    print(f"Loading model: {cfg.model}")

    try:
        model = WhisperModel(cfg.model, device=device, compute_type=compute_type)
    except Exception as e:
        msg = str(e)
        if device == "cuda" and "cudnn_ops64_9.dll" in msg:
            raise RuntimeError(
                "GPU mode is unavailable: faster-whisper/ctranslate2 could not find cuDNN 9. "
                "Run with --device cpu or fix CUDA/cuDNN in the environment."
            ) from e
        raise

    pipe = None
    if cfg.diarize:
        token = (cfg.hf_token or "").strip()
        if not token:
            token = (os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
        pipe = load_diarization_pipeline(token)

    input_path = Path(cfg.input_path)
    files, processed_dir = resolve_input_files(
        input_path,
        organize_lectures=cfg.organize_lectures,
        include_eng=cfg.include_eng,
    )

    if not files:
        print("No files to process.", file=sys.stderr)
        raise SystemExit(2)

    print(f"Files to process: {len(files)}")

    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path}")
        try:
            transcribe_one(
                model,
                str(path),
                cfg.language,
                cfg.diarize,
                pipe,
                do_merge=cfg.merge,
                max_gap=cfg.max_gap,
                max_len=cfg.max_len,
            )
            if cfg.move_processed:
                move_to_processed(path, processed_dir)
        except Exception as e:
            print(f"[fail] {path}: {e}", file=sys.stderr)

    print("All done.")

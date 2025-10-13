"""
Необходимые библиотеки:
pip install torch faster-whisper numpy soundfile pyannote.audio huggingface_hub
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Команды для запуска (замени "hf_ТВОЙ_ТОКЕН" на свой токен HuggingFace):
python OpenAI-Whisper.py "..\data\raw" --model medium --language ru --diarize --hf_token "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" --merge --max-len 45 --max-gap 1.5
python OpenAI-Whisper.py "../data/raw" --model large-v3 --language ru --safe_int8 --diarize --hf_token "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" --merge --max-len 45 --max-gap 1.5


Transcribe WAV files using faster-whisper on GPU/CPU, optional diarization (pyannote).
Outputs:
  - TXT: one line per segment with [start–end] and optional SPK label
  - SRT: subtitles, also with SPK labels if diarization is enabled
"""

import os, sys, glob, argparse
from typing import List, Optional, Tuple

try:
    import torch
except Exception:
    torch = None

from faster_whisper import WhisperModel

SUPPORTED_EXTS = (".wav",)

# ---------- time formatting ----------

def sec_to_timestamp(t: float) -> str:
    if t is None: t = 0.0
    hrs = int(t // 3600); t -= 3600 * hrs
    mins = int(t // 60);  t -= 60 * mins
    secs = int(t);        ms = int(round((t - secs) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

def sec_to_brief(t: float) -> str:
    if t is None: t = 0.0
    hrs = int(t // 3600); t -= 3600 * hrs
    mins = int(t // 60);  t -= 60 * mins
    secs = int(t);        ms = int(round((t - secs) * 1000))
    return f"{hrs}:{mins:02d}:{secs:02d}.{ms:03d}" if hrs>0 else f"{mins:02d}:{secs:02d}.{ms:03d}"

# ---------- outputs ----------

def write_srt(segments: List[dict], out_path: str, speaker_by_idx: Optional[List[str]] = None) -> None:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = sec_to_timestamp(seg["start"])
        end   = sec_to_timestamp(seg["end"])
        text  = " ".join((seg["text"] or "").split())
        spk   = (speaker_by_idx[i-1] if speaker_by_idx else "").strip()
        if spk:
            text = f"{spk}: {text}"
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def write_txt_segments(segments: List[dict], out_path: str, speaker_by_idx: Optional[List[str]] = None) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            s = sec_to_brief(seg["start"])
            e = sec_to_brief(seg["end"])
            text = " ".join((seg["text"] or "").split())
            spk  = (speaker_by_idx[i-1] if speaker_by_idx else "").strip() if speaker_by_idx else ""
            if spk:
                f.write(f"[{s}–{e}] {spk}: {text}\n")
            else:
                f.write(f"[{s}–{e}] {text}\n")

# ---------- diarization ----------

def load_diarization_pipeline(hf_token: Optional[str]):
    if not hf_token:
        raise RuntimeError("HUGGINGFACE_TOKEN не задан. Передай --hf_token или задай переменную окружения.")
    from pyannote.audio import Pipeline
    return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

def diarize_file(pipe, path: str) -> List[Tuple[float, float, str]]:
    diar = pipe(path)
    out = []
    for seg, _, spk in diar.itertracks(yield_label=True):
        out.append((float(seg.start), float(seg.end), str(spk)))
    return out

def assign_speakers_to_segments(whisper_segments: List[dict],
                                diar_segments: List[Tuple[float, float, str]]) -> List[str]:
    speakers = []
    for ws in whisper_segments:
        ws_start, ws_end = float(ws["start"]), float(ws["end"])
        best_spk, best_ovlp = None, 0.0
        for ds_start, ds_end, ds_spk in diar_segments:
            ovlp = max(0.0, min(ws_end, ds_end) - max(ws_start, ds_start))
            if ovlp > best_ovlp:
                best_ovlp, best_spk = ovlp, ds_spk
        speakers.append(best_spk or "SPK")
    # canonical order: SPK01, SPK02 ...
    mapping, cnt, canon = {}, 1, []
    for spk in speakers:
        if spk not in mapping:
            mapping[spk] = f"SPK{cnt:02d}"; cnt += 1
        canon.append(mapping[spk])
    return canon

# ---------- merging (per speaker) ----------

def merge_segments_with_speakers(segments: List[dict],
                                 speakers: Optional[List[str]],
                                 max_gap: float,
                                 max_len: float) -> Tuple[List[dict], Optional[List[str]]]:
    """
    Склеивает только соседние сегменты одного и того же спикера,
    если пауза <= max_gap и итоговая длительность блока <= max_len.
    """
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
            if merged_spk is not None: merged_spk.append(cur_spk)
            cur = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            cur_spk = spk

    merged.append(cur)
    if merged_spk is not None: merged_spk.append(cur_spk)
    return merged, merged_spk

# ---------- IO ----------

def list_wavs(root: str):
    if os.path.isfile(root):
        return [root] if root.lower().endswith(SUPPORTED_EXTS) else []
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(glob.glob(os.path.join(root, f"*{ext}")))
    return sorted(files)

# ---------- device ----------

def pick_device_and_compute_type(prefer_fp16: bool = True) -> tuple:
    if torch is not None and torch.cuda.is_available():
        return "cuda", ("float16" if prefer_fp16 else "int8_float16")
    else:
        return "cpu", "int8"

# ---------- core ----------

def transcribe_one(model: WhisperModel,
                   path: str,
                   language: str,
                   diarize: bool,
                   pipe,
                   do_merge: bool,
                   max_gap: float,
                   max_len: float) -> None:
    base, _ = os.path.splitext(path)
    out_txt, out_srt = base + ".txt", base + ".srt"

    print(f"[run ] {path}")
    seg_it, info = model.transcribe(
        path,
        language=language,
        task="transcribe",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=800),  # можно менять
        beam_size=5,
        best_of=5
    )

    segments = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in seg_it]

    speaker_by_idx: Optional[List[str]] = None
    if diarize:
        diar_segments = diarize_file(pipe, path)
        speaker_by_idx = assign_speakers_to_segments(segments, diar_segments)

    if do_merge and max_len > 0:
        segments, speaker_by_idx = merge_segments_with_speakers(
            segments, speaker_by_idx, max_gap=max_gap, max_len=max_len
        )

    write_txt_segments(segments, out_txt, speaker_by_idx)
    write_srt(segments, out_srt, speaker_by_idx)

    print(f"[done] {out_txt}")
    print(f"[done] {out_srt}")

def main():
    ap = argparse.ArgumentParser(description="faster-whisper WAV transcriber with diarization and merging")
    ap.add_argument("input_path", type=str, help="Путь к WAV-файлу или папке с WAV")
    ap.add_argument("--model", type=str, default="medium", help="tiny/base/small/medium/large-v3 (default: medium)")
    ap.add_argument("--language", type=str, default="ru", help="Язык распознавания (default: ru)")
    ap.add_argument("--diarize", action="store_true", help="Включить спикеров (pyannote)")
    ap.add_argument("--hf_token", type=str, default=os.getenv("HUGGINGFACE_TOKEN", ""), help="HuggingFace token")
    ap.add_argument("--safe_int8", action="store_true", help="GPU: compute_type=int8_float16 (экономия VRAM)")
    ap.add_argument("--merge", action="store_true", help="Склеивать сегменты в пределах одного спикера")
    ap.add_argument("--max-len", type=float, default=45.0, help="Макс. длина объединённого сегмента (сек)")
    ap.add_argument("--max-gap", type=float, default=1.5, help="Макс. пауза между сегментами для склейки (сек)")
    args = ap.parse_args()

    device, compute_type = pick_device_and_compute_type(prefer_fp16=not args.safe_int8)
    print(f"Device: {device} | compute_type: {compute_type}")
    print(f"Loading model: {args.model}")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    pipe = None
    if args.diarize:
        token = (args.hf_token or "").strip()
        pipe = load_diarization_pipeline(token)

    files = list_wavs(args.input_path)
    if not files:
        print("Нет WAV-файлов для обработки.", file=sys.stderr); sys.exit(2)

    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {p}")
        try:
            transcribe_one(model, p, args.language, args.diarize, pipe,
                           do_merge=args.merge, max_gap=args.max_gap, max_len=args.max_len)
        except Exception as e:
            print(f"[fail] {p}: {e}", file=sys.stderr)

    print("All done.")

if __name__ == "__main__":
    main()
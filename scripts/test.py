python fw_wav_diarize.py "C:\Users\Timofey\Desktop\Video Transcribe" --model medium --language ru --diarize --hf_token "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" --merge --max-len 45 --max-gap 1.5

import os, pathlib
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HOME"] = str(pathlib.Path.cwd() / ".hf_cache")
os.environ["PYCTCDECODE_CACHE_DIR"] = str(pathlib.Path.cwd() / ".pyctc_cache")

# --- маячок: покажет, какой файл реально запускается ---
from pathlib import Path
print("RUNNING:", __file__)

# --- подчистка от старых артефактов ---
for p in ["output.txt", "output_with_text.txt", "output.srt", "output_wholefile.txt"]:
    Path(p).unlink(missing_ok=True)

# ------------------ ДИАРИЗАЦИЯ ------------------
from pyannote.audio import Pipeline
HF_TOKEN = "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"
AUDIO_PATH = "input.wav"  # WAV 16 кГц, моно

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
d = pipeline(AUDIO_PATH)
segments = list(d.itertracks(yield_label=True))

# ------------------ ASR ------------------
import numpy as np, librosa, torch
from transformers import AutoProcessor, AutoModelForCTC

ASR_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
processor = AutoProcessor.from_pretrained(ASR_ID)
asr_model = AutoModelForCTC.from_pretrained(ASR_ID).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model.to(device)

wav, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)

MAX_CHUNK_SEC = 15.0
MIN_SEG_SEC = 0.5

def crop(sig, sr, start, end):
    s = int(start * sr); e = int(end * sr)
    return sig[s:e]

def transcribe_chunk(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    with torch.inference_mode():
        ins = processor(x.tolist(), sampling_rate=16000, return_tensors="pt", padding=True)
        if device.type == "cuda":
            ins = {k: v.to(device) for k, v in ins.items()}
        logits = asr_model(**ins).logits
        ids = logits.argmax(-1)
        return processor.batch_decode(ids)[0].strip()

def ts(t):
    h, rem = divmod(int(t), 3600); m, s = divmod(rem, 60); ms = int(round((t-int(t))*1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# Итоговые файлы (УНИКАЛЬНЫЕ имена)
OUT_TXT = Path("transcript_with_speakers.txt")
OUT_SRT = Path("transcript_with_speakers.srt")

with open(OUT_TXT, "w", encoding="utf-8") as f_txt, open(OUT_SRT, "w", encoding="utf-8") as f_srt:
    idx = 0
    for (turn, _, spk) in segments:
        st, en = float(turn.start), float(turn.end)
        dur = en - st
        if dur < MIN_SEG_SEC: 
            continue
        sig = crop(wav, sr, st, en)
        if sig.size == 0:
            continue

        # режем длинные сегменты
        parts = []
        n_parts = int(np.ceil(dur / MAX_CHUNK_SEC))
        for i in range(n_parts):
            pst, pen = i * MAX_CHUNK_SEC, min((i + 1) * MAX_CHUNK_SEC, dur)
            piece = sig[int(pst*sr): int(pen*sr)]
            if piece.size == 0:
                continue
            parts.append(transcribe_chunk(piece))
        text = " ".join(t for t in parts if t)

        f_txt.write(f"{st:.1f}–{en:.1f} | {spk}: {text}\n")
        idx += 1
        f_srt.write(f"{idx}\n{ts(st)} --> {ts(en)}\n{spk}: {text}\n\n")

print("DONE.\nTXT:", OUT_TXT, "\nSRT:", OUT_SRT)

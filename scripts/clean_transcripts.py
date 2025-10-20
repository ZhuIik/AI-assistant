import re, os, glob

IN_DIR  = "data/transcripts"
OUT_DIR = "data/cleaned"
os.makedirs(OUT_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    # 1. Удаляем спикеров
    text = re.sub(r"SPK\d{1,2}:", " ", text)
    
    # 2. Удаляем любые таймкоды
    text = re.sub(
        r"\[\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*[-–—→to]+\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*\]",
        " ",
        text
    )
    
    # 3. Повторяющиеся знаки
    text = re.sub(r"[.,!?]{2,}", ".", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    
    # 4. Междометия и короткие вставки
    text = re.sub(
        r"\b(да|угу|ага|всё|ладно|хорошо|ясно|понятно|ок|угу|угу,|ага,|всё,|всё\.|угу\.)\b",
        " ",
        text,
        flags=re.I
    )
    
    # 5. Убираем обрывки ", , ,"
    text = re.sub(r"(,\s*){2,}", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # 6. Удаляем короткие предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]
    
    # 7. Склеиваем обратно
    cleaned = " ".join(sentences)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned



for path in glob.glob(os.path.join(IN_DIR, "*.txt")):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = clean_text(raw)
    out_path = os.path.join(OUT_DIR, os.path.basename(path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)
    print("✔ Cleaned:", os.path.basename(path))

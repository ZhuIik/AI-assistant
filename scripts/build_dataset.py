import os, json

INPUT_DIR = "data/cleaned"
OUTPUT_FILE = "data/datasets/lectures_v1.jsonl"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

MAX_WORDS = 500  # длина чанка (можно 400–600)
MIN_WORDS = 80    # чтобы не брать слишком короткие куски

def chunk_text(text, max_words):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for file_name in sorted(os.listdir(INPUT_DIR)):
        if not file_name.endswith(".txt"):
            continue
        lecture_name = file_name.replace(".txt", "")
        with open(os.path.join(INPUT_DIR, file_name), "r", encoding="utf-8") as f:
            text = f.read().strip()
        for chunk in chunk_text(text, MAX_WORDS):
            if len(chunk.split()) < MIN_WORDS:
                continue
            record = {
                "instruction": f"Continue the lecture explanation about {lecture_name}.",
                "input": "",
                "output": chunk
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

print("✅ Dataset saved to", OUTPUT_FILE)

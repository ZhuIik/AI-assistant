import os
import json
import ollama

MODEL = "mistral:instruct"

def split_text(text, max_words=1000):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def summarize_chunk(chunk):
    prompt = (
        "Summarize this lecture segment clearly and concisely. "
        "Preserve key definitions and examples.\n\n" + chunk
    )
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"].strip()

# --- путь до папок ---
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
transcripts_path = os.path.join(base, "data", "transcripts")
output_path = os.path.join(base, "data", "summaries")

os.makedirs(output_path, exist_ok=True)

for filename in os.listdir(transcripts_path):
    if not filename.endswith(".txt"):
        continue

    print(f"\n📘 Обработка {filename}")
    with open(os.path.join(transcripts_path, filename), "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_text(text)
    summaries = []

    for i, chunk in enumerate(chunks):
        print(f"  → {i+1}/{len(chunks)}")
        summary = summarize_chunk(chunk)
        summaries.append({"lecture": filename, "chunk": i, "summary": summary})

    with open(os.path.join(output_path, filename.replace(".txt", "_summaries.json")), "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

print("\n✅ Всё готово! Саммари сохранены в data/summaries/")

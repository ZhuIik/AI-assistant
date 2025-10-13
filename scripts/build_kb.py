import os
import json

SUMMARIES_DIR = "../data/summaries"
OUTPUT_FILE = "../data/knowledge_base/kb.jsonl"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for filename in os.listdir(SUMMARIES_DIR):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(SUMMARIES_DIR, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ Knowledge base created:", OUTPUT_FILE)

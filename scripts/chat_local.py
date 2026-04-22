import os
import sys

# текущий файл: ...\Ai-assistant\scripts\chat_local.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)  # поднимаемся на уровень выше: Ai-assistant

sys.path.insert(0, ROOT)

from src.llm.chat import chat_once

if __name__ == "__main__":
    print("✅ Модель готова! Можешь задавать вопросы:\n")

    while True:
        prompt = input("❓ Вопрос: ")
        if prompt.lower() in ["exit", "quit", "stop"]:
            print("🛑 Завершено.")
            break

        answer = chat_once(prompt)
        print("\n💬 Ответ модели:\n", answer)
        print("-" * 80)

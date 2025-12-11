import os
import sys

from flask import Flask, request, jsonify

# ==== Настройка путей как в chat_local.py ====
# текущий файл: ...\Ai-assistant\scripts\server.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)  # поднимаемся на уровень выше: Ai-assistant

# добавляем корень проекта в sys.path, чтобы работал import src.llm.chat
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.llm.chat import chat_once  # используем твою модель + RAG


app = Flask(__name__)


def generate_answer(text: str) -> str:
    """
    Обертка над твоей функцией chat_once.
    Здесь пока просто пробрасываем вопрос и возвращаем ответ как есть.
    Если захотим потом добавить форматирование, источники и т.п. — делается тут.
    """
    answer = chat_once(text)
    return answer


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    """
    Эндпоинт, к которому стучится Telegram-бот.
    Ожидает JSON вида: {"text": "<вопрос>"}.
    Возвращает JSON: {"reply": "<ответ модели>"}.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "no text provided"}), 400

    try:
        reply = generate_answer(text)
    except Exception as e:
        # Чтобы в боте было видно, что именно пошло не так
        return jsonify({"error": f"internal error: {e}"}), 500

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

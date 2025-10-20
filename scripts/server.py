from flask import Flask, request, jsonify
from reg_chat import ask  # импортируем твою функцию

app = Flask(__name__)

@app.route("/message", methods=["POST"])
def handle_message():
    data = request.get_json()
    question = data.get("text", "")
    if not question:
        return jsonify({"error": "Пустой запрос"}), 400
    
    answer, sources = ask(question)
    return jsonify({
        "reply": answer,
        "sources": sources
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

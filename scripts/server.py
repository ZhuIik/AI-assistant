import json
import logging
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.llm.chat import chat_once
from src.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


def generate_answer(text: str, session_id: str | None = None) -> str:
    return chat_once(text, session_id=session_id)


try:
    from flask import Flask, jsonify, request
except ModuleNotFoundError:
    Flask = None


if Flask is not None:
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/chat", methods=["POST"])
    def chat():
        start_time = time.perf_counter()
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        session_id = str(data.get("session_id") or request.remote_addr or "default")
        if not text:
            logger.info("server_chat rejected=no_text session_id=%s", session_id)
            return jsonify({"error": "no text provided"}), 400

        try:
            logger.info("server_chat request session_id=%s text=%r", session_id, text[:500])
            reply = generate_answer(text, session_id=session_id)
        except Exception as e:
            logger.exception("server_chat error session_id=%s", session_id)
            return jsonify({"error": f"internal error: {e}"}), 500

        logger.info(
            "server_chat response session_id=%s elapsed_ms=%d",
            session_id,
            int((time.perf_counter() - start_time) * 1000),
        )
        return jsonify({"reply": reply})


class StdlibHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/chat":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "invalid json"})
            return

        text = (data.get("text") or "").strip()
        session_id = str(data.get("session_id") or self.client_address[0] or "default")
        if not text:
            logger.info("stdlib_chat rejected=no_text session_id=%s", session_id)
            self._send_json(400, {"error": "no text provided"})
            return

        try:
            start_time = time.perf_counter()
            logger.info("stdlib_chat request session_id=%s text=%r", session_id, text[:500])
            self._send_json(200, {"reply": generate_answer(text, session_id=session_id)})
            logger.info(
                "stdlib_chat response session_id=%s elapsed_ms=%d",
                session_id,
                int((time.perf_counter() - start_time) * 1000),
            )
        except Exception as e:
            logger.exception("stdlib_chat error session_id=%s", session_id)
            self._send_json(500, {"error": f"internal error: {e}"})

    def log_message(self, format, *args):
        return


def run_stdlib_server():
    server = HTTPServer(("127.0.0.1", 5000), StdlibHandler)
    print("Server running at http://127.0.0.1:5000")
    server.serve_forever()


if __name__ == "__main__":
    if Flask is not None:
        app.run(host="127.0.0.1", port=5000)
    else:
        run_stdlib_server()

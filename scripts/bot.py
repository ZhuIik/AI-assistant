import asyncio
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv(path=None):
        if path is None:
            path = Path(__file__).resolve().parents[1] / ".env"
        path = Path(path)
        if not path.exists():
            return False
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return True


load_dotenv()

from src.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:5000/chat")


def ask_server(text: str, session_id: str | int | None = None) -> str:
    payload = json.dumps(
        {"text": text, "session_id": str(session_id) if session_id is not None else None},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("reply") or f"В ответе сервера нет поля reply: {data}"


try:
    import requests
    from aiogram import Bot, Dispatcher, F
    from aiogram.filters import CommandStart
    from aiogram.types import Message
except ModuleNotFoundError:
    requests = None
    Bot = None
    Dispatcher = None
    F = None
    CommandStart = None
    Message = None


if Dispatcher is not None:
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def cmd_start(message: Message):
        await message.answer(
            "Привет! Я помощник по лекциям. Задавай вопрос, а я отвечу по материалам курса и добавлю таймкоды."
        )

    @dp.message(F.text)
    async def handle_text(message: Message):
        user_text = message.text.strip()
        if not user_text:
            return

        await message.answer("Думаю...")

        try:
            logger.info("bot_aiogram request chat_id=%s text=%r", message.chat.id, user_text[:500])
            resp = requests.post(
                SERVER_URL,
                json={"text": user_text, "session_id": str(message.chat.id)},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            reply = data.get("reply")
        except Exception as e:
            logger.exception("bot_aiogram error chat_id=%s", message.chat.id)
            await message.answer(f"Не могу получить ответ от сервера: {e}")
            return

        await message.answer(reply or f"В ответе сервера нет поля reply: {data}")

    async def main_aiogram():
        if not BOT_TOKEN:
            raise RuntimeError("Установите BOT_TOKEN или TELEGRAM_BOT_TOKEN.")
        bot = Bot(BOT_TOKEN)
        await dp.start_polling(bot)


def telegram_api(method: str, payload: dict | None = None) -> dict:
    if not BOT_TOKEN:
        raise RuntimeError("Установите BOT_TOKEN или TELEGRAM_BOT_TOKEN.")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=130) as response:
        return json.loads(response.read().decode("utf-8"))


def send_message(chat_id: int, text: str):
    telegram_api("sendMessage", {"chat_id": chat_id, "text": text[:4000]})


def main_stdlib():
    offset = 0
    print("Telegram bot polling started.")
    while True:
        try:
            updates = telegram_api(
                "getUpdates",
                {"offset": offset, "timeout": 60, "allowed_updates": ["message"]},
            )
            for update in updates.get("result", []):
                offset = max(offset, update["update_id"] + 1)
                message = update.get("message") or {}
                text = (message.get("text") or "").strip()
                chat = message.get("chat") or {}
                chat_id = chat.get("id")
                if not text or chat_id is None:
                    continue
                if text.startswith("/start"):
                    send_message(chat_id, "Привет! Задай вопрос по лекциям, я отвечу с таймкодами.")
                    continue
                send_message(chat_id, "Думаю...")
                logger.info("bot_stdlib request chat_id=%s text=%r", chat_id, text[:500])
                send_message(chat_id, ask_server(text, session_id=chat_id))
        except urllib.error.URLError as e:
            logger.warning("bot_stdlib network_error=%s", e)
            time.sleep(5)
        except Exception as e:
            logger.exception("bot_stdlib error=%s", e)
            time.sleep(5)


if __name__ == "__main__":
    if Dispatcher is not None:
        asyncio.run(main_aiogram())
    else:
        main_stdlib()

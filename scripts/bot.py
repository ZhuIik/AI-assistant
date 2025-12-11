
import asyncio
import logging
import os

import requests
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message

BOT_TOKEN = "7680460650:AAFTgS-qYMKdxaetZdzE0X6basEJujKY3qk"

SERVER_URL = "http://127.0.0.1:5000/chat"

bot = Bot(BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я твой помощник по Практическому предпринимательству\n"
        "Задавай свои вопрос — и я отвечу на них."
    )


@dp.message(F.text)
async def handle_text(message: Message):
    user_text = message.text.strip()
    if not user_text:
        return

    await message.answer("💬 Думаю...")

    try:
        resp = requests.post(
            SERVER_URL,
            json={"text": user_text},
            timeout=120,
        )
    except Exception as e:
        await message.answer(f"⚠️ Не могу достучаться до сервера: {e}")
        return

    if resp.status_code != 200:
        await message.answer(f"⚠️ Сервер вернул статус {resp.status_code}: {resp.text}")
        return

    try:
        data = resp.json()
    except Exception as e:
        await message.answer(f"⚠️ Не смог прочитать JSON от сервера: {e}\nТело: {resp.text}")
        return

    reply = data.get("reply")
    if not reply:
        await message.answer(f"⚠️ В ответе сервера нет поля 'reply': {data}")
        return

    await message.answer(reply)


async def main():
    logging.basicConfig(level=logging.INFO)
    if not BOT_TOKEN:
        raise RuntimeError("Не указан токен бота. Вставь его в BOT_TOKEN или через переменную окружения.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

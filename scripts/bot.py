import asyncio
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

TOKEN = "7680460650:AAFTgS-qYMKdxaetZdzE0X6basEJujKY3qk"
REGCHAT_URL = "http://127.0.0.1:5000/message"

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

@dp.message(CommandStart())
async def start_command(message: types.Message):
    await message.answer("Привет! Я подключён к твоему RAG-чату 🤖")

@dp.message()
async def handle_message(message: types.Message):
    user_input = message.text
    await message.answer("💬 Думаю...")

    try:
        response = requests.post(REGCHAT_URL, json={"text": user_input}, timeout=60)
        data = response.json()
        reply = data.get("reply", "Нет ответа.")
        sources = ", ".join(data.get("sources", []))
        text = f"{reply}\n\n📚 Источники: {sources}"
    except Exception as e:
        text = f"⚠️ Ошибка: {e}"

    await message.answer(text)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

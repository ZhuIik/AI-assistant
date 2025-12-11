AI-ASSISTANT — Local Retrieval-Augmented Generation System

Проект предназначен для автономной интеллектуальной обработки учебных материалов. Система принимает аудио, транскрибирует его, очищает текст, формирует векторную базу знаний и отвечает на вопросы, комбинируя retrieved-контекст и локальную LLM.

Цель — создать полностью офлайн-инструмент, который может обучаться на лекциях курса и использоваться студентами/преподавателями через любой интерфейс.

1. Ключевые возможности
1.1. Транскрибация

Модуль (src/audio, scripts/OpenAI-Whisper.py) преобразует аудио/видео в текст:

Faster-Whisper

автоматическое преобразование в WAV (convert_to_wav.py)

сохранение результатов в data/transcripts

1.2. Очистка и нормализация текста

scripts/clean_transcripts.py + src/text_pipeline:

удаление шумов речи, служебных фраз

корректировка форматирования

подготовка текста к эмбеддингам

Результат → data/cleaned.

1.3. Построение базы знаний

scripts/embed_kb.py + src/rag:

создание эмбеддингов Sentence-Transformers

разбиение текста на чанки

сохранение индекса в data/datasets

Поддерживаются:

FAISS

ChromaDB

1.4. Retrieval-Augmented Generation

Основной модуль (src/rag):

Преобразует вопрос в embedding.

Находит K ближайших фрагментов.

Строит контекст.

Передаёт его в модель генерации.

Если контекст отсутствует:
модель формирует полезный ответ без ссылок на конкретные куски лекций.

1.5. Модели LLM

Модуль src/llm поддерживает:

Ollama (Gemma-2B/Gemma-3-2B)

HuggingFace модели (опционально)

1.6. API и клиенты
Локальный API

scripts/server.py + src/api
Предоставляет REST-интерфейс:

POST /ask
{
  "query": "Вопрос пользователя"
}

Telegram (демонстрационный клиент)

scripts/bot.py
Подключается к API и предоставляет удобный чат-интерфейс.

Позже можно заменить на Web UI, desktop или CLI.

2. Структура проекта
AI-ASSISTANT/
│
├── data/
│   ├── cleaned/           # очищенные транскрипты
│   ├── datasets/          # векторные индексы
│   ├── outputs/           # ответы/результаты
│   ├── raw/               # исходные аудио/видео
│   └── transcripts/       # результаты whisper
│
├── legacy/                # старые версии скриптов
│
├── scripts/
│   ├── bot.py
│   ├── chat_local.py
│   ├── clean_transcripts.py
│   ├── convert_to_wav.py
│   ├── embed_kb.py
│   ├── finetune_lora.py
│   ├── OpenAI-Whisper.py
│   └── server.py          # локальный API
│
├── src/
│   ├── api/               # серверные endpoint’ы
│   ├── audio/             # whisper + конвертация
│   ├── llm/               # работа с моделями LLM
│   ├── rag/               # retrieval + generation
│   ├── text_pipeline/     # чистка текста
│   └── config.py
│
├── README.md
└── requirements.txt

3. Быстрый старт
Установка
pip install -r requirements.txt

1. Конвертация аудио
python scripts/convert_to_wav.py

2. Транскрибация
python scripts/OpenAI-Whisper.py

3. Очистка текста
python scripts/clean_transcripts.py

4. Создание эмбеддингов
python scripts/embed_kb.py

5. Запуск сервера
python scripts/server.py

6. Запуск клиента (Telegram или другой)
python scripts/bot.py

4. Минимальные системные требования

GPU: 6–8 GB VRAM
RAM: 16 GB
OS: Windows / Linux
Python: 3.9+
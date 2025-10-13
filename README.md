🧩 AI-Assistant

AI-Assistant is an experimental project that integrates local document retrieval, speech recognition, and OpenAI-based chat capabilities.
It uses a Retrieval-Augmented Generation (RAG) architecture, combining large language model reasoning with a structured local knowledge base.

🧠 Overview

The system allows you to:

Build and update a local knowledge base from text data

Generate vector embeddings for fast semantic search

Transcribe audio using OpenAI Whisper

Summarize large text sources

Run an interactive RAG chat that answers using both local and external knowledge

📁 Project Structure
AI-ASSISTANT/
├── .venv/                  # Virtual environment (ignored)
├── data/
│   ├── knowledge_base/     # Source knowledge base and JSONL file
│   ├── raw/                # Raw input data
│   ├── summaries/          # Summarized text data
│   └── transcripts/        # Transcribed audio files
├── embeddings/             # FAISS index and metadata
├── logs/                   # Runtime logs
├── models/                 # Local models (ignored)
├── notebooks/              # Jupyter notebooks and experiments
├── scripts/
│   ├── build_kb.py         # Build structured knowledge base
│   ├── embed_kb.py         # Create FAISS embeddings
│   ├── OpenAI-Whisper.py   # Convert audio to text
│   ├── rag_chat.py         # RAG chat interface
│   ├── summarize.py        # Summarize large text inputs
│   └── test.py             # Testing utilities
└── requirements.txt

⚙️ Installation

Clone the repository:

git clone https://github.com/ZhuIik/AI-Assistant.git
cd AI-Assistant


Create and activate a virtual environment:

python -m venv .venv
# For Windows
.venv\Scripts\activate
# For macOS/Linux
source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt

🚀 Usage

1. Build the knowledge base

python scripts/build_kb.py


2. Generate vector embeddings

python scripts/embed_kb.py


3. Start the RAG chat

python scripts/rag_chat.py


4. (Optional) Transcribe audio

python scripts/OpenAI-Whisper.py


5. (Optional) Summarize text

python scripts/summarize.py

🧰 Technologies Used

🐍 Python 3.10+

🧮 FAISS — semantic vector search

🎧 OpenAI Whisper — speech-to-text

🧠 LangChain / OpenAI API — LLM integration

📊 NumPy / pandas — data processing and analysis

👤 Author

Timofey Gorbatenkov
📍 UrFU
📧 reincon19@example.com
from pathlib import Path

# Корень проекта: Ai-assistant/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Модель по умолчанию
BASE_MODEL = "google/gemma-2-2b-it"

# Относительный путь к LoRA адаптеру
LORA_REL = "data/outputs/gemma_lectures_lora_v1/checkpoint-3"

# Абсолютный путь к адаптеру
LORA_PATH = (PROJECT_ROOT / LORA_REL).resolve()

from pathlib import Path
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.config import BASE_MODEL, LORA_PATH


def load_pipeline():
    print("🔹 Загружаем базовую модель Gemma...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print(f"🔹 Проверяем LoRA адаптер по пути:\n{LORA_PATH}")

    adapter_config = LORA_PATH / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"\n❌ Не найден adapter_config.json!\n"
            f"Проверенный путь: {adapter_config}\n"
            f"Проверь LORA_PATH в config.py"
        )

    print("🔹 Подключаем LoRA адаптер...")
    model = PeftModel.from_pretrained(model, str(LORA_PATH))

    print("🔹 Собираем пайплайн...")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print("✅ Модель с LoRA готова!")
    return pipe

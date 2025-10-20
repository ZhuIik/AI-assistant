# scripts/finetune_lora.py
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# === 1. Пути и параметры ===
DATA_PATH = "data/datasets/lectures_v1.jsonl"
OUTPUT_DIR = "outputs/gemma_lectures_lora_v1"
MODEL_NAME = "google/gemma-2-2b-it"

# === 2. Защита от .to() ошибок (патч для bitsandbytes) ===
old_to = nn.Module.to
def safe_to(self, *args, **kwargs):
    if "bitsandbytes" in str(type(self)).lower():
        return self
    return old_to(self, *args, **kwargs)
nn.Module.to = safe_to

# === 3. Загружаем датасет ===
dataset = load_dataset("json", data_files=DATA_PATH)

def formatting_func(example):
    text = f"User: {example['instruction']}\nAssistant: {example['output']}"
    return [text]  # важно возвращать список!

# === 4. Токенайзер ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# === 5. Квантизация модели (QLoRA, 4-bit) ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16   # ← было bfloat16
)


print("🔹 Загружаем Gemma 2 9B в 4-битном режиме...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16           
)


# === 6. Настройка LoRA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# === 7. Аргументы обучения ===
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,                  # ← пониже lr, чтобы убрать нестабильность
    fp16=True,                           # ← включаем fp16
    bf16=False,                          # ← выключаем bf16
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
)


# === 8. Trainer ===
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=args,
    max_seq_length=768,
    formatting_func=formatting_func,
    packing=False,
)

# === 9. Запуск ===
if __name__ == "__main__":
    print("🚀 Начинаем обучение Gemma 2 LoRA (4-bit)...")
    trainer.train()
    print("✅ Обучение завершено! Результаты сохранены в:", OUTPUT_DIR)

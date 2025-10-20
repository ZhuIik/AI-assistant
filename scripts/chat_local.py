# scripts/chat_local.py
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE = "google/gemma-2-2b-it"                 # или "google/gemma-2-9b-it"
ADAPTER = "outputs/gemma_lectures_lora_v1/checkpoint-3"  # путь к твоим обученным весам

print("🔹 Загружаем модель...")
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, ADAPTER)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16
)

print("✅ Модель готова! Можешь задавать вопросы:\n")

while True:
    prompt = input("❓ Вопрос: ")
    if prompt.lower() in ["exit", "quit", "stop"]:
        print("🛑 Завершено.")
        break
    output = pipe(prompt, max_new_tokens=250, do_sample=True, temperature=0.7)[0]["generated_text"]
    print("\n💬 Ответ модели:\n", output)
    print("-" * 80)

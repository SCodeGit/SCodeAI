import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "codellama/CodeLlama-7b-Python-hf"
LORA_PATH = "./outputs/scode_ai/checkpoint-2000"
EXPORT_PATH = "./models/scode_ai_safetensors"

print("=== Step 3: Merging LoRA Adapter into Full Model ===")

print("Loading base model in bfloat16...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="cpu"
)

print("Loading adapter weights...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving standalone safetensors model to {EXPORT_PATH}...")
model.save_pretrained(EXPORT_PATH, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
tokenizer.save_pretrained(EXPORT_PATH)

print(f"✅ Step 3 Complete: Full SCode AI model saved to {EXPORT_PATH}")

import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = "./outputs/scode_ai/checkpoint-2000"
BASE_MODEL = "codellama/CodeLlama-7b-Python-hf"

print("=== Step 2: Verification Checklist ===")

# 1. Check required output files
required_files = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json"
]

missing = []
for f in required_files:
    path = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(path):
        print(f"[OK] Found {f}")
    else:
        print(f"[FAIL] Missing {f}")
        missing.append(f)

if missing:
    print(f"\nVerification FAILED: Missing files in {OUTPUT_DIR}")
    exit(1)

# 2. Test loading adapter on base model
print("\n=== Testing Model Loading ===")
try:
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="cpu"
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    print("[OK] LoRA Adapter and Tokenizer loaded successfully!")
    print("\n✅ Step 2 Complete: Model verified and ready for Step 3 (Merge).")
except Exception as e:
    print(f"[FAIL] Load test failed with error: {e}")

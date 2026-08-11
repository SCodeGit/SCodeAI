import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ============================================================
# SCode AI - 10 Step Training Diagnostic
# ============================================================

MODEL_NAME = "codellama/CodeLlama-7b-Python-hf"
DATASET_PATH = "./datasets/combined_scode_dataset.jsonl"
OUTPUT_DIR = "./outputs/scode_ai_test"

MAX_SEQ_LENGTH = 4096

print("=" * 70)
print("SCode AI TRAINING DIAGNOSTIC")
print("=" * 70)

# ------------------------------------------------------------
# 1. Tokenizer
# ------------------------------------------------------------

print("\n=== Loading Tokenizer ===")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer:", tokenizer.name_or_path)
print("EOS token:", repr(tokenizer.eos_token))
print("EOS ID:", tokenizer.eos_token_id)
print("PAD token:", repr(tokenizer.pad_token))
print("PAD ID:", tokenizer.pad_token_id)

# ------------------------------------------------------------
# 2. Model
# ------------------------------------------------------------

print("\n=== Loading CodeLlama in BF16 ===")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False

print("Model dtype:", next(model.parameters()).dtype)
print("Model device:", next(model.parameters()).device)

# ------------------------------------------------------------
# 3. LoRA
# ------------------------------------------------------------

print("\n=== Configuring LoRA ===")

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

# ------------------------------------------------------------
# 4. Dataset
# ------------------------------------------------------------

print("\n=== Loading Dataset ===")

full_dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)

print("Total dataset examples:", len(full_dataset))

# IMPORTANT:
# Only use a small subset for this diagnostic.
# The real training will use the complete dataset.
dataset = full_dataset.select(range(min(1000, len(full_dataset))))

print("Diagnostic examples:", len(dataset))

# ------------------------------------------------------------
# 5. Format dataset
# ------------------------------------------------------------

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

{}

### Input:

{}

### Response:

{}"""

def format_prompts(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    texts = []

    for instruction, input_text, output in zip(
        instructions,
        inputs,
        outputs
    ):
        text = alpaca_prompt.format(
            instruction,
            input_text,
            output
        ) + tokenizer.eos_token

        texts.append(text)

    return {"text": texts}

dataset = dataset.map(
    format_prompts,
    batched=True,
    num_proc=1
)

print("Formatted columns:", dataset.column_names)

# ------------------------------------------------------------
# 6. Check first formatted example
# ------------------------------------------------------------

print("\n=== FORMATTED EXAMPLE CHECK ===")

example = dataset[0]["text"]

print("Characters:", len(example))

tokens = tokenizer(
    example,
    truncation=True,
    max_length=MAX_SEQ_LENGTH
)

print("Tokens:", len(tokens["input_ids"]))

if len(tokens["input_ids"]) == 0:
    raise RuntimeError("❌ Tokenizer produced ZERO tokens.")

print("First token IDs:", tokens["input_ids"][:20])
print("Last token IDs:", tokens["input_ids"][-20:])

# ------------------------------------------------------------
# 7. Training configuration
# ------------------------------------------------------------

print("\n=== Configuring SFTTrainer ===")

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,

    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH,

    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,

    # ONLY 10 optimizer steps for diagnostic
    max_steps=10,

    warmup_steps=2,

    learning_rate=1e-4,

    bf16=True,
    fp16=False,

    # Prevent exploding gradients
    max_grad_norm=1.0,

    logging_steps=1,

    optim="adamw_torch",

    weight_decay=0.01,

    lr_scheduler_type="linear",

    seed=3407,

    save_strategy="no",

    report_to="none",

    dataset_num_proc=1,

    packing=False,

    gradient_checkpointing=False,
)

# ------------------------------------------------------------
# 8. Trainer
# ------------------------------------------------------------

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=sft_config,
)

# ------------------------------------------------------------
# 9. Start test
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STARTING 10-STEP DIAGNOSTIC TRAINING")
print("=" * 70)

result = trainer.train()

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

print("\nTraining result:")
print(result)

# ------------------------------------------------------------
# 10. Save diagnostic adapter
# ------------------------------------------------------------

print("\n=== Saving Diagnostic Adapter ===")

os.makedirs(OUTPUT_DIR, exist_ok=True)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Saved to:", OUTPUT_DIR)

print("\n=== SCode AI Diagnostic Finished ===")
PY

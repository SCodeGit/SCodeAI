import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


# ============================================================
# SCode AI - LoRA Fine-Tuning
# ============================================================

MODEL_NAME = "codellama/CodeLlama-7b-Python-hf"

DATASET_PATH = "./datasets/combined_scode_dataset.jsonl"

OUTPUT_DIR = "./outputs/scode_ai"

MAX_SEQ_LENGTH = 2048


# ============================================================
# 1. Load Tokenizer
# ============================================================

print("=== Loading Tokenizer ===")

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


# ============================================================
# 2. Load Base Model
# ============================================================

print("\n=== Loading CodeLlama in BF16 ===")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Required when using gradient checkpointing
model.config.use_cache = False

print("Model dtype:", next(model.parameters()).dtype)
print("Model device:", next(model.parameters()).device)


# ============================================================
# 3. Configure LoRA
# ============================================================

print("\n=== Configuring LoRA Adapters ===")

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


# ============================================================
# 4. Prompt Formatting
# ============================================================

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


# ============================================================
# 5. Load FULL Dataset
# ============================================================

print("\n=== Loading FULL SCode Dataset ===")

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)

print("Total training examples:", len(dataset))

dataset = dataset.map(
    format_prompts,
    batched=True,
    num_proc=1
)

print("Dataset formatting complete.")


# ============================================================
# 6. Training Configuration
# ============================================================

print("\n=== Configuring SFT Training ===")

sft_config = SFTConfig(

    output_dir=OUTPUT_DIR,

    dataset_text_field="text",

    max_length=MAX_SEQ_LENGTH,

    # --------------------------------------------------------
    # GPU memory-safe configuration
    # --------------------------------------------------------

    per_device_train_batch_size=1,

    gradient_accumulation_steps=16,

    # --------------------------------------------------------
    # EXACTLY 3,000 optimizer steps
    # --------------------------------------------------------

    max_steps=3000,

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    learning_rate=1e-4,

    warmup_steps=100,

    # --------------------------------------------------------
    # AMD MI300X BF16
    # --------------------------------------------------------

    bf16=True,

    fp16=False,

    # --------------------------------------------------------
    # Gradient stability
    # --------------------------------------------------------

    max_grad_norm=1.0,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_steps=10,

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optim="adamw_torch",

    weight_decay=0.01,

    lr_scheduler_type="linear",

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=3407,

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    save_strategy="steps",

    save_steps=500,

    save_total_limit=3,

    # --------------------------------------------------------
    # No external reporting
    # --------------------------------------------------------

    report_to="none",

    # --------------------------------------------------------
    # Dataset processing
    # --------------------------------------------------------

    dataset_num_proc=1,

    packing=False,

    # --------------------------------------------------------
    # Memory optimization
    # --------------------------------------------------------

    gradient_checkpointing=True,
)


# ============================================================
# 7. Create Trainer
# ============================================================

print("\n=== Creating SFTTrainer ===")

trainer = SFTTrainer(

    model=model,

    train_dataset=dataset,

    processing_class=tokenizer,

    args=sft_config,
)


# ============================================================
# 8. Start Training
# ============================================================

print("\n")
print("=" * 70)
print("STARTING SCODE AI FINE-TUNING")
print("=" * 70)
print()
print("Dataset examples:", len(dataset))
print("Training steps:", 3000)
print("Batch size:", 1)
print("Gradient accumulation:", 16)
print("Effective batch size:", 16)
print("Maximum sequence length:", MAX_SEQ_LENGTH)
print("Learning rate:", 1e-4)
print("Precision: BF16")
print()
print("=" * 70)


trainer.train()


# ============================================================
# 9. Save Final LoRA Adapter
# ============================================================

print("\n=== Fine-Tuning Complete ===")

os.makedirs(OUTPUT_DIR, exist_ok=True)

model.save_pretrained(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print()
print("=" * 70)
print("SCODE AI TRAINING COMPLETE")
print("=" * 70)
print()
print("LoRA adapter saved to:")
print(OUTPUT_DIR)
print()

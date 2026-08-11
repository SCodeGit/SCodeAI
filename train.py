import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

max_seq_length = 2048
dtype = None # Auto detection (bfloat16 for modern GPUs)
load_in_4bit = True # Set to False if you want float16/bfloat16 fine-tuning

MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct"

print(f"Loading Unsloth model {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Apply PEFT / LoRA target parameters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Unsloth optimized at 0
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 30% less VRAM consumption
    random_state = 3407,
)

# Load your local dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files="my_dataset/data.json", split="train")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

print("Starting Unsloth training...")
trainer_stats = trainer.train()

print("Saving LoRA adapter...")
model.save_pretrained("scode_qwen_adapter")
tokenizer.save_pretrained("scode_qwen_adapter")
print("Training complete!")

"""
Stage 2 v2: Function Calling SFT
Uses Stage 1 v2 checkpoint + smaller FC dataset

Usage: python scripts/train_stage2_v2.py
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# ============== CONFIG ==============
MODEL_NAME = "./checkpoints/stage1_v2/merged"  # From Stage 1 v2
DATASET_NAME = "shekkari21/qwen-fc-sft-data-v2"
OUTPUT_DIR = "./checkpoints/stage2_v2"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 1e-4  # Lower LR for stage 2
NUM_EPOCHS = 1
LORA_R = 16
LORA_ALPHA = 16
# ====================================

print("="*60)
print("STAGE 2 v2: FUNCTION CALLING SFT")
print("="*60)

# Load Stage 1 model
print(f"\nLoading Stage 1 model from {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Add LoRA adapters (fresh for stage 2)
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Load dataset
print("\nLoading function calling dataset...")
dataset = load_dataset(DATASET_NAME, split="stage2_fc")
print(f"Loaded {len(dataset)} examples")

# Format function
def format_chat(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

# Apply formatting
dataset = dataset.map(format_chat, batched=True, remove_columns=dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_steps=50,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    optim="adamw_8bit",
    seed=42,
    report_to="none",
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

# Print training info
print(f"\n{'='*60}")
print("TRAINING CONFIG")
print(f"{'='*60}")
print(f"Base Model:     {MODEL_NAME}")
print(f"Dataset:        {DATASET_NAME}")
print(f"LoRA Rank:      {LORA_R}")
print(f"Batch Size:     {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"Learning Rate:  {LEARNING_RATE}")
print(f"Epochs:         {NUM_EPOCHS}")
print(f"Dataset Size:   {len(dataset)}")
print(f"Output:         {OUTPUT_DIR}")
print(f"{'='*60}\n")

# Train!
print("Starting training...")
trainer.train()

# Save
print("\nSaving model...")
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print("Saving merged model...")
model.save_pretrained_merged(f"{OUTPUT_DIR}/merged", tokenizer, save_method="merged_16bit")

print("\n" + "="*60)
print("STAGE 2 v2 COMPLETE!")
print(f"Final model saved to: {OUTPUT_DIR}/merged")
print("="*60)

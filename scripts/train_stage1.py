"""
Stage 1: Chat SFT Training
Trains model on short Q&A pairs to learn stopping behavior

Usage:
1. python scripts/create_minimal_data.py
2. python scripts/train_stage1.py
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from datasets import load_dataset
import torch


class SampleGenerationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, every_n_steps=10):
        self.model = model
        self.tokenizer = tokenizer
        self.every_n_steps = every_n_steps
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.test_prompts = [
            "What is 2 + 2?",
            "What is the capital of France?",
            "What color is the sky?",
        ]

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0 and state.global_step > 0:
            print(f"\n{'='*60}")
            print(f"Step {state.global_step}")
            print(f"{'='*60}")

            for prompt in self.test_prompts:
                messages = [{"role": "user", "content": prompt}]
                inputs = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(self.model.device)
                input_len = inputs.shape[1]

                self.model.eval()
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.im_end_id,
                    )
                self.model.train()

                generated = outputs[0][input_len:]
                response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                has_eos = self.im_end_id in generated.tolist()

                print(f"Q: {prompt}")
                print(f"A: {response} [{'STOPPED' if has_eos else 'NO STOP'}]")

            print(f"{'='*60}\n")
        return control


# Config
MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = "./checkpoints/stage1_minimal"
MAX_SEQ_LENGTH = 512  # Short sequences
BATCH_SIZE = 4
GRAD_ACCUM = 1  # Small dataset, no need
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3  # Multiple epochs on small data

print("="*60)
print("MINIMAL TRAINING: 200 short Q&A pairs")
print("="*60)

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Load minimal dataset
print("\nLoading minimal dataset...")
dataset = load_dataset("json", data_files="data/minimal/train.jsonl", split="train")
print(f"Loaded {len(dataset)} examples")

def format_chat(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_chat, batched=True, remove_columns=dataset.column_names)

# ============================================================
# VERIFY: Show that chat template is applied correctly
# ============================================================
print("\n" + "="*60)
print("CHAT TEMPLATE VERIFICATION")
print("="*60)
print("Sample formatted text (first example):")
print(repr(dataset[0]["text"]))
print("\nLook for: <|im_start|>, <|im_end|> tokens")
print("="*60 + "\n")

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_steps=10,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=5,
    save_steps=100,
    bf16=True,
    optim="adamw_8bit",
    seed=42,
    report_to="none",
)

callback = SampleGenerationCallback(model, tokenizer, every_n_steps=10)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
    callbacks=[callback],
)

# Apply label masking
print("Applying label masking...")
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

print(f"\nTraining on {len(dataset)} examples for {NUM_EPOCHS} epochs")
print(f"Total steps: ~{len(dataset) * NUM_EPOCHS // BATCH_SIZE}")

trainer.train()

# Save
print("\nSaving...")
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
model.save_pretrained_merged(f"{OUTPUT_DIR}/merged", tokenizer, save_method="merged_16bit")

print("\nDone! Test with: python scripts/test_inference.py")

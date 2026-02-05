"""
Verify label masking on original dataset
Shows token-by-token breakdown for first 5 examples
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42,
)

# Load original dataset - first 5 examples
print("\nLoading original dataset (first 5 examples)...")
dataset = load_dataset("shekkari21/qwen-fc-sft-data-v2", split="stage1_chat[:5]")

def format_chat(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_chat, batched=True, remove_columns=dataset.column_names)

# Create trainer
training_args = TrainingArguments(
    output_dir="./tmp", per_device_train_batch_size=1,
    num_train_epochs=1, report_to="none",
)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    dataset_text_field="text", max_seq_length=512, args=training_args,
)

# Apply masking
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# Get dataloader
dataloader = trainer.get_train_dataloader()

print("\n" + "="*70)
print("LABEL MASKING VERIFICATION - ORIGINAL DATASET")
print("="*70)

for idx, batch in enumerate(dataloader):
    if idx >= 5:
        break

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    print(f"\n{'='*70}")
    print(f"EXAMPLE {idx + 1}")
    print(f"{'='*70}")

    # Show the raw text first
    text = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"\nFormatted text:\n{text[:500]}...")

    print(f"\n{'Token':<25} {'Label':<10} {'Training?'}")
    print("-" * 50)

    # Find where assistant response starts
    in_assistant = False
    for i in range(len(input_ids)):
        token_id = input_ids[i].item()
        label = labels[i].item()
        token_text = tokenizer.decode([token_id]).replace('\n', '\\n')[:20]

        if token_text == "assistant":
            in_assistant = True

        # Only show around the transition (last 5 user + first 10 assistant)
        if label != -100 or (i > 0 and labels[i-1].item() == -100 and i < len(labels)-1):
            training = "NO (-100)" if label == -100 else f"YES ({label})"
            marker = " ←" if label != -100 else ""
            print(f"{token_text:<25} {label:<10} {training}{marker}")

    print("-" * 50)
    masked = (labels == -100).sum().item()
    trained = (labels != -100).sum().item()
    print(f"Masked (user/system): {masked} tokens")
    print(f"Trained (assistant):  {trained} tokens")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)

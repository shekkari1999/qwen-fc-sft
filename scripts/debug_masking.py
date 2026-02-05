"""Debug label masking - see exactly which tokens are trained vs masked."""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch
import argparse

def debug_masking(data_path):
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B",
        max_seq_length=2048,
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

    # Load just 1 example
    if data_path.startswith("hf://"):
        parts = data_path[5:].split(":")
        repo_id, split_name = parts[0], parts[1] if len(parts) > 1 else "stage1_chat"
        dataset = load_dataset(repo_id, split=f"{split_name}[:1]")
    else:
        dataset = load_dataset("json", data_files=data_path, split="train[:1]")

    def format_chat(examples):
        texts = []
        for m in examples["messages"]:
            text = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            # Remove trailing newline so model learns to stop at <|im_end|>
            text = text.rstrip('\n')
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_chat, batched=True, remove_columns=dataset.column_names)

    print("\n" + "="*60)
    print("FORMATTED TEXT:")
    print("="*60)
    print(dataset[0]["text"])
    print("="*60)

    # Create trainer
    training_args = TrainingArguments(
        output_dir="./debug",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        dataset_text_field="text", max_seq_length=2048, args=training_args,
    )

    # Apply masking
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Get batch and analyze
    batch = next(iter(trainer.get_train_dataloader()))
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    print("\n" + "="*60)
    print("TOKEN-BY-TOKEN ANALYSIS:")
    print("="*60)
    print(f"{'Pos':>4} | {'Token ID':>8} | {'Label':>8} | {'Train?':>6} | Token")
    print("-"*60)

    trained_tokens = []
    masked_tokens = []

    for i, (tok_id, label) in enumerate(zip(input_ids, labels)):
        tok_id = tok_id.item()
        label = label.item()
        token = tokenizer.decode([tok_id])
        is_trained = label != -100

        if is_trained:
            trained_tokens.append(token)
        else:
            masked_tokens.append(token)

        # Show key boundaries
        marker = ""
        if "<|im_start|>" in token:
            marker = " <-- START"
        elif "<|im_end|>" in token:
            marker = " <-- END"

        status = "TRAIN" if is_trained else "mask"
        print(f"{i:4d} | {tok_id:8d} | {label:8d} | {status:>6} | {repr(token)}{marker}")

    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Total tokens: {len(input_ids)}")
    print(f"Masked (-100): {len(masked_tokens)}")
    print(f"Trained: {len(trained_tokens)}")
    print(f"Ratio trained: {len(trained_tokens)/len(input_ids)*100:.1f}%")

    print("\n" + "="*60)
    print("TRAINED TOKENS (what the model learns to predict):")
    print("="*60)
    print("".join(trained_tokens))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="hf://shekkari21/qwen-fc-sft-data-v2:stage1_chat")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None, help="Shortcut: --stage 1 or --stage 2")
    args = parser.parse_args()

    if args.stage:
        args.data = f"hf://shekkari21/qwen-fc-sft-data-v2:stage{args.stage}_{'chat' if args.stage == 1 else 'fc'}"

    debug_masking(args.data)

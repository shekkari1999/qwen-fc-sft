"""
Train model: Stage 1 (Chat) or Stage 2 (Function Calling)

Usage (local data):
  python scripts/train.py --stage 1
  python scripts/train.py --stage 2

Usage (HuggingFace Hub):
  python scripts/train.py --stage 1 --data hf://shekkari21/qwen-fc-sft-data-v2:stage1_chat
  python scripts/train.py --stage 2 --data hf://shekkari21/qwen-fc-sft-data-v2:stage2_fc
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from datasets import load_dataset
import torch
import argparse


class GenerationCallback(TrainerCallback):
    """Monitor training with sample generations - interactive mode"""
    def __init__(self, model, tokenizer, every_n_steps=50):
        self.model = model
        self.tokenizer = tokenizer
        self.every_n_steps = every_n_steps
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.default_prompts = ["What is 2 + 2?", "What is the capital of France?"]

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0 and state.global_step > 0:
            print(f"\n{'='*50} Step {state.global_step} {'='*50}")

            # Ask user for custom question (or press Enter to skip)
            try:
                user_q = input("Enter a test question (or press Enter for defaults): ").strip()
                if user_q:
                    prompts = [user_q] + self.default_prompts[:1]  # User question + 1 default
                else:
                    prompts = self.default_prompts
            except EOFError:
                prompts = self.default_prompts

            for prompt in prompts:
                inputs = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(self.model.device)

                self.model.eval()
                with torch.no_grad():
                    out = self.model.generate(
                        input_ids=inputs, max_new_tokens=50, do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.im_end_id,
                    )
                self.model.train()

                gen = out[0][inputs.shape[1]:]
                resp = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
                stopped = "YES" if self.im_end_id in gen.tolist() else "NO"
                print(f"Q: {prompt}\nA: {resp} [stopped: {stopped}]\n")
        return control


def train(stage, data_path, base_model, output_dir, epochs, lr, batch_size):
    print("="*60)
    print(f"STAGE {stage} TRAINING")
    print("="*60)

    # Load model
    print(f"\nLoading {base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # Add LoRA (include lm_head to learn to OUTPUT special tokens like <|im_end|>)
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                        "lm_head"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    # Load data
    print(f"\nLoading data from {data_path}...")
    if data_path.startswith("hf://"):
        # Load from HuggingFace Hub: hf://username/repo:split
        parts = data_path[5:].split(":")
        repo_id, split_name = parts[0], parts[1] if len(parts) > 1 else "stage1_chat"
        dataset = load_dataset(repo_id, split=split_name)
    else:
        dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"  {len(dataset)} examples")

    def format_chat(examples):
        texts = []
        for m in examples["messages"]:
            text = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            # Remove trailing newline so model learns to stop at <|im_end|>
            text = text.rstrip('\n')
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_chat, batched=True, remove_columns=dataset.column_names)

    # Show sample
    print(f"\nSample formatted text:\n{repr(dataset[0]['text'][:300])}...")

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=10,
        save_steps=500,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        dataset_text_field="text", max_seq_length=2048, args=training_args,
        callbacks=[GenerationCallback(model, tokenizer)],
    )

    # Apply label masking
    print("\nApplying label masking (train on assistant only)...")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Verify masking
    batch = next(iter(trainer.get_train_dataloader()))
    masked = (batch["labels"] == -100).sum().item()
    trained = (batch["labels"] != -100).sum().item()
    print(f"  Masked tokens: {masked}, Trained tokens: {trained}")

    # Train
    print(f"\nStarting training ({epochs} epochs, lr={lr})...")
    trainer.train()

    # Save
    print("\nSaving model...")
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    model.save_pretrained_merged(f"{output_dir}/merged", tokenizer, save_method="merged_16bit")

    print(f"\nDone! Model saved to {output_dir}/merged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--data", default=None, help="Auto-selects based on stage if not provided")
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    # Auto-select data and output based on stage (default to HuggingFace Hub)
    if args.data is None:
        args.data = f"hf://shekkari21/qwen-fc-sft-data-v2:stage{args.stage}_{'chat' if args.stage == 1 else 'fc'}"

    if args.output is None:
        args.output = f"./checkpoints/stage{args.stage}"

    if args.stage == 2:
        args.lr = 1e-4  # Lower LR for stage 2
        if args.base == "Qwen/Qwen2.5-3B":
            args.base = "./checkpoints/stage1/merged"

    train(args.stage, args.data, args.base, args.output, args.epochs, args.lr, args.batch)

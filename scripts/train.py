"""
Train model: Stage 1 (Chat), Stage 2 (Function Calling), or Stage 3 (DPO)

Usage:
  python scripts/train.py --stage 1   # SFT on chat data
  python scripts/train.py --stage 2   # SFT on function calling data
  python scripts/train.py --stage 3   # DPO for tool use preferences

With HuggingFace Hub:
  python scripts/train.py --stage 1 --data hf://shekkari21/qwen-fc-sft-data-v2:stage1_chat
  python scripts/train.py --stage 2 --data hf://shekkari21/qwen-fc-sft-data-v2:stage2_fc
  python scripts/train.py --stage 3 --data hf://shekkari21/qwen-fc-sft-data-v2:stage3_dpo
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, DPOTrainer, DPOConfig
from transformers import TrainingArguments, TrainerCallback
from datasets import load_dataset
import torch
import argparse


class GenerationCallback(TrainerCallback):
    """Monitor training with sample generations"""
    def __init__(self, model, tokenizer, every_n_steps=50, stage=1):
        self.model = model
        self.tokenizer = tokenizer
        self.every_n_steps = every_n_steps
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.stage = stage

        if stage == 1:
            self.prompts = ["What is 2 + 2?", "What is the capital of France?", "Say hello."]
            self.system_msg = None
            self.interactive = False
        else:
            # Stage 2 & 3: Test prompts (DIFFERENT from training data)
            self.all_prompts = [
                # Weather - SHOULD use tool
                ("Is it sunny in Barcelona right now?", "SHOULD: get_weather"),
                ("What's the forecast for Seattle?", "SHOULD: get_weather"),
                # Calculate - SHOULD use tool
                ("What's 1729 divided by 13?", "SHOULD: calculate"),
                ("Compute 88 times 77 please", "SHOULD: calculate"),
                # Search - SHOULD use tool
                ("Find me the latest Tesla stock news", "SHOULD: search_web"),
                ("What movies came out this weekend?", "SHOULD: search_web"),
                # Greetings - should NOT use tool
                ("Hey there! What's going on?", "NO tool"),
                ("Good evening!", "NO tool"),
                # General knowledge - should NOT use tool
                ("Who invented the telephone?", "NO tool"),
                ("Why do leaves change color in fall?", "NO tool"),
            ]
            self.prompts = None  # Will randomly select 3
            self.interactive = True  # Allow typing custom questions
            self.system_msg = """You are a helpful assistant with access to the following tools:

[{"name": "get_weather", "parameters": {"location": {"type": "str"}}},
{"name": "calculate", "parameters": {"expression": {"type": "str"}}},
{"name": "search_web", "parameters": {"query": {"type": "str"}}}]

When you need to use a tool, respond with:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call>

Only use tools when necessary. For greetings, general knowledge, or casual conversation, respond directly."""

    def on_step_end(self, args, state, control, **kwargs):
        import random as rnd
        if state.global_step % self.every_n_steps == 0 and state.global_step > 0:
            print(f"\n{'='*50} Step {state.global_step} {'='*50}")

            # Get prompts to test
            if self.prompts is not None:
                test_prompts = [(p, "") for p in self.prompts]
            else:
                # Randomly select 3 from pool of 10
                test_prompts = rnd.sample(self.all_prompts, 3)

            for prompt, expected in test_prompts:
                self._generate_response(prompt, expected)

            # Interactive mode - allow custom questions
            if getattr(self, 'interactive', False):
                try:
                    user_input = input("\n[Enter custom question or press Enter to skip]: ").strip()
                    if user_input:
                        self._generate_response(user_input, "custom")
                except EOFError:
                    pass  # Non-interactive environment

        return control

    def _generate_response(self, prompt, expected):
        """Generate and display response for a prompt"""
        try:
            if self.system_msg:
                messages = [
                    {"role": "system", "content": self.system_msg},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)

            self.model.eval()
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=inputs, max_new_tokens=100, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.im_end_id,
                )
            self.model.train()

            gen = out[0][inputs.shape[1]:]
            resp = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
            stopped = "YES" if self.im_end_id in gen.tolist() else "NO"
            has_tool = "YES" if "<tool_call>" in resp else "NO"

            exp_str = f" [expected: {expected}]" if expected else ""
            print(f"Q: {prompt}{exp_str}\nA: {resp}\n   [stopped: {stopped}, tool_call: {has_tool}]\n")
        except Exception as e:
            print(f"Q: {prompt}\nA: [Generation skipped - {type(e).__name__}]\n")


def train(stage, data_path, base_model, output_dir, epochs, lr, batch_size, push_to_hub=None):
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

    if stage == 1:
        # Stage 1: Add fresh LoRA adapters to base model
        print("Adding LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model, r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                            "lm_head"],
            lora_alpha=16, lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth", random_state=42,
        )
    else:
        # Stage 2: Continue training existing LoRA adapter on FC data
        # Adapter is already loaded from Stage 1 checkpoint
        print("Continuing training existing LoRA adapter on FC data...")

        # Enable lm_head training (needed for <tool_call> special tokens 151657/151658)
        for name, param in model.named_parameters():
            if 'lm_head' in name:
                param.requires_grad = True
                print(f"  Enabled: {name} ({param.numel():,} params)")

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
        callbacks=[GenerationCallback(model, tokenizer, stage=stage)],
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

    # Push to HuggingFace Hub if specified
    if push_to_hub:
        print(f"\nPushing adapter to HuggingFace Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub, token=True)
        tokenizer.push_to_hub(push_to_hub, token=True)
        print(f"Pushed to: https://huggingface.co/{push_to_hub}")

    print(f"\nDone! Adapter saved to {output_dir}/final")


def train_dpo(data_path, base_model, output_dir, epochs, lr, batch_size, push_to_hub=None):
    """Stage 3: DPO training for tool use preferences"""
    print("="*60)
    print("STAGE 3: DPO TRAINING (Tool Use Preferences)")
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

    # Enable lm_head training for DPO
    print("Enabling lm_head for DPO...")
    for name, param in model.named_parameters():
        if 'lm_head' in name:
            param.requires_grad = True
            print(f"  Enabled: {name} ({param.numel():,} params)")

    # Verify all trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Total trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Load DPO data (already in training-ready format with chat template applied)
    print(f"\nLoading DPO data from {data_path}...")
    if data_path.startswith("hf://"):
        parts = data_path[5:].split(":")
        repo_id, split_name = parts[0], parts[1] if len(parts) > 1 else "stage3_dpo"
        dataset = load_dataset(repo_id, split=split_name)
    else:
        dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"  {len(dataset)} preference pairs")

    # Show sample
    print(f"\nSample DPO pair:")
    print(f"  Prompt: {dataset[0]['prompt'][:200]}...")
    print(f"  Chosen: {dataset[0]['chosen'][:100]}...")
    print(f"  Rejected: {dataset[0]['rejected'][:100]}...")

    # DPO config
    dpo_config = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        beta=0.1,  # DPO temperature - lower = stronger preference learning
        max_length=2048,
        max_prompt_length=1024,
    )

    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use implicit reference model (more memory efficient)
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=[GenerationCallback(model, tokenizer, every_n_steps=25, stage=3)],
    )

    # Train
    print(f"\nStarting DPO training ({epochs} epochs, lr={lr}, beta=0.1)...")
    trainer.train()

    # Save
    print("\nSaving model...")
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    # Push to HuggingFace Hub
    if push_to_hub:
        print(f"\nPushing adapter to HuggingFace Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub, token=True)
        tokenizer.push_to_hub(push_to_hub, token=True)
        print(f"Pushed to: https://huggingface.co/{push_to_hub}")

    print(f"\nDone! DPO adapter saved to {output_dir}/final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--data", default=None, help="Auto-selects based on stage if not provided")
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--push-to-hub", type=str, default=None, help="HuggingFace repo to push adapter")
    args = parser.parse_args()

    # Auto-select data and output based on stage
    if args.data is None:
        if args.stage == 1:
            args.data = "hf://shekkari21/qwen-fc-sft-data-v2:stage1_chat"
        elif args.stage == 2:
            args.data = "hf://shekkari21/qwen-fc-sft-data-v2:stage2_fc"
        else:  # stage 3 - DPO data in separate repo (different schema)
            args.data = "hf://shekkari21/qwen-fc-sft-dpo-data-v2:train"

    if args.output is None:
        args.output = f"./checkpoints/stage{args.stage}"

    # Stage-specific defaults
    if args.stage == 2:
        args.lr = 5e-5  # Lower LR for stage 2
        if args.base == "Qwen/Qwen2.5-3B":
            args.base = "./checkpoints/stage1/final"

    elif args.stage == 3:
        args.lr = 5e-6  # Even lower LR for DPO
        args.epochs = 3  # DPO typically needs more epochs on small data
        if args.base == "Qwen/Qwen2.5-3B":
            args.base = "./checkpoints/stage2/final"

    # Run appropriate training
    if args.stage in [1, 2]:
        train(args.stage, args.data, args.base, args.output, args.epochs, args.lr, args.batch, args.push_to_hub)
    else:
        train_dpo(args.data, args.base, args.output, args.epochs, args.lr, args.batch, args.push_to_hub)

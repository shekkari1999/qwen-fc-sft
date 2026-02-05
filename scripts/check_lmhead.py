"""Check if lm_head LoRA exists in Stage 1 adapter"""
from unsloth import FastLanguageModel
import torch

print("Loading Stage 1 adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./checkpoints/stage1/final",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

print("\n--- All parameters with 'lm_head' in name ---")
for name, param in model.named_parameters():
    if 'lm_head' in name.lower():
        print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

print("\n--- All trainable parameters ---")
trainable = []
for name, param in model.named_parameters():
    if param.requires_grad:
        trainable.append((name, param.numel()))

for name, count in trainable[:20]:
    print(f"  {name}: {count:,}")

print(f"\n... and {len(trainable)-20} more" if len(trainable) > 20 else "")
print(f"\nTotal trainable: {sum(c for _, c in trainable):,}")

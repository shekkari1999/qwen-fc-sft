from datasets import load_dataset
import json
from pathlib import Path

print("="*50)
print("STAGE 1: Downloading Chat Dataset")
print("="*50)

# Download UltraChat 200k (high quality conversations)
print("\nDownloading HuggingFaceH4/ultrachat_200k...")
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

print(f"Total examples: {len(dataset)}")

# Take a subset (50k is enough for learning)
subset_size = 50000
dataset = dataset.shuffle(seed=42).select(range(subset_size))
print(f"Using subset: {subset_size} examples")

# Save to disk
output_dir = Path("data/stage1_chat")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nSaving to {output_dir}/train.jsonl...")
with open(output_dir / "train.jsonl", "w") as f:
    for example in dataset:
        f.write(json.dumps(example) + "\n")

print(f"Done! Saved {subset_size} examples")

# Show sample
print("\n" + "="*50)
print("SAMPLE EXAMPLE:")
print("="*50)
sample = dataset[0]
print(f"\nMessages ({len(sample['messages'])} turns):")
for msg in sample['messages'][:4]:  # First 4 messages
    role = msg['role']
    content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
    print(f"\n[{role}]: {content}")
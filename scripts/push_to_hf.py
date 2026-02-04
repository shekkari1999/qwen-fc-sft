"""
Push fine-tuned model to HuggingFace Hub
Usage: python scripts/push_to_hf.py
"""

from unsloth import FastLanguageModel

# Config
MODEL_PATH = "./checkpoints/stage1/merged"
HF_REPO = "shekkari21/qwen-fc-stage1"  # Change this for stage2

print("="*60)
print("Pushing Model to HuggingFace")
print("="*60)

print(f"\nLoading model from {MODEL_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_PATH,
    max_seq_length=2048,
)

print(f"\nPushing to {HF_REPO}...")
model.push_to_hub_merged(
    HF_REPO,
    tokenizer,
    save_method="merged_16bit",
)

print("\n" + "="*60)
print("Upload complete!")
print(f"Model: https://huggingface.co/{HF_REPO}")
print("="*60)

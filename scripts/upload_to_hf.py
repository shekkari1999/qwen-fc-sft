"""
Upload processed datasets to HuggingFace Hub
This makes it easy to access the data from RunPod

Usage:
1. Login: huggingface-cli login
2. Run: python scripts/upload_to_hf.py
"""

from datasets import load_dataset, DatasetDict

HF_USERNAME = "shekkari21"
REPO_NAME = "qwen-fc-sft-data"

def main():
    print("="*60)
    print("Uploading datasets to HuggingFace Hub")
    print("="*60)
    
    # Load processed datasets
    print("\nLoading Stage 1 (Chat) data...")
    stage1 = load_dataset("json", data_files="data/stage1_chat/train_qwen.jsonl", split="train")
    print(f"  Loaded {len(stage1)} examples")
    
    print("\nLoading Stage 2 (FC) data...")
    stage2 = load_dataset("json", data_files="data/stage2_fc/train_qwen.jsonl", split="train")
    print(f"  Loaded {len(stage2)} examples")
    
    # Create dataset dict
    dataset_dict = DatasetDict({
        "stage1_chat": stage1,
        "stage2_fc": stage2,
    })
    
    # Push to hub
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    print(f"\nPushing to {repo_id}...")
    
    dataset_dict.push_to_hub(
        repo_id,
        private=True,
    )
    
    print("\n" + "="*60)
    print("Upload complete!")
    print(f"Dataset: https://huggingface.co/datasets/{repo_id}")
    print("="*60)

if __name__ == "__main__":
    main()

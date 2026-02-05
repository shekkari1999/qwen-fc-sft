"""
Upload datasets to HuggingFace Hub
Usage: python scripts/upload.py --repo your-username/repo-name
"""
from datasets import load_dataset, DatasetDict
import argparse


def upload_to_hub(repo_id, private=True):
    """Upload processed datasets to HuggingFace Hub"""
    print("="*60)
    print(f"UPLOADING TO: {repo_id}")
    print("="*60)

    # Load processed datasets
    print("\nLoading Stage 1 (Chat) data...")
    stage1 = load_dataset("json", data_files="data/stage1_chat/train_single.jsonl", split="train")
    print(f"  Loaded {len(stage1)} examples")

    print("\nLoading Stage 2 (FC) data...")
    stage2 = load_dataset("json", data_files="data/stage2_fc/train_single.jsonl", split="train")
    print(f"  Loaded {len(stage2)} examples")

    # Create dataset dict
    dataset_dict = DatasetDict({
        "stage1_chat": stage1,
        "stage2_fc": stage2,
    })

    # Push to hub
    print(f"\nPushing to {repo_id}...")
    dataset_dict.push_to_hub(repo_id, private=private)

    print("\n" + "="*60)
    print("Upload complete!")
    print(f"Dataset: https://huggingface.co/datasets/{repo_id}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HuggingFace repo (username/repo-name)")
    parser.add_argument("--public", action="store_true", help="Make dataset public")
    args = parser.parse_args()

    upload_to_hub(args.repo, private=not args.public)

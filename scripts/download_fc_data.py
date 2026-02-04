from datasets import load_dataset 
import json
from pathlib import Path                                                                       

# Download the dataset                                                                         
print("Downloading Salesforce/xlam-function-calling-60k...")
dataset = load_dataset("Salesforce/xlam-function-calling-60k")

# Save raw data
raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

# Save as jsonl for easy inspection
print(f"Saving to {raw_dir}/train.jsonl...")
with open(raw_dir / "train.jsonl", "w") as f:
    for example in dataset["train"]:
        f.write(json.dumps(example) + "\n")

print(f"Done! {len(dataset['train'])} examples saved.")

# Show a sample
print("\n=== SAMPLE EXAMPLE ===")
print(json.dumps(dataset["train"][0], indent=2))
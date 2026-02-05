"""
Prepare training data: download, preprocess, create minimal dataset
Usage:
  python scripts/prepare_data.py --action download
  python scripts/prepare_data.py --action preprocess
  python scripts/prepare_data.py --action minimal
"""
import json
from pathlib import Path
from datasets import load_dataset
import random
import argparse


def download_data():
    """Download raw datasets"""
    print("="*60)
    print("DOWNLOADING RAW DATA")
    print("="*60)

    # Chat data (UltraChat)
    print("\nDownloading UltraChat (chat data)...")
    chat_data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:10000]")

    Path("data/stage1_chat").mkdir(parents=True, exist_ok=True)
    with open("data/stage1_chat/train.jsonl", "w") as f:
        for example in chat_data:
            f.write(json.dumps({"messages": example["messages"]}) + "\n")
    print(f"  Saved {len(chat_data)} examples to data/stage1_chat/train.jsonl")

    # FC data (Salesforce)
    print("\nDownloading xlam-function-calling (FC data)...")
    fc_data = load_dataset("Salesforce/xlam-function-calling-60k", split="train[:10000]")

    Path("data/stage2_fc").mkdir(parents=True, exist_ok=True)
    with open("data/stage2_fc/train.jsonl", "w") as f:
        for example in fc_data:
            f.write(json.dumps(example) + "\n")
    print(f"  Saved {len(fc_data)} examples to data/stage2_fc/train.jsonl")

    print("\nDownload complete!")


def preprocess_single_turn(max_examples=10000):
    """Convert multi-turn to single-turn and format FC data"""
    print("="*60)
    print("PREPROCESSING TO SINGLE-TURN")
    print("="*60)

    # Chat data
    print("\nProcessing chat data...")
    input_file = Path("data/stage1_chat/train.jsonl")
    output_file = Path("data/stage1_chat/train_single.jsonl")

    single_examples = []
    with open(input_file) as f:
        for line in f:
            example = json.loads(line)
            messages = example["messages"]

            i = 0
            if messages[0]["role"] == "system":
                i = 1

            while i < len(messages) - 1:
                if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                    single_examples.append({
                        "messages": [messages[i], messages[i+1]]
                    })
                i += 2

            if len(single_examples) >= max_examples * 2:
                break

    random.seed(42)
    random.shuffle(single_examples)
    single_examples = single_examples[:max_examples]

    with open(output_file, "w") as f:
        for ex in single_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Created {len(single_examples)} single-turn chat examples")

    # FC data
    print("\nProcessing FC data...")
    input_file = Path("data/stage2_fc/train.jsonl")
    output_file = Path("data/stage2_fc/train_single.jsonl")

    fc_examples = []
    with open(input_file) as f:
        for line in f:
            if len(fc_examples) >= max_examples:
                break
            example = json.loads(line)
            try:
                query = example["query"]
                tools = json.loads(example["tools"])
                answers = json.loads(example["answers"])

                tools_desc = json.dumps(tools, indent=2)
                system_prompt = f"""You are a helpful assistant with tools:

{tools_desc}

Use tools with: <tool_call>{{"name": "...", "arguments": {{...}}}}</tool_call>"""

                if len(answers) == 1:
                    response = f"<tool_call>\n{json.dumps(answers[0])}\n</tool_call>"
                else:
                    response = "\n".join([f"<tool_call>\n{json.dumps(a)}\n</tool_call>" for a in answers])

                fc_examples.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": response}
                    ]
                })
            except:
                continue

    with open(output_file, "w") as f:
        for ex in fc_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Created {len(fc_examples)} FC examples")

    print("\nPreprocessing complete!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["download", "preprocess", "minimal", "all"], default="all")
    parser.add_argument("--max", type=int, default=10000)
    args = parser.parse_args()

    if args.action in ["download", "all"]:
        download_data()
    if args.action in ["preprocess", "all"]:
        preprocess_single_turn(args.max)

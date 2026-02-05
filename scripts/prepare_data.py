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


def create_minimal(num_examples=200):
    """Create minimal short Q&A dataset for testing"""
    print("="*60)
    print(f"CREATING MINIMAL DATASET ({num_examples} examples)")
    print("="*60)

    qa_pairs = [
        ("What is 2 + 2?", "4"),
        ("What is 5 + 3?", "8"),
        ("What is 10 - 4?", "6"),
        ("What is the capital of France?", "Paris."),
        ("What is the capital of Japan?", "Tokyo."),
        ("What is the capital of Germany?", "Berlin."),
        ("What color is the sky?", "Blue."),
        ("What color is grass?", "Green."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
        ("What is the largest planet?", "Jupiter."),
        ("How many days in a week?", "Seven."),
        ("Say hello.", "Hello!"),
        ("Say goodbye.", "Goodbye!"),
        ("Is water wet?", "Yes."),
        ("Is fire hot?", "Yes."),
        ("What is H2O?", "Water."),
        ("What do dogs say?", "Woof."),
        ("What do cats say?", "Meow."),
        ("What is 100 + 1?", "101"),
    ]

    examples = []
    for q, a in qa_pairs:
        examples.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})

    while len(examples) < num_examples:
        examples.append(random.choice(examples[:len(qa_pairs)]))

    random.seed(42)
    random.shuffle(examples)
    examples = examples[:num_examples]

    Path("data/minimal").mkdir(parents=True, exist_ok=True)
    with open("data/minimal/train.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"  Created {len(examples)} short Q&A examples")
    print(f"  Saved to data/minimal/train.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["download", "preprocess", "minimal", "all"], default="all")
    parser.add_argument("--max", type=int, default=10000)
    args = parser.parse_args()

    if args.action in ["download", "all"]:
        download_data()
    if args.action in ["preprocess", "all"]:
        preprocess_single_turn(args.max)
    if args.action in ["minimal", "all"]:
        create_minimal()

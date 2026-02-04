"""
Preprocess data to create SINGLE-TURN examples
This helps the model learn when to stop generating.

Usage: python scripts/preprocess_single_turn.py
"""

import json
from pathlib import Path
import random

# Config
MAX_CHAT_EXAMPLES = 10000
MAX_FC_EXAMPLES = 10000  # Also reduce FC data for faster Stage 2

def preprocess_chat_single_turn():
    """Convert multi-turn UltraChat to single-turn examples"""
    print("Processing Stage 1: Chat data (SINGLE TURN)...")

    input_file = Path("data/stage1_chat/train.jsonl")
    output_file = Path("data/stage1_chat/train_qwen_single.jsonl")

    single_turn_examples = []

    with open(input_file) as f_in:
        for line in f_in:
            example = json.loads(line)
            messages = example["messages"]

            # Extract each user-assistant pair as separate example
            system_msg = {"role": "system", "content": "You are a helpful assistant."}

            i = 0
            # Skip system if present
            if messages[0]["role"] == "system":
                system_msg = messages[0]
                i = 1

            # Extract pairs
            while i < len(messages) - 1:
                if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                    single_example = {
                        "messages": [
                            system_msg,
                            messages[i],      # user
                            messages[i+1]     # assistant
                        ]
                    }
                    single_turn_examples.append(single_example)
                i += 2

            # Stop if we have enough
            if len(single_turn_examples) >= MAX_CHAT_EXAMPLES * 2:
                break

    # Shuffle and take only what we need
    random.seed(42)
    random.shuffle(single_turn_examples)
    single_turn_examples = single_turn_examples[:MAX_CHAT_EXAMPLES]

    # Save
    with open(output_file, "w") as f_out:
        for ex in single_turn_examples:
            f_out.write(json.dumps(ex) + "\n")

    print(f"  Created {len(single_turn_examples)} single-turn examples")
    print(f"  Saved to {output_file}")
    return len(single_turn_examples)


def preprocess_fc_data():
    """Convert Salesforce FC data to Qwen format (subset)"""
    print("\nProcessing Stage 2: Function calling data...")

    input_file = Path("data/stage2_fc/train.jsonl")
    output_file = Path("data/stage2_fc/train_qwen_single.jsonl")

    examples = []
    skipped = 0

    with open(input_file) as f_in:
        for line in f_in:
            if len(examples) >= MAX_FC_EXAMPLES:
                break

            example = json.loads(line)

            try:
                query = example["query"]
                tools = json.loads(example["tools"])
                answers = json.loads(example["answers"])

                # Build system prompt with tools
                tools_desc = json.dumps(tools, indent=2)
                system_prompt = f"""You are a helpful assistant with access to the following tools:

{tools_desc}

When you need to use a tool, respond with:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
</tool_call>"""

                # Build assistant response with tool calls
                if len(answers) == 1:
                    tool_response = f"<tool_call>\n{json.dumps(answers[0])}\n</tool_call>"
                else:
                    tool_response = "\n".join([
                        f"<tool_call>\n{json.dumps(ans)}\n</tool_call>"
                        for ans in answers
                    ])

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": tool_response}
                ]

                examples.append({"messages": messages})

            except Exception as e:
                skipped += 1
                continue

    # Shuffle
    random.seed(42)
    random.shuffle(examples)

    # Save
    with open(output_file, "w") as f_out:
        for ex in examples:
            f_out.write(json.dumps(ex) + "\n")

    print(f"  Created {len(examples)} examples")
    print(f"  Skipped {skipped} (parsing errors)")
    print(f"  Saved to {output_file}")
    return len(examples)


if __name__ == "__main__":
    print("="*60)
    print("PREPROCESSING DATA - SINGLE TURN EXAMPLES")
    print("="*60)
    print(f"Target: {MAX_CHAT_EXAMPLES} chat + {MAX_FC_EXAMPLES} FC examples\n")

    chat_count = preprocess_chat_single_turn()
    fc_count = preprocess_fc_data()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Stage 1 (Chat):     {chat_count:,} single-turn examples")
    print(f"Stage 2 (FC):       {fc_count:,} examples")
    print(f"Total:              {chat_count + fc_count:,} examples")
    print("\nNext steps:")
    print("1. Run: python scripts/upload_single_turn.py")
    print("2. Update train scripts to use new dataset split")

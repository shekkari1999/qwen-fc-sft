import json
from pathlib import Path

def preprocess_chat_data():
    """Convert UltraChat to Qwen format"""
    print("Processing Stage 1: Chat data...")

    input_file = Path("data/stage1_chat/train.jsonl")
    output_file = Path("data/stage1_chat/train_qwen.jsonl")

    count = 0
    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            example = json.loads(line)

            # UltraChat already has messages format, just ensure system prompt
            messages = example["messages"]

            # Add system prompt if not present
            if messages[0]["role"] != "system":
                messages.insert(0, {
                    "role": "system",
                    "content": "You are a helpful assistant."
                })

            f_out.write(json.dumps({"messages": messages}) + "\n")
            count += 1

    print(f"  Saved {count} examples to {output_file}")
    return count


def preprocess_fc_data():
    """Convert Salesforce FC data to Qwen format"""
    print("Processing Stage 2: Function calling data...")

    input_file = Path("data/stage2_fc/train.jsonl")
    output_file = Path("data/stage2_fc/train_qwen.jsonl")

    count = 0
    skipped = 0

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
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
                    # Multiple tool calls
                    tool_response = "\n".join([
                        f"<tool_call>\n{json.dumps(ans)}\n</tool_call>"
                        for ans in answers
                    ])

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": tool_response}
                ]

                f_out.write(json.dumps({"messages": messages}) + "\n")
                count += 1

            except Exception as e:
                skipped += 1
                continue

    print(f"  Saved {count} examples to {output_file}")
    print(f"  Skipped {skipped} examples (parsing errors)")
    return count
if __name__ == "__main__":
    print("="*50)
    print("PREPROCESSING DATA FOR QWEN SFT")
    print("="*50 + "\n")

    chat_count = preprocess_chat_data()
    fc_count = preprocess_fc_data()

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Stage 1 (Chat):     {chat_count:,} examples")
    print(f"Stage 2 (FC):       {fc_count:,} examples")
    print(f"Total:              {chat_count + fc_count:,} examples")
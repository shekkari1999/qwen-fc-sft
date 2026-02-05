"""Show training examples with chat template applied."""
import json
import argparse
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to jsonl file")
    parser.add_argument("--num", type=int, default=5, help="Number of examples")
    parser.add_argument("--output", type=str, default="docs/training_examples.md")
    args = parser.parse_args()

    # Load Qwen tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

    # Load examples
    examples = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            if i >= args.num:
                break
            examples.append(json.loads(line))

    # Format output
    output_lines = ["# Training Examples with Chat Template Applied\n"]
    output_lines.append(f"Source: `{args.data}`\n")
    output_lines.append(f"Examples: {len(examples)}\n")
    output_lines.append("---\n")

    for i, ex in enumerate(examples):
        messages = ex["messages"]

        # Apply chat template
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        output_lines.append(f"## Example {i+1}\n")
        output_lines.append("**Raw messages:**\n```json")
        output_lines.append(json.dumps(messages, indent=2))
        output_lines.append("```\n")
        output_lines.append("**After chat template:**\n```")
        output_lines.append(formatted)
        output_lines.append("```\n")
        output_lines.append("---\n")

    # Save
    with open(args.output, "w") as f:
        f.write("\n".join(output_lines))

    print(f"Saved {len(examples)} examples to {args.output}")

if __name__ == "__main__":
    main()

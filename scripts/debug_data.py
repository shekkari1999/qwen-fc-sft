"""
Debug: Check the actual training data format from HuggingFace Hub
"""
from datasets import load_dataset
from transformers import AutoTokenizer


def debug_data():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    print(f"<|im_end|> token ID: {im_end_id}")
    print(f"<|im_start|> token ID: {im_start_id}")

    print("\n" + "="*70)
    print("LOADING DATA FROM HUB")
    print("="*70)

    dataset = load_dataset("shekkari21/qwen-fc-sft-data-v2", split="stage1_chat[:3]")

    for i, example in enumerate(dataset):
        print(f"\n{'='*70}")
        print(f"EXAMPLE {i+1}")
        print("="*70)

        messages = example["messages"]
        print(f"\nRaw messages ({len(messages)} turns):")
        for msg in messages:
            print(f"  [{msg['role']}]: {msg['content'][:100]}...")

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        print(f"\n After chat template (last 200 chars):")
        print(repr(text[-200:]))

        # Check if it ends with <|im_end|>
        ends_clean = text.rstrip().endswith('<|im_end|>')
        ends_with_newline = text.endswith('<|im_end|>\n')
        print(f"\n Ends with '<|im_end|>': {ends_clean}")
        print(f" Ends with '<|im_end|>' + newline: {ends_with_newline}")

        # Tokenize and check last tokens
        tokens = tokenizer.encode(text)
        print(f"\n Last 10 token IDs: {tokens[-10:]}")
        print(f" Last 10 tokens decoded:")
        for tid in tokens[-10:]:
            print(f"   {tid}: {repr(tokenizer.decode([tid]))}")

        # Check if <|im_end|> is the last meaningful token
        last_non_newline = tokens[-1] if tokenizer.decode([tokens[-1]]).strip() else tokens[-2]
        print(f"\n Last non-whitespace token: {last_non_newline} = {repr(tokenizer.decode([last_non_newline]))}")
        print(f" Is it <|im_end|>? {last_non_newline == im_end_id}")


if __name__ == "__main__":
    debug_data()

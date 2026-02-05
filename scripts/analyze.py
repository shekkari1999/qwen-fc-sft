"""
Analyze dataset: sequence lengths, label masking verification, data inspection
Usage: python scripts/analyze.py
"""
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import argparse

def plot_seq_lengths(dataset_name, split):
    """Plot sequence length distribution"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

    print(f"Loading {dataset_name} ({split})...")
    dataset = load_dataset(dataset_name, split=split)

    print(f"Processing {len(dataset)} examples...")
    seq_lengths = []
    for example in dataset:
        messages = example["messages"]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer.encode(formatted, add_special_tokens=False)
        seq_lengths.append(len(tokens))

    # Stats
    print(f"\nSequence Length Statistics:")
    print(f"  Min:    {min(seq_lengths)}")
    print(f"  Max:    {max(seq_lengths)}")
    print(f"  Mean:   {sum(seq_lengths) // len(seq_lengths)}")
    print(f"  Median: {sorted(seq_lengths)[len(seq_lengths)//2]}")

    under_256 = sum(1 for x in seq_lengths if x < 256)
    under_512 = sum(1 for x in seq_lengths if x < 512)
    print(f"\n  < 256 tokens:  {under_256:,} ({100*under_256/len(seq_lengths):.1f}%)")
    print(f"  < 512 tokens:  {under_512:,} ({100*under_512/len(seq_lengths):.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(seq_lengths, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=256, color='r', linestyle='--', label='256')
    axes[0].axvline(x=512, color='g', linestyle='--', label='512')
    axes[0].set_xlabel('Sequence Length (tokens)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Sequence Length Distribution')
    axes[0].legend()

    sorted_lengths = sorted(seq_lengths)
    cumulative = [i/len(sorted_lengths) * 100 for i in range(len(sorted_lengths))]
    axes[1].plot(sorted_lengths, cumulative)
    axes[1].set_xlabel('Sequence Length (tokens)')
    axes[1].set_ylabel('Cumulative %')
    axes[1].set_title('Cumulative Distribution')

    plt.tight_layout()
    plt.savefig('seq_lengths.png', dpi=150)
    print(f"\nPlot saved to: seq_lengths.png")
    plt.show()


def verify_labels(dataset_name, split, num_examples=5):
    """Show how label masking works on examples"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

    print(f"Loading {dataset_name} ({split})...")
    dataset = load_dataset(dataset_name, split=f"{split}[:{num_examples}]")

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    print("\n" + "="*70)
    print("LABEL MASKING VERIFICATION")
    print("="*70)

    for idx, example in enumerate(dataset):
        messages = example["messages"]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer.encode(formatted, add_special_tokens=False)

        print(f"\n--- Example {idx+1} ---")
        print(f"Roles: {[m['role'] for m in messages]}")
        print(f"Total tokens: {len(tokens)}")

        # Find assistant start
        im_start_positions = [i for i, t in enumerate(tokens) if t == im_start_id]
        last_im_start = im_start_positions[-1] if im_start_positions else 0

        print(f"\nToken breakdown (around assistant start):")
        for i in range(max(0, last_im_start-2), min(len(tokens), last_im_start+8)):
            token_text = tokenizer.decode([tokens[i]]).replace('\n', '\\n')
            label = "-100 (masked)" if i < last_im_start + 3 else f"{tokens[i]} (TRAINED)"
            print(f"  [{i:3d}] {token_text:<20} → {label}")

    print("\n" + "="*70)
    print("train_on_responses_only() masks everything before '<|im_start|>assistant\\n'")
    print("="*70)


def inspect_data(dataset_name, split, num_examples=5):
    """Show raw examples from dataset"""
    dataset = load_dataset(dataset_name, split=f"{split}[:{num_examples}]")

    print("\n" + "="*70)
    print(f"DATA INSPECTION: {dataset_name}")
    print("="*70)

    for idx, example in enumerate(dataset):
        print(f"\n--- Example {idx+1} ---")
        for msg in example["messages"]:
            content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
            print(f"[{msg['role']}]: {content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["plot", "verify", "inspect"], default="plot")
    parser.add_argument("--dataset", default="shekkari21/qwen-fc-sft-data-v2")
    parser.add_argument("--split", default="stage1_chat")
    parser.add_argument("--num", type=int, default=5)
    args = parser.parse_args()

    if args.action == "plot":
        plot_seq_lengths(args.dataset, args.split)
    elif args.action == "verify":
        verify_labels(args.dataset, args.split, args.num)
    elif args.action == "inspect":
        inspect_data(args.dataset, args.split, args.num)

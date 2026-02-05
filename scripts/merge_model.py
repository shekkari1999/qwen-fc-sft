"""
Re-merge LoRA adapter with proper tied weights handling
Usage: python scripts/merge_model.py --adapter ./checkpoints/stage1/final --output ./checkpoints/stage1/merged_fixed
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import argparse

def merge_model(adapter_path, output_path):
    print(f"Loading LoRA adapter from {adapter_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # Load in full precision for merging
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()

    # Re-tie weights if needed
    if hasattr(merged_model.config, 'tie_word_embeddings') and merged_model.config.tie_word_embeddings:
        print("Re-tying word embeddings after merge...")
        merged_model.tie_weights()

    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="./checkpoints/stage1/final")
    parser.add_argument("--output", default="./checkpoints/stage1/merged_fixed")
    args = parser.parse_args()

    merge_model(args.adapter, args.output)

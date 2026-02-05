"""
Debug: Check what probabilities the model assigns to <|im_end|> vs other tokens
after generating a response.
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import torch.nn.functional as F
import argparse


def debug_probs(model_path):
    print(f"Loading model from {model_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    print(f"\n<|im_end|> token ID: {im_end_id}")
    print(f"<|im_start|> token ID: {im_start_id}")

    # Test prompts with expected short answers
    test_cases = [
        ("What is 2 + 2?", "4"),
        ("What is the capital of France?", "Paris"),
        ("Say hello.", "Hello!"),
    ]

    print("\n" + "="*70)
    print("DEBUGGING TOKEN PROBABILITIES")
    print("="*70)

    for prompt, expected in test_cases:
        print(f"\n{'='*70}")
        print(f"Prompt: {prompt}")
        print(f"Expected answer contains: {expected}")
        print("="*70)

        # Create input with generation prompt
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        # Generate step by step and check probabilities
        generated_ids = []
        current_input = inputs

        for step in range(30):  # Max 30 tokens
            with torch.no_grad():
                outputs = model(current_input)
                logits = outputs.logits[0, -1, :]  # Last token logits
                probs = F.softmax(logits, dim=-1)

                # Get top 10 tokens
                top_probs, top_ids = torch.topk(probs, 10)

                # Get probability of <|im_end|>
                im_end_prob = probs[im_end_id].item()
                im_end_rank = (probs > probs[im_end_id]).sum().item() + 1

                # Greedy next token
                next_id = logits.argmax().item()
                next_token = tokenizer.decode([next_id])

                generated_ids.append(next_id)
                generated_text = tokenizer.decode(generated_ids)

                # Print status
                print(f"\nStep {step + 1}:")
                print(f"  Generated so far: {repr(generated_text)}")
                print(f"  Next token: {repr(next_token)} (id={next_id})")
                print(f"  <|im_end|> prob: {im_end_prob:.6f} (rank #{im_end_rank})")
                print(f"  Top 5 tokens:")
                for i in range(5):
                    tok = tokenizer.decode([top_ids[i].item()])
                    print(f"    {i+1}. {repr(tok)} ({top_probs[i].item():.4f})")

                # Stop if we hit im_end or generated enough
                if next_id == im_end_id:
                    print("\n  >>> MODEL GENERATED <|im_end|> - STOPPING!")
                    break

                # Check if we've passed a reasonable stopping point
                if expected.lower() in generated_text.lower() and step > 5:
                    print(f"\n  >>> Answer '{expected}' found but model didn't stop")
                    print(f"  >>> <|im_end|> probability: {im_end_prob:.6f} (rank #{im_end_rank})")

                # Update input for next step
                current_input = torch.cat([
                    current_input,
                    torch.tensor([[next_id]], device=model.device)
                ], dim=1)

        print(f"\nFinal generation: {tokenizer.decode(generated_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./checkpoints/stage1/merged")
    args = parser.parse_args()
    debug_probs(args.model)

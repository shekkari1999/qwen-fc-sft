"""
Test trained model inference
Usage: python scripts/inference.py --model ./checkpoints/stage1/merged
       python scripts/inference.py --model Qwen/Qwen2.5-3B -i
"""
import torch
import argparse

# Try unsloth first (GPU), fall back to transformers (CPU/MPS)
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    USE_UNSLOTH = True
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    USE_UNSLOTH = False
    print("Unsloth not available, using transformers (slower)")


def load_model(model_path):
    """Load model with unsloth (GPU) or transformers (CPU/MPS)"""
    print(f"Loading model from {model_path}...")

    if USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
        FastLanguageModel.for_inference(model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Use MPS on Mac if available, else CPU
        if torch.backends.mps.is_available():
            device = "mps"
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model = model.to(device)
        else:
            device = "cpu"
            model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        print(f"Using device: {device}")

    return model, tokenizer


def test_model(model_path, prompts=None):
    model, tokenizer = load_model(model_path)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"Stop token: <|im_end|> (id: {im_end_id})")

    if prompts is None:
        prompts = [
            "What is 2 + 2?",
            "What is the capital of France?",
            "What color is the sky?",
            "Who wrote Romeo and Juliet?",
            "Say hello.",
        ]

    print("\n" + "="*60)
    print("INFERENCE TEST")
    print("="*60)

    for prompt in prompts:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=im_end_id,
            )

        generated = outputs[0][inputs.shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        stopped = im_end_id in generated.tolist()

        print(f"\nQ: {prompt}")
        print(f"A: {response}")
        print(f"   [tokens: {len(generated)}, stopped: {'YES' if stopped else 'NO'}]")

    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./checkpoints/stage1/merged")
    parser.add_argument("--prompt", type=str, nargs="*", help="Custom prompts")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive:
        # Interactive chat mode
        model, tokenizer = load_model(args.model)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        print(f"\nInteractive mode. Type 'quit' to exit.")
        print(f"Stop token: <|im_end|> (id: {im_end_id})\n")

        while True:
            try:
                prompt = input("You: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if not prompt:
                    continue

                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs,
                        max_new_tokens=200,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=im_end_id,
                    )

                generated = outputs[0][inputs.shape[1]:]
                response = tokenizer.decode(generated, skip_special_tokens=True).strip()
                stopped = im_end_id in generated.tolist()

                print(f"AI: {response}")
                print(f"   [tokens: {len(generated)}, stopped: {'YES' if stopped else 'NO'}]\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
    else:
        test_model(args.model, args.prompt)

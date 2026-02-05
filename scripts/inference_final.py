"""
Final inference script for fine-tuned model (uses LoRA adapter)
Usage:
  python scripts/inference_final.py                    # Interactive mode
  python scripts/inference_final.py --prompt "Hello"   # Single prompt
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import argparse


def load_model(model_path="./checkpoints/stage1/final"):
    """Load model with LoRA adapter"""
    print(f"Loading model from {model_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=256, temperature=0.0):
    """Generate response for a prompt"""
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        if temperature == 0:
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=im_end_id,
            )
        else:
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=im_end_id,
            )

    generated = outputs[0][inputs.shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    stopped = im_end_id in generated.tolist()

    return response, len(generated), stopped


def interactive_mode(model, tokenizer):
    """Interactive chat mode"""
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    print("\n" + "="*60)
    print("INTERACTIVE CHAT")
    print(f"Model: LoRA adapter (stop token: {im_end_id})")
    print("Commands: 'quit' to exit, 'temp X' to set temperature")
    print("="*60 + "\n")

    temperature = 0.0

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if user_input.lower().startswith('temp '):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"Temperature set to {temperature}")
                except:
                    print("Usage: temp 0.7")
                continue

            if not user_input:
                continue

            response, tokens, stopped = generate(model, tokenizer, user_input, temperature=temperature)

            print(f"AI: {response}")
            print(f"   [tokens: {tokens}, stopped: {'YES' if stopped else 'NO'}]\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned model")
    parser.add_argument("--model", default="./checkpoints/stage1/final",
                        help="Path to LoRA adapter")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (if not provided, enters interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.prompt:
        response, tokens, stopped = generate(
            model, tokenizer, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(f"Response: {response}")
        print(f"[tokens: {tokens}, stopped: {'YES' if stopped else 'NO'}]")
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()

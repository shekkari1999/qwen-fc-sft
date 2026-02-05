"""
Compare base model vs fine-tuned model side by side
Usage: python scripts/compare.py
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

def load_model(model_path, name):
    print(f"Loading {name} from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate(model, tokenizer, prompt, max_tokens=100):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=im_end_id,
        )

    generated = outputs[0][inputs.shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    stopped = im_end_id in generated.tolist()

    return response, len(generated), stopped

def main():
    # Load both models
    base_model, base_tok = load_model("Qwen/Qwen2.5-3B", "BASE MODEL")
    ft_model, ft_tok = load_model("./checkpoints/stage1/merged", "FINE-TUNED MODEL")

    print("\n" + "="*70)
    print("SIDE-BY-SIDE COMPARISON: Base vs Fine-tuned")
    print("Type 'quit' to exit")
    print("="*70 + "\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if not prompt:
                continue

            # Generate from both
            base_resp, base_toks, base_stopped = generate(base_model, base_tok, prompt)
            ft_resp, ft_toks, ft_stopped = generate(ft_model, ft_tok, prompt)

            # Display side by side
            print("\n" + "-"*70)
            print("BASE MODEL:")
            print(f"  {base_resp}")
            print(f"  [tokens: {base_toks}, stopped: {'YES' if base_stopped else 'NO'}]")
            print()
            print("FINE-TUNED:")
            print(f"  {ft_resp}")
            print(f"  [tokens: {ft_toks}, stopped: {'YES' if ft_stopped else 'NO'}]")
            print("-"*70 + "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()

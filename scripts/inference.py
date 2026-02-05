"""
Test trained model inference
Usage: python scripts/inference.py --model ./checkpoints/stage1/merged
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import argparse


def test_model(model_path, prompts=None):
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
    args = parser.parse_args()

    test_model(args.model, args.prompt)

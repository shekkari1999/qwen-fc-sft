"""
Debug tool_call token output - check what's really being generated
Usage: python scripts/debug_toolcall.py --model ./checkpoints/stage2/checkpoint-500
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./checkpoints/stage2/checkpoint-500")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"<|im_end|> token ID: {im_end_id}")

    # Check if <tool_call> is a special token
    tool_call_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    print(f"<tool_call> token ID: {tool_call_id}")

    # System prompt with tools
    system_msg = """You are a helpful assistant with access to the following tools:

[{"name": "get_weather", "parameters": {"location": {"type": "str"}}},
{"name": "calculate", "parameters": {"expression": {"type": "str"}}}]

When you need to use a tool, respond with:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call>"""

    prompts = ["What's the weather in Tokyo?", "Calculate 15 * 7"]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Q: {prompt}")
        print("="*60)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]

        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
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

        # Show raw token IDs
        print(f"\nRaw token IDs (first 30): {generated.tolist()[:30]}")

        # Decode each token individually
        print("\nToken-by-token decode:")
        for i, tok_id in enumerate(generated.tolist()[:20]):
            tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
            print(f"  {i}: ID={tok_id} -> {repr(tok_str)}")

        # Full decode with and without skip_special_tokens
        resp_with_special = tokenizer.decode(generated, skip_special_tokens=False)
        resp_no_special = tokenizer.decode(generated, skip_special_tokens=True)

        print(f"\nWith special tokens: {repr(resp_with_special)}")
        print(f"\nWithout special tokens: {repr(resp_no_special)}")

        stopped = im_end_id in generated.tolist()
        print(f"\nStopped at <|im_end|>: {stopped}")

if __name__ == "__main__":
    main()

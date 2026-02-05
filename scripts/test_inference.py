"""
Test trained model with proper stop token detection
Usage: python scripts/test_inference.py
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

# Load the merged model
MODEL_PATH = "./checkpoints/stage1_minimal/merged"
print(f"Loading model from {MODEL_PATH}...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
FastLanguageModel.for_inference(model)

# Get stop token
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"Stop token: <|im_end|> (id: {im_end_id})")

# Test prompts
test_prompts = [
    "What is 2 + 2?",
    "What is the capital of France?",
    "What color is the sky?",
    "Who wrote Romeo and Juliet?",
    "Say hello.",
]

print("\n" + "="*60)
print("INFERENCE TEST")
print("="*60)

for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    input_len = inputs.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=im_end_id,
        )

    generated = outputs[0][input_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    stopped = im_end_id in generated.tolist()

    print(f"\nQ: {prompt}")
    print(f"A: {response}")
    print(f"   [tokens: {len(generated)}, stopped: {'YES' if stopped else 'NO'}]")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)

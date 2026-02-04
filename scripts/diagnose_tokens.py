"""
Diagnose token issues with Unsloth
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

MODEL_PATH = "./checkpoints/stage1_v2/merged"
print(f"Loading model from {MODEL_PATH}...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

print("\n" + "="*60)
print("BEFORE get_chat_template")
print("="*60)
print(f"eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
im_end_before = tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"<|im_end|> id: {im_end_before}")

# Apply chat template
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

print("\n" + "="*60)
print("AFTER get_chat_template")
print("="*60)
print(f"eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
im_end_after = tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"<|im_end|> id: {im_end_after}")

# Check what the template produces
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"}
]

print("\n" + "="*60)
print("TRAINING DATA FORMAT")
print("="*60)
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(repr(text))

tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
if hasattr(tokens, 'tolist'):
    tokens = tokens.tolist()
elif hasattr(tokens, 'ids'):
    tokens = tokens.ids

print(f"\nLast 10 tokens: {tokens[-10:]}")
print(f"Decoded: {[tokenizer.decode([t]) for t in tokens[-10:]]}")

# Generate ONE short response and look at raw tokens
print("\n" + "="*60)
print("RAW GENERATION TEST")
print("="*60)

FastLanguageModel.for_inference(model)

messages_test = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]

inputs = tokenizer.apply_chat_template(
    messages_test,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

input_len = inputs.shape[1]

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=30,  # Short!
        do_sample=False,  # Deterministic
        pad_token_id=tokenizer.pad_token_id,
    )

generated = outputs[0][input_len:]
print(f"Generated token IDs: {generated.tolist()}")
print(f"Decoded tokens: {[tokenizer.decode([t]) for t in generated.tolist()]}")
print(f"\n<|im_end|> (id {im_end_after}) in output? {im_end_after in generated.tolist()}")
print(f"<|endoftext|> (id {tokenizer.eos_token_id}) in output? {tokenizer.eos_token_id in generated.tolist()}")

# Check for any end-like tokens in first 30
print("\nFull decoded output:")
print(tokenizer.decode(generated, skip_special_tokens=False))

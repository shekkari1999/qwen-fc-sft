"""
Debug script to investigate why model isn't stopping properly
"""

from unsloth import FastLanguageModel
from datasets import load_dataset

print("="*60)
print("DEBUGGING INFERENCE ISSUE")
print("="*60)

# 1. Check training data format
print("\n[1] Checking training data format...")
try:
    ds = load_dataset("shekkari21/qwen-fc-sft-data", split="stage1_chat")
    sample = ds[0]
    print(f"Sample messages: {sample['messages']}")
    print(f"\nNumber of messages: {len(sample['messages'])}")
    for i, msg in enumerate(sample['messages']):
        print(f"  Message {i}: role={msg['role']}, content={msg['content'][:100]}...")
except Exception as e:
    print(f"Error loading dataset: {e}")

# 2. Load model and tokenizer
print("\n[2] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    "/workspace/qwen-fc-sft/checkpoints/stage1/merged",
    max_seq_length=2048,
)
FastLanguageModel.for_inference(model)

# 3. Check special tokens
print("\n[3] Special tokens:")
print(f"  EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
print(f"  PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"  BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")

# Check for im_end token
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
print(f"  <|im_end|> id: {im_end_id}")
print(f"  <|im_start|> id: {im_start_id}")

# 4. Check chat template
print("\n[4] Chat template output:")
messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
print(f"  Template output:\n{repr(text)}")

# 5. Check what training data looks like after template
print("\n[5] Training data after chat template:")
if 'ds' in dir() and len(ds) > 0:
    sample_messages = ds[0]['messages']
    formatted = tokenizer.apply_chat_template(sample_messages, tokenize=False)
    print(f"  Formatted:\n{repr(formatted[:500])}...")

# 6. Generate and check token by token
print("\n[6] Token-by-token generation:")
messages = [{"role": "user", "content": "What is the capital of France?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

print(f"  Input length: {inputs.shape[1]} tokens")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=50,
    do_sample=False,  # Greedy for debugging
)

print(f"  Output length: {outputs.shape[1]} tokens")
print(f"\n  Token-by-token (generated part only):")

generated_tokens = outputs[0][inputs.shape[1]:].tolist()
for i, tok in enumerate(generated_tokens):
    decoded = tokenizer.decode([tok])
    print(f"    {i}: token_id={tok:6d} -> {repr(decoded)}")
    if tok == im_end_id:
        print(f"    ^^^ Found <|im_end|> at position {i}!")
    if tok == tokenizer.eos_token_id:
        print(f"    ^^^ Found EOS at position {i}!")

# 7. Full decoded output
print("\n[7] Full decoded output:")
print(tokenizer.decode(outputs[0]))

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)

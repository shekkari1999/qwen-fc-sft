"""
Debug: Check what the chat template actually produces
"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

# Check special tokens
print("="*60)
print("SPECIAL TOKENS")
print("="*60)
print(f"EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
print(f"PAD token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")

# Check im_end token
print("\n" + "="*60)
print("CHECKING QWEN SPECIAL TOKENS")
print("="*60)

# Get the im_end token id
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

print(f"<|im_end|> token id: {im_end_id}")
print(f"<|im_start|> token id: {im_start_id}")
print(f"<|endoftext|> token id: {endoftext_id}")
print(f"\neos_token_id: {tokenizer.eos_token_id}")
print(f"Is <|im_end|> == eos_token_id? {im_end_id == tokenizer.eos_token_id}")

# Test what training data looks like
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."}
]

print("\n" + "="*60)
print("TRAINING FORMAT (add_generation_prompt=False)")
print("="*60)
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(repr(text))  # Show with escape chars

print("\n" + "="*60)
print("TOKEN IDS AT END")
print("="*60)
tokens = tokenizer.encode(text, add_special_tokens=False)
print(f"Last 10 token IDs: {tokens[-10:]}")
print(f"Last 10 decoded: {[tokenizer.decode([t]) for t in tokens[-10:]]}")
print(f"\n<|im_end|> ({im_end_id}) appears at positions: {[i for i, t in enumerate(tokens) if t == im_end_id]}")
print(f"<|endoftext|> ({endoftext_id}) appears at positions: {[i for i, t in enumerate(tokens) if t == endoftext_id]}")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)
if im_end_id != tokenizer.eos_token_id:
    print("PROBLEM FOUND!")
    print(f"Training data ends with <|im_end|> (id: {im_end_id})")
    print(f"But eos_token_id is <|endoftext|> (id: {endoftext_id})")
    print("Model learns to generate <|im_end|> but inference looks for <|endoftext|>!")
    print("\nFIX: Use <|im_end|> as stop token:")
    print(f"    eos_token_id={im_end_id}  # or")
    print(f"    stop_token_ids=[{im_end_id}, {endoftext_id}]")

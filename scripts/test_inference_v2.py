"""
Test Stage 1 V2 Model - Proper Inference with Stop Token Detection
Usage: python scripts/test_inference_v2.py
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

# Load the merged model
MODEL_PATH = "./checkpoints/stage1_v2/merged"
print(f"Loading model from {MODEL_PATH}...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Apply chat template
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# Enable fast inference
FastLanguageModel.for_inference(model)

# Test prompts
test_prompts = [
    "What is 2 + 2?",
    "What is the capital of France?",
    "Explain machine learning in one sentence.",
    "Who wrote Romeo and Juliet?",
    "What color is the sky?",
]

# Show the token fix
im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"\nUsing <|im_end|> (id: {im_end_token_id}) as stop token")
print(f"(NOT <|endoftext|> which is id: {tokenizer.eos_token_id})")

print("\n" + "="*70)
print("INFERENCE TEST - Stage 1 V2 (Single-Turn)")
print("="*70)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n--- Test {i} ---")
    print(f"Q: {prompt}")

    # Prepare input
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs.shape[1]

    # Get the correct stop token (<|im_end|> not <|endoftext|>)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Generate with proper settings
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=im_end_id,  # FIXED: Use <|im_end|> (151645) not <|endoftext|> (151643)
        )

    # Get only the generated part
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # Check for EOS (im_end token)
    generated_len = len(generated_ids)
    has_eos = im_end_id in generated_ids.tolist()

    # Clean response for display
    clean_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(f"A: {clean_response}")
    print(f"   [tokens: {generated_len}, stopped: {'YES (EOS)' if has_eos else 'NO (hit max)'}]")

print("\n" + "="*70)
print("GREEDY COMPARISON (to show difference)")
print("="*70)

# Compare with greedy for "2+2"
prompt = "What is 2 + 2?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

input_len = inputs.shape[1]

# Greedy (like callback) - with correct stop token
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
with torch.no_grad():
    outputs_greedy = model.generate(
        input_ids=inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=im_end_id,  # FIXED
    )

generated_greedy = outputs_greedy[0][input_len:]
response_greedy = tokenizer.decode(generated_greedy, skip_special_tokens=True).strip()

print(f"\nQ: {prompt}")
print(f"Greedy (do_sample=False): {response_greedy[:200]}...")
print(f"   [tokens: {len(generated_greedy)}]")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

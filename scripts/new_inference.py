from unsloth import FastLanguageModel

# Load Stage 1 model
model, tokenizer = FastLanguageModel.from_pretrained(
  "./checkpoints/stage1/merged",
  max_seq_length=2048,
)

# Set to inference mode
FastLanguageModel.for_inference(model)

# Test chat
messages = [
  {"role": "user", "content": "What is the capital of France?"}
]

inputs = tokenizer.apply_chat_template(
  messages,
  tokenize=True,
  add_generation_prompt=True,
  return_tensors="pt"
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""
Simple TARS server - transformers + peft only
Run: pip install transformers peft accelerate flask && python simple_server.py
"""
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = Flask(__name__)

# Globals
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Loading adapter...")
    model = PeftModel.from_pretrained(base, "shekkari21/qwen-fc-sft-stage3")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    print("Ready!")

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 256)

    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=im_end
        )

    gen = out[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(gen, skip_special_tokens=True).strip()

    return jsonify({
        "choices": [{"message": {"role": "assistant", "content": response}}]
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8000)

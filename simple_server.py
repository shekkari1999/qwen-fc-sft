"""
TARS server - supports all 3 training stages
Run: pip install transformers peft accelerate uvicorn fastapi && python simple_server.py
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import torch
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn

models = {}
tokenizer = None

ADAPTERS = {
    "stage1": "shekkari21/qwen-fc-sft-stage1",
    "stage2": "shekkari21/qwen-fc-sft-stage2",
    "tars": "shekkari21/qwen-fc-sft-stage3"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global models, tokenizer

    print("Loading base model (Qwen2.5-3B)...")
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)

    # Load all 3 adapters
    for name, adapter_path in ADAPTERS.items():
        print(f"Loading {name} adapter...")
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
        models[name] = model
        # Reload base for next adapter
        if name != "tars":
            base = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-3B",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

    print("\n" + "="*50)
    print("   TARS Ready! http://0.0.0.0:8000")
    print("   Models: stage1, stage2, tars")
    print("="*50 + "\n")
    yield

app = FastAPI(lifespan=lifespan)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "tars"
    messages: List[Message]
    max_tokens: int = 256
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.95

@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": k} for k in ADAPTERS.keys()]}

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    model_name = req.model if req.model in models else "tars"
    model = models[model_name]

    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature if req.temperature and req.temperature > 0 else None,
            top_p=req.top_p,
            do_sample=req.temperature is not None and req.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=im_end
        )

    gen = out[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(gen, skip_special_tokens=True).strip()

    return {
        "model": model_name,
        "choices": [{"message": {"role": "assistant", "content": response}}]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "models": list(models.keys())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

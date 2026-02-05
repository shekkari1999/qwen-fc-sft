"""
TARS - RunPod Setup & Server
Upload this single file to RunPod and run: python setup_runpod.py
Uses transformers + peft (no unsloth dependency issues)
"""
import subprocess
import sys

# Install dependencies
print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "pydantic", "accelerate", "peft"])

# Now import and run server
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
import time
import uvicorn
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

models = {}
tokenizer = None

AVAILABLE_MODELS = {
    "stage1": "shekkari21/qwen-fc-sft-stage1",
    "stage2": "shekkari21/qwen-fc-sft-stage2",
    "tars": "shekkari21/qwen-fc-sft-stage3"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models, tokenizer

    print("\n" + "="*50)
    print("   TARS - Loading Model")
    print("="*50 + "\n")

    # Load base model
    print("Loading base model (Qwen2.5-3B)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)

    # Load LoRA adapter
    print("Loading TARS adapter (stage3)...")
    model = PeftModel.from_pretrained(base_model, "shekkari21/qwen-fc-sft-stage3")
    model.eval()
    models["tars"] = model

    print("\n" + "="*50)
    print("   TARS Ready!")
    print("   API: http://0.0.0.0:8000")
    print("="*50 + "\n")

    yield


app = FastAPI(title="TARS API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "tars"
    messages: List[Message]
    temperature: float = 0.1
    max_tokens: int = 256
    top_p: float = 0.95


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": k, "object": "model", "owned_by": "user"} for k in AVAILABLE_MODELS]
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": list(models.keys())}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    global tokenizer

    start_time = time.time()

    # Get model (default to tars)
    model = models.get(request.model, models["tars"])
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Format messages using Qwen chat template
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature if request.temperature > 0 else None,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=im_end_id,
        )

    # Decode
    generated = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    completion_tokens = len(generated)

    elapsed = time.time() - start_time

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "elapsed_time": round(elapsed, 2),
            "tokens_per_second": round(completion_tokens / elapsed, 1) if elapsed > 0 else 0
        }
    }


if __name__ == "__main__":
    print("\n" + "="*50)
    print("   TARS - Function Calling Assistant")
    print("   Starting server...")
    print("="*50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)

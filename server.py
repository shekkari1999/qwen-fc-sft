"""
TARS - Custom Inference Server with Unsloth
Supports LoRA adapters with modules_to_save (lm_head)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import time
import uvicorn
from contextlib import asynccontextmanager

# Global model storage
models = {}
tokenizer = None

AVAILABLE_MODELS = {
    "stage1": "shekkari21/qwen-fc-sft-stage1",
    "stage2": "shekkari21/qwen-fc-sft-stage2",
    "tars": "shekkari21/qwen-fc-sft-stage3",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    global models, tokenizer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    print("Loading models...")

    # Load TARS (stage3) by default - others loaded on demand
    print("  Loading tars (stage3)...")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=AVAILABLE_MODELS["tars"],
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tok, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)
    models["tars"] = model
    print("  Loaded tars!")

    yield

    # Cleanup
    models.clear()


app = FastAPI(title="TARS API", lifespan=lifespan)

# CORS for Gradio
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
    stream: bool = False


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "user"


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "tars", "object": "model", "owned_by": "user"},
            {"id": "stage1", "object": "model", "owned_by": "user"},
            {"id": "stage2", "object": "model", "owned_by": "user"},
        ]
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": list(models.keys())}


def load_model_if_needed(model_name: str):
    """Lazy load models on demand"""
    global models, tokenizer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    if model_name not in models:
        if model_name not in AVAILABLE_MODELS:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        print(f"Loading {model_name}...")
        model, tok = FastLanguageModel.from_pretrained(
            model_name=AVAILABLE_MODELS[model_name],
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        tokenizer = get_chat_template(tok, chat_template="qwen-2.5")
        FastLanguageModel.for_inference(model)
        models[model_name] = model
        print(f"Loaded {model_name}!")

    return models[model_name]


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    global tokenizer

    start_time = time.time()

    # Get model
    model = load_model_if_needed(request.model)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Format messages
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Tokenize
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    prompt_tokens = inputs.shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature if request.temperature > 0 else None,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=im_end_id,
        )

    # Decode
    generated = outputs[0][inputs.shape[1]:]
    response_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    completion_tokens = len(generated)

    elapsed = time.time() - start_time

    return ChatResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "elapsed_time": round(elapsed, 2),
            "tokens_per_second": round(completion_tokens / elapsed, 1) if elapsed > 0 else 0
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

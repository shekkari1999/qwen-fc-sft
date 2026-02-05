"""
Merge LoRA adapter into base model and push to HuggingFace
Run on RunPod: python merge_and_push.py
"""
import subprocess
import sys

# Install minimal deps
print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "transformers", "peft", "accelerate", "huggingface_hub"])

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login

# Config
BASE_MODEL = "Qwen/Qwen2.5-3B"
ADAPTER = "shekkari21/qwen-fc-sft-stage3"
OUTPUT_REPO = "shekkari21/tars-3b-merged"

print("\n" + "="*50)
print("   TARS - Merge Adapter")
print("="*50)

# Login to HF (will prompt for token if not logged in)
print("\nLogging into HuggingFace...")
login()

# Load base model (full precision for merging)
print(f"\nLoading base model: {BASE_MODEL}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,  # float16 for merge
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# Load adapter
print(f"\nLoading adapter: {ADAPTER}")
model = PeftModel.from_pretrained(base_model, ADAPTER)

# Merge adapter into base model
print("\nMerging adapter into base model...")
model = model.merge_and_unload()

# Save locally first
print("\nSaving merged model locally...")
model.save_pretrained("./tars-3b-merged", safe_serialization=True)
tokenizer.save_pretrained("./tars-3b-merged")

# Push to HuggingFace
print(f"\nPushing to HuggingFace: {OUTPUT_REPO}")
model.push_to_hub(OUTPUT_REPO, safe_serialization=True)
tokenizer.push_to_hub(OUTPUT_REPO)

print("\n" + "="*50)
print(f"   Done! Merged model at:")
print(f"   https://huggingface.co/{OUTPUT_REPO}")
print("="*50)

print("\nNow you can serve with vLLM:")
print(f"  vllm serve {OUTPUT_REPO}")

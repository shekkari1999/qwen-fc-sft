# TARS Deployment on RunPod

Deploy your fine-tuned TARS model with vLLM on RunPod.

## Step 1: Merge Adapter (One-time)

Run on any GPU machine:

```bash
git clone https://github.com/shekkari1999/qwen-fc-sft.git
cd qwen-fc-sft
python merge_and_push.py
```

This merges the LoRA adapter into the base model and pushes to `shekkari21/tars-3b-merged`.

---

## Step 2: Deploy vLLM on RunPod

### Quick Start

1. Go to [RunPod](https://runpod.io) → **Pods** → **+ Deploy**
2. Select GPU: **RTX 4090** (~$0.40/hr) or **RTX 3090** (~$0.30/hr)
3. Template: **vLLM**
4. Set start command:
   ```
   --host 0.0.0.0 --port 8000 --model shekkari21/tars-3b-merged --dtype bfloat16 --gpu-memory-utilization 0.9 --max-model-len 2048
   ```
5. Deploy

### Access API

Once running (~2 min), your endpoint:
```
https://{POD_ID}-8000.proxy.runpod.net
```

### Test

```bash
curl https://{POD_ID}-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "shekkari21/tars-3b-merged",
    "messages": [
      {"role": "system", "content": "You are TARS with tools: get_weather, calculate, search_web. Use <tool_call> when needed."},
      {"role": "user", "content": "What is the weather in Tokyo?"}
    ]
  }'
```

---

## Step 3: Connect Gradio UI (Optional)

Update `app.py` with your RunPod URL:
```python
VLLM_URL = "https://{POD_ID}-8000.proxy.runpod.net/v1/chat/completions"
```

Run locally:
```bash
python app.py
```

Open `http://localhost:7860`

---

## Cost Estimates

| GPU | VRAM | $/hr | Monthly (24/7) |
|-----|------|------|----------------|
| RTX 3090 | 24GB | ~$0.30 | ~$220 |
| RTX 4090 | 24GB | ~$0.40 | ~$290 |

---

## Troubleshooting

**"Model not found"**: Wait for model download (~2 min on first start)

**"CUDA OOM"**: Reduce `--max-model-len` to 1024

**Slow first request**: Model loading, subsequent requests are fast

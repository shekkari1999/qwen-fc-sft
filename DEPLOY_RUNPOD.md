# TARS Deployment on RunPod

Deploy your fine-tuned TARS model on RunPod.

> **Note:** vLLM doesn't support LoRA adapters with `modules_to_save` (needed for lm_head training). Use the custom Unsloth server instead.

---

## Recommended: Custom Unsloth Server

### 1. Create a Pod

1. Go to [RunPod](https://runpod.io) → **Pods** → **+ Deploy**
2. Select GPU: **RTX 4090** (24GB) or **RTX 3090**
3. Template: **RunPod PyTorch 2.1** (NOT vLLM)
4. Container disk: **30GB**
5. Deploy

### 2. Connect via SSH or Web Terminal

```bash
# Get SSH command from RunPod dashboard
ssh root@{POD_IP} -p {PORT}
```

### 3. Install Dependencies

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install fastapi uvicorn pydantic gradio requests
```

### 4. Upload Files

Upload `server.py` and `app.py` via:
- RunPod file browser, or
- `scp server.py app.py root@{POD_IP}:{PORT}:~/`

### 5. Start Server

```bash
# Terminal 1: Start API server
python server.py

# Terminal 2: Start Gradio UI (optional)
python app.py
```

### 6. Access

- API: `https://{POD_ID}-8000.proxy.runpod.net`
- UI: `https://{POD_ID}-7860.proxy.runpod.net`

---

## Alternative: vLLM (Requires Merged Model)

If you want to use vLLM, you must merge the adapter first:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("shekkari21/qwen-fc-sft-stage3", ...)
model.push_to_hub_merged("shekkari21/tars-3b-merged", tokenizer, save_method="merged_16bit")
```

Then use vLLM with `--model shekkari21/tars-3b-merged` (no LoRA flags).

---

## DEPRECATED: Quick Start (RunPod vLLM Template)

### 1. Create a Pod

1. Go to [RunPod](https://runpod.io) → **Pods** → **+ Deploy**
2. Select GPU: **RTX 3090** or **RTX 4090** (24GB VRAM) or **A40** (48GB)
3. Choose template: **RunPod vLLM**
4. Set environment variables:
   ```
   MODEL_NAME=Qwen/Qwen2.5-3B
   EXTRA_ARGS=--enable-lora --lora-modules stage1=shekkari21/qwen-fc-sft-stage1 stage2=shekkari21/qwen-fc-sft-stage2 tars=shekkari21/qwen-fc-sft-stage3 --max-lora-rank 16
   ```
5. Deploy

### 2. Access the API

Once running, your endpoint will be:
```
https://{POD_ID}-8000.proxy.runpod.net/v1/chat/completions
```

Test with curl:
```bash
curl https://{POD_ID}-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tars",
    "messages": [
      {"role": "system", "content": "You are TARS, a helpful assistant with tools: get_weather, calculate, search_web."},
      {"role": "user", "content": "What is the weather in Paris?"}
    ]
  }'
```

---

## Full Deployment (With Gradio UI)

### 1. Create Pod with PyTorch Template

1. Go to RunPod → **Pods** → **+ Deploy**
2. Select GPU: **RTX 4090** (24GB) recommended
3. Template: **RunPod PyTorch 2.1**
4. Container disk: **50GB** minimum
5. Deploy

### 2. Connect to Pod

```bash
# SSH into pod (get command from RunPod dashboard)
ssh root@{POD_IP} -p {PORT}
```

### 3. Install Dependencies

```bash
# Clone repo
git clone https://github.com/yourusername/qwen-fc-sft.git
cd qwen-fc-sft

# Install requirements
pip install vllm gradio openai requests

# Or create requirements.txt
pip install -r requirements.txt
```

### 4. Start vLLM Server (Background)

```bash
chmod +x serve.sh
nohup ./serve.sh > vllm.log 2>&1 &

# Check logs
tail -f vllm.log
```

Wait for: `Uvicorn running on http://0.0.0.0:8000`

### 5. Start Gradio UI

```bash
python app.py
```

Access UI at: `https://{POD_ID}-7860.proxy.runpod.net`

---

## RunPod Serverless (Production)

For production with auto-scaling:

### 1. Create Serverless Endpoint

1. Go to **Serverless** → **+ New Endpoint**
2. Select **vLLM Worker**
3. Configure:
   - **Model**: `Qwen/Qwen2.5-3B`
   - **Max Workers**: 3
   - **GPU**: RTX 4090
4. Add environment variable:
   ```
   EXTRA_ARGS=--enable-lora --lora-modules tars=shekkari21/qwen-fc-sft-stage3 --max-lora-rank 16
   ```

### 2. Call Endpoint

```python
import requests

RUNPOD_API_KEY = "your_api_key"
ENDPOINT_ID = "your_endpoint_id"

response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
    json={
        "input": {
            "model": "tars",
            "messages": [
                {"role": "system", "content": "You are TARS with tools..."},
                {"role": "user", "content": "Calculate 15% of 2500"}
            ],
            "max_tokens": 256
        }
    }
)
print(response.json())
```

---

## Docker Deployment (Custom)

### Dockerfile

```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# Install dependencies
RUN pip install vllm gradio openai requests

# Copy app files
COPY app.py serve.sh ./
RUN chmod +x serve.sh

# Download models on build (optional - speeds up cold start)
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-3B'); \
    snapshot_download('shekkari21/qwen-fc-sft-stage1'); \
    snapshot_download('shekkari21/qwen-fc-sft-stage2'); \
    snapshot_download('shekkari21/qwen-fc-sft-stage3')"

# Start script
COPY start.sh ./
RUN chmod +x start.sh

CMD ["./start.sh"]
```

### start.sh

```bash
#!/bin/bash
# Start vLLM in background
./serve.sh &
# Wait for vLLM to be ready
sleep 30
# Start Gradio
python app.py
```

### Build and Push

```bash
docker build -t your-dockerhub/tars-vllm:latest .
docker push your-dockerhub/tars-vllm:latest
```

Then use this image in RunPod custom template.

---

## Cost Estimates

| GPU | VRAM | $/hr | Best For |
|-----|------|------|----------|
| RTX 3090 | 24GB | ~$0.30 | Development |
| RTX 4090 | 24GB | ~$0.40 | Production |
| A40 | 48GB | ~$0.80 | Multiple models |
| A100 | 80GB | ~$1.50 | High throughput |

**Estimated monthly cost** (24/7 operation):
- RTX 4090: ~$290/month
- With serverless (auto-scale): ~$50-100/month (depending on traffic)

---

## API Endpoints

Once vLLM is running:

### Chat Completions
```
POST /v1/chat/completions
```

### Models List
```
GET /v1/models
```

### Health Check
```
GET /health
```

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `--gpu-memory-utilization` to 0.8
- Use `--max-model-len 1024`

### "LoRA adapter not found"
- Check HuggingFace repo is public
- Verify repo names in serve.sh

### Slow first request
- First request downloads adapters
- Use Docker pre-download for faster cold starts

### Gradio not accessible
- Ensure port 7860 is exposed
- Check RunPod proxy URL format

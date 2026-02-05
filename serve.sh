#!/bin/bash
# TARS - vLLM Server with Multiple LoRA Adapters
# Serves all 3 training stages simultaneously

echo "=========================================="
echo "   TARS - vLLM Server"
echo "   Base: Qwen2.5-3B"
echo "   LoRA Adapters: stage1, stage2, tars"
echo "=========================================="

# Configuration
BASE_MODEL="Qwen/Qwen2.5-3B"
PORT=8000
MAX_LORA_RANK=16

# LoRA adapters from HuggingFace
STAGE1_ADAPTER="shekkari21/qwen-fc-sft-stage1"
STAGE2_ADAPTER="shekkari21/qwen-fc-sft-stage2"
STAGE3_ADAPTER="shekkari21/qwen-fc-sft-stage3"

echo ""
echo "Starting vLLM server..."
echo "  - Port: $PORT"
echo "  - Base model: $BASE_MODEL"
echo "  - Adapters: stage1, stage2, tars"
echo ""

vllm serve $BASE_MODEL \
    --enable-lora \
    --lora-modules stage1=$STAGE1_ADAPTER \
                   stage2=$STAGE2_ADAPTER \
                   tars=$STAGE3_ADAPTER \
    --max-lora-rank $MAX_LORA_RANK \
    --port $PORT \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048

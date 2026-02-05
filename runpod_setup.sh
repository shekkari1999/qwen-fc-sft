#!/bin/bash
# TARS - RunPod Setup Script
# Run this after SSH into a RunPod pod

echo "=========================================="
echo "   TARS - RunPod Setup"
echo "=========================================="

# Install dependencies
echo "Installing dependencies..."
pip install unsloth fastapi uvicorn pydantic gradio requests

# Clone repo (or upload files)
echo ""
echo "Downloading server files..."
curl -O https://raw.githubusercontent.com/YOUR_REPO/qwen-fc-sft/main/server.py
curl -O https://raw.githubusercontent.com/YOUR_REPO/qwen-fc-sft/main/app.py

# Or if you uploaded manually, skip the curl commands

echo ""
echo "=========================================="
echo "   Starting TARS Server"
echo "=========================================="
echo ""
echo "Run in separate terminals:"
echo "  Terminal 1: python server.py"
echo "  Terminal 2: python app.py"
echo ""
echo "Or run server in background:"
echo "  nohup python server.py > server.log 2>&1 &"
echo "  python app.py"
echo ""

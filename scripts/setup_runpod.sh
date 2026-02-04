#!/bin/bash
# Setup script for RunPod training environment
# Run this first after SSH into RunPod

set -e

echo "=============================================="
echo "Setting up RunPod for Qwen SFT Training"
echo "=============================================="

# Update pip
pip install --upgrade pip

# Install Unsloth (this handles most dependencies)
echo ""
echo "Installing Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install additional requirements
echo ""
echo "Installing additional packages..."
pip install datasets trl peft accelerate bitsandbytes xformers

# Create directories
echo ""
echo "Creating directories..."
mkdir -p checkpoints/stage1 checkpoints/stage2

# Verify installation
echo ""
echo "Verifying installation..."
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
python -c "from trl import SFTTrainer; print('TRL OK')"
python -c "import torch; print(f'PyTorch OK - CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=============================================="
echo "Setup complete! Ready for training."
echo ""
echo "Next steps:"
echo "1. Upload your data to data/stage1_chat/ and data/stage2_fc/"
echo "2. Run: python scripts/train_stage1.py"
echo "3. Run: python scripts/train_stage2.py"
echo "=============================================="

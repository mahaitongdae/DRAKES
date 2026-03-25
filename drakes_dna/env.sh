#!/bin/bash
set -e

# Replace 'my-storage' with the actual name of your Lambda filesystem
export PERSISTENT_ROOT="/workspace"

# Move Python interpreters here
export UV_PYTHON_INSTALL_DIR="$PERSISTENT_ROOT/.uv/python"

# Move the cache (highly recommended for speed and space)
export UV_CACHE_DIR="$PERSISTENT_ROOT/.uv/cache"

# Move tool binaries
export UV_TOOL_DIR="$PERSISTENT_ROOT/.uv/tools"

# Create venv (use --clear to recreate if exists)
uv venv ./.venv --python=3.9

# Activate venv
source ./.venv/bin/activate

# PyTorch with CUDA 12.1
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
uv pip install packaging ninja transformers datasets omegaconf

# Jupyter kernel
uv pip install ipykernel
python -m ipykernel install --user --name sedd --display-name "Python (sedd)"

# Hydra
uv pip install hydra-core hydra-submitit-launcher

# For mdlm
uv pip install lightning timm rich

# Additional
uv pip install scipy wandb

# Install gReLU package from GitHub
uv pip install git+https://github.com/Genentech/gReLU.git@v1.0.2


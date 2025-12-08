#!/usr/bin/env bash
set -euo pipefail

# === Lambda Cloud Instance Setup (One-Time) ===
# This script prepares a fresh Lambda Cloud instance for the parallel-decoder-transformer project.
# Run this ONCE when you first create the instance, then use scripts/lambda_deploy.sh to launch training.
#
# What this script does:
# 1. Installs system dependencies (apt packages)
# 2. Installs uv (Python package manager)
# 3. Detects CUDA and configures PyTorch
# 4. Creates Python 3.12 virtual environment
# 5. Installs all project dependencies
# 6. Downloads GPT-OSS-20B model weights (~40GB)
# 7. Verifies GPU availability
#
# Usage:
#   cd /path/to/parallel-decoder-transformer
#   bash scripts/setup_lambda_gpu.sh
#
# After this completes, run scripts/lambda_deploy.sh to start training.

echo "Starting environment setup for parallel-decoder-transformer..."

# 1. Determine project root directory
# The script is in <root>/scripts/, so we go up one level.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
echo "--> Project root detected at: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# 2. Install base system dependencies
# Using sudo to ensure permissions. apt-get update is run first to refresh lists.
echo "--> Updating package lists and installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip unzip tmux wget curl

# 3. Ensure 'uv' is installed
# uv is used for fast Python environment and package management.
if ! command -v uv >/dev/null 2>&1; then
  echo "--> 'uv' not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs to ~/.local/bin, add it to PATH
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "--> 'uv' is already installed."
fi
# Ensure uv is in PATH regardless of installation method
export PATH="$HOME/.local/bin:$PATH"

# 4. Detect CUDA version and select PyTorch wheel index
# We map the driver's CUDA version to a compatible, official PyTorch wheel index.
CUDA_INDEX=""
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VER_RAW=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+') || true
  if [[ -n "${CUDA_VER_RAW:-}" ]]; then
    CUDA_MAJOR=${CUDA_VER_RAW%%.*}
    # Lambda Stack 24.04 provides CUDA 12.8 driver. 
    # We target cu124 wheels for PyTorch 2.5+ compatibility.
    if [[ "$CUDA_MAJOR" -ge 12 ]]; then
      CUDA_INDEX="https://download.pytorch.org/whl/cu124"
    elif [[ "$CUDA_MAJOR" -eq 11 ]]; then
      CUDA_INDEX="https://download.pytorch.org/whl/cu118"
    fi
  fi
fi

echo "Detected CUDA driver: ${CUDA_VER_RAW:-none}"
if [[ -n "$CUDA_INDEX" ]]; then
  echo "Using PyTorch index for CUDA: $CUDA_INDEX"
else
  echo "WARNING: No matching CUDA index detected. Will attempt to use CPU-only wheels."
fi

# 5. Install Python and project dependencies
# Lambda Stack Ubuntu 24.04 has Python 3.12 by default.
# We initialize the venv using the system python3.12 to leverage Lambda's pre-installed libs if possible,
# but `uv sync` will effectively isolate us.
echo "--> Creating .venv with system Python 3.12..."
uv venv .venv --python 3.12

# Pre-install build dependencies to avoid issues during sync
echo "--> Pre-installing build dependencies..."
uv pip install setuptools packaging wheel

echo "--> Syncing all project dependencies (base, data, test) into $PROJECT_ROOT/.venv ..."
if [[ -n "$CUDA_INDEX" ]]; then
  # Force the index URL for PyTorch to ensure we get the cu124 build
  UV_INDEX_URL="$CUDA_INDEX" uv sync --frozen --extra data --extra test --extra gpu
else
  uv sync --frozen --extra data --extra test
fi

# 6. Install FlashAttention-2 (Architecture-Specific)
# Detect GPU architecture and compile for specific compute capability
echo "--> Detecting GPU architecture for FlashAttention-2..."

GPU_ARCH=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  echo "    Detected GPU: $GPU_NAME"
  
  # Determine compute capability based on GPU name
  if [[ "$GPU_NAME" == *"B200"* ]] || [[ "$GPU_NAME" == *"B100"* ]]; then
    GPU_ARCH="10.0"  # Blackwell architecture
    echo "    Architecture: Blackwell (SM 10.0)"
  elif [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"H200"* ]]; then
    GPU_ARCH="9.0"   # Hopper architecture
    echo "    Architecture: Hopper (SM 9.0)"
  elif [[ "$GPU_NAME" == *"A100"* ]] || [[ "$GPU_NAME" == *"A40"* ]]; then
    GPU_ARCH="8.0"   # Ampere architecture
    echo "    Architecture: Ampere (SM 8.0)"
  else
    echo "    WARNING: Unknown GPU, will compile for multiple architectures (slower)"
  fi
fi

# Install FlashAttention-2 from source with architecture-specific compilation
echo "--> Compiling FlashAttention-2 from source (optimized for detected GPU)..."
if [[ -n "$GPU_ARCH" ]]; then
  # Compile ONLY for the detected architecture (faster build, smaller binary)
  echo "    Compiling for compute capability: $GPU_ARCH"
  
  # Convert compute capability to CMake format (e.g., 10.0 -> 100, 9.0 -> 90)
  CMAKE_ARCH=$(echo "$GPU_ARCH" | tr -d '.')
  
  echo "    Downloading FlashAttention source to patch setup.py..."
  cd /tmp
  rm -rf flash-attention
  git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  
  # Patch setup.py to only compile for the detected architecture
  echo "    Patching setup.py to compile only for SM $CMAKE_ARCH..."
  
  # Replace ALL gencode lines with a single target architecture
  # This works by finding lines with "-gencode" and replacing the entire gencode_archs list
  python3 << PYTHON_PATCH
import re

target_arch = "${CMAKE_ARCH}"

with open('setup.py', 'r') as f:
    content = f.read()

# Replace the gencode_archs list with single architecture
# Pattern matches the entire list assignment
pattern = r'gencode_archs\s*=\s*\[[^\]]+\]'
replacement = f'gencode_archs = ["-gencode", f"arch=compute_{target_arch},code=sm_{target_arch}"]'

content = re.sub(pattern, replacement, content)

with open('setup.py', 'w') as f:
    f.write(content)
    
print(f"Patched setup.py to compile only for SM {target_arch}")
PYTHON_PATCH
  
  # Install from the patched source
  echo "    Installing from patched source (single architecture only)..."
  cd "$PROJECT_ROOT"
  MAX_JOBS=8 uv pip install /tmp/flash-attention --no-build-isolation
  
  # Clean up
  rm -rf /tmp/flash-attention
else
  # Fallback: compile for common architectures (no patch)
  echo "    Compiling for multiple architectures (will be slow)..."
  MAX_JOBS=8 uv pip install flash-attn --no-build-isolation
fi

echo "    FlashAttention-2 compilation complete!"

# 7. Verify the installation
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
echo "--> Verifying PyTorch GPU availability and DDP backend..."
"$VENV_PYTHON" - <<'PY'
import torch
import torch.distributed as dist

print("=" * 50)
print("ENVIRONMENT VERIFICATION")
print("=" * 50)

# PyTorch and CUDA
print(f"\nPyTorch version: {torch.__version__}")
cuda_version = getattr(torch.version, "cuda", "N/A")
print(f"PyTorch CUDA version: {cuda_version}")

# GPU Detection
is_available = torch.cuda.is_available()
print(f"\nCUDA available: {is_available}")
if is_available:
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s):")
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  - Device {i}: {name} ({total_mem:.1f} GB)")
    
    # DDP Backend Check
    print(f"\nNCCL backend available: {dist.is_nccl_available()}")
    
    # Training readiness
    if device_count == 8:
        print("\n✅ READY: 8 GPUs detected (optimal for DDP training)")
    elif device_count > 0:
        print(f"\n⚠️  WARNING: {device_count} GPU(s) detected. Training is configured for 8 GPUs.")
        print("   Update configs/gpt_oss_transfer_production.yaml if using different GPU count.")
    
    # Check for Flash Attention
    try:
        from flash_attn import flash_attn_func
        print("\n✅ FlashAttention-2 installed successfully")
    except ImportError:
        print("\n❌ FlashAttention-2 NOT found (will cause training failure)")
else:
    print("\n❌ CRITICAL: No CUDA-enabled GPUs detected. Training will fail.")

print("=" * 50)
PY

# 8. Download GPT-OSS-20B model weights
# This is required for running the model.
echo "--> Downloading GPT-OSS-20B model weights..."
# Ensure the download script is executable and run it from the project root
chmod +x "$PROJECT_ROOT/scripts/download_gpt_oss_20b.sh"
"$PROJECT_ROOT/scripts/download_gpt_oss_20b.sh"

# 9. Create required directory structure
echo "--> Creating required directory structure..."
mkdir -p "$PROJECT_ROOT/data/processed/pdt_10k_gpt41"
mkdir -p "$PROJECT_ROOT/data/teacher_cache"
mkdir -p "$PROJECT_ROOT/experiments/gpt_oss"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/temp-outputs"

# 10. Verify critical files and provide post-setup instructions
echo ""
echo "=========================================="
echo "Setup complete! Environment is ready."
echo "=========================================="
echo ""
echo "NEXT STEPS (Manual):"
echo ""
echo "✅ H100 FILES ALREADY TRANSFERRED (Dec 2025)"
echo "   Location: /home/ubuntu/backup/h100_transfer_20251206_122158/"
echo "   Contains: wandb.txt, checkpoints, teacher_cache, dataset (64 GB total)"
echo "   See ASSIST.md for deployment commands."
echo ""
echo "# COMMENTED OUT - Already completed transfer"
# echo "1. Transfer data from H100 server (192.222.53.40) - Total: ~9 GB"
# echo "   # Pull wandb.txt (secrets - 40 bytes)"
# echo "   rsync -avz --progress -e \"ssh -i ~/.ssh/personal_key\" \\"
# echo "     ubuntu@192.222.53.40:/home/ubuntu/parallel-decoder-transformer/wandb.txt \\"
# echo "     $PROJECT_ROOT/"
# echo ""
# echo "   # Pull teacher cache (54 MB compressed, saves 8-12 hours preprocessing)"
# echo "   rsync -avz --progress -e \"ssh -i ~/.ssh/personal_key\" \\"
# echo "     ubuntu@192.222.53.40:/home/ubuntu/teacher_cache_pdt10k.tar.gz \\"
# echo "     $PROJECT_ROOT/"
# echo ""
# echo "   # Pull checkpoints (8.9 GB compressed - 3 adapter checkpoints)"
# echo "   rsync -avz --progress -e \"ssh -i ~/.ssh/personal_key\" \\"
# echo "     ubuntu@192.222.53.40:/home/ubuntu/checkpoints_50k_h100.tar.gz \\"
# echo "     $PROJECT_ROOT/"
# echo ""
# echo "   # Expected transfer time: ~5-10 minutes depending on network speed"
# echo ""
# echo "2. Download training dataset (~56 GB):"
# echo "   # Place these files in: $PROJECT_ROOT/data/processed/pdt_10k_gpt41/"
# echo "   - kd_train.jsonl (45 GB)"
# echo "   - kd_validation.jsonl (5.6 GB)"
# echo "   - kd_test.jsonl (5.0 GB)"
# echo ""
# echo "3. (Later) Extract teacher cache:"
# echo "   tar -xzf $PROJECT_ROOT/teacher_cache_pdt10k.tar.gz -C $PROJECT_ROOT/"
# echo ""
# echo "4. (Later) Extract checkpoints:"
# echo "   mkdir -p $PROJECT_ROOT/experiments/gpt_oss/gpt-oss-8xH100-50000steps"
# echo "   tar -xzf $PROJECT_ROOT/checkpoints_50k_h100.tar.gz \\"
# echo "     -C $PROJECT_ROOT/experiments/gpt_oss/gpt-oss-8xH100-50000steps/"
# echo ""
# echo "5. Verify all transferred files:"
# echo "   ls -lh $PROJECT_ROOT/wandb.txt"
# echo "   ls -lh $PROJECT_ROOT/teacher_cache_pdt10k.tar.gz"
# echo "   ls -lh $PROJECT_ROOT/checkpoints_50k_h100.tar.gz"
# echo "   cat $PROJECT_ROOT/wandb.txt  # Should show WandB API key"
# echo ""
echo "1. (B200 ONLY) Update config for increased memory:"
echo "   # Edit: configs/gpt_oss_transfer_production.yaml"
echo "   # Change: resume_from_checkpoint: true  (B200 has 180GB, can resume)"
echo "   # Change: batch_size: 2  (optional, faster training with more memory)"
echo "   # Add Stage 4 & 5 to stage_schedule: [0, 3750, 10000, 17500, 22500, 40000]"
echo "   # Stage 4 (22500-40000) enables full trunk fine-tuning"
echo ""
echo "2. Make launch scripts executable:"
echo "   chmod +x $PROJECT_ROOT/scripts/*.sh"
echo ""
echo "3. When ready to train, launch in tmux session:"
echo "   tmux new-session -s training"
echo "   # Inside tmux session, export WandB key and launch DDP training:"
echo "   export WANDB_API_KEY=\$(cat wandb.txt)"
echo "   uv run torchrun --nproc_per_node=8 scripts/train_wandb.py --config configs/gpt_oss_transfer_production.yaml"
echo "   # Detach from tmux: Ctrl+B then D"
echo ""
echo "=========================================="

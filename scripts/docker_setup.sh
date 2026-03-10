#!/bin/bash
# ============================================================
# FSAD Docker Setup — 다른 서버에서 실행
# ============================================================
# Usage:
#   1. 이 프로젝트를 서버에 clone/copy
#   2. DATA_PATH를 데이터셋 경로로 설정
#   3. bash scripts/docker_setup.sh
# ============================================================

set -e

# ---------- Config (서버별 수정) ----------
export DATA_PATH="${DATA_PATH:-/data}"          # MVTec-AD, VisA 등 데이터 경로
export CACHE_PATH="${CACHE_PATH:-$HOME/.cache}" # pretrained model cache

# ---------- Pre-flight checks ----------
echo "[1/4] Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "ERROR: docker not found. Install: https://docs.docker.com/engine/install/"
    exit 1
fi

if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    if ! command -v nvidia-container-toolkit &> /dev/null && \
       ! dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then
        echo "WARNING: nvidia-container-toolkit not detected."
        echo "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        echo "Continuing anyway (may fail at GPU access)..."
    fi
fi

echo "  Docker: $(docker --version)"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  DATA_PATH:  $DATA_PATH"
echo "  CACHE_PATH: $CACHE_PATH"

# ---------- Build ----------
echo ""
echo "[2/4] Building Docker image..."
docker compose build

# ---------- Test ----------
echo ""
echo "[3/4] Verifying setup..."
docker compose run --rm fsad python -c "
import torch
import torchvision
import timm
import sklearn
import cv2
import numpy as np

print(f'PyTorch:      {torch.__version__}')
print(f'TorchVision:  {torchvision.__version__}')
print(f'timm:         {timm.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'OpenCV:       {cv2.__version__}')
print(f'NumPy:        {np.__version__}')
print(f'CUDA:         {torch.version.cuda}')
print(f'cuDNN:        {torch.backends.cudnn.version()}')
print(f'GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'GPU Memory:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else '')
"

# ---------- Done ----------
echo ""
echo "[4/4] Setup complete!"
echo ""
echo "Usage:"
echo "  # Interactive shell"
echo "  docker compose run --rm fsad bash"
echo ""
echo "  # Run experiment (구현 후)"
echo "  docker compose run --rm fsad python main.py --config configs/default.yaml"
echo ""
echo "  # Background container"
echo "  docker compose up -d"
echo "  docker exec -it hun_FSAD bash"

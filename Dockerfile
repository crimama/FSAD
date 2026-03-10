# FSAD Research Project — Few-Shot Anomaly Detection
# Base: PyTorch 2.5 + CUDA 12.1 (runtime compatible with host CUDA 12.2~12.8)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    htop \
    tmux \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Working directory
WORKDIR /workspace/FSAD

# Default command
CMD ["bash"]

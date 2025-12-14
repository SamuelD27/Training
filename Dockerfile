# Identity LoRA Training - FLUX.1-dev
# Based on DeepResearchReport.md (December 2025)
#
# Build: docker build -t lora-flux-trainer:latest -f docker/Dockerfile .
# Run:   docker run --gpus all -it -v /path/to/dataset:/workspace/lora_training/data/subject lora-flux-trainer:latest

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

LABEL maintainer="lora-training"
LABEL description="FLUX.1-dev Identity LoRA Training Environment"
LABEL version="1.0.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    tmux \
    nano \
    vim \
    jq \
    rsync \
    htop \
    nvtop \
    unzip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libffi-dev \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip and install uv for fast dependency management
RUN python -m pip install --upgrade pip setuptools wheel uv

# Set working directory
WORKDIR /workspace

# Install PyTorch (pinned to 2.5.1 + CUDA 12.1)
RUN uv pip install --system \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core training dependencies (pinned versions)
RUN uv pip install --system \
    accelerate==0.25.0 \
    bitsandbytes==0.44.1 \
    transformers==4.46.3 \
    safetensors==0.4.5 \
    diffusers==0.31.0 \
    huggingface-hub==0.26.2

# Install sd-scripts dependencies
RUN uv pip install --system \
    lion-pytorch==0.2.3 \
    prodigyopt==1.0 \
    schedulefree==1.2.6 \
    dadaptation==3.2 \
    pytorch-lightning==2.4.0 \
    tensorboard==2.18.0

# Install dataset analysis dependencies
RUN uv pip install --system \
    Pillow==10.4.0 \
    imagehash==4.3.1 \
    opencv-python-headless==4.10.0.84 \
    numpy==1.26.4

# Install utilities
RUN uv pip install --system \
    pyyaml==6.0.2 \
    toml==0.10.2 \
    tqdm==4.66.5 \
    sentencepiece==0.2.0 \
    ftfy==6.3.1 \
    einops

# Install xformers compatible with torch 2.5.1
RUN uv pip install --system xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121

# Install nvitop for GPU monitoring
RUN uv pip install --system nvitop

# Clone sd-scripts (pinned version)
ENV SDSCRIPTS_VERSION=v0.9.1
RUN git clone https://github.com/kohya-ss/sd-scripts.git /workspace/sd-scripts \
    && cd /workspace/sd-scripts \
    && git checkout ${SDSCRIPTS_VERSION} \
    && uv pip install --system -r requirements.txt

# Copy repository
COPY . /workspace/lora_training

# Create symlink for sd-scripts in third_party
RUN ln -sf /workspace/sd-scripts /workspace/lora_training/third_party/sd-scripts

# Create runtime directories
RUN mkdir -p /workspace/lora_training/output \
    /workspace/lora_training/logs \
    /workspace/lora_training/data/subject/images \
    /workspace/lora_training/data/subject/captions \
    /workspace/lora_training/data/reg/images \
    /workspace/lora_training/data/reg/captions

# Create default accelerate config (non-interactive)
RUN mkdir -p /root/.cache/huggingface/accelerate \
    && echo 'compute_environment: LOCAL_MACHINE\n\
debug: false\n\
distributed_type: "NO"\n\
downcast_bf16: "no"\n\
enable_cpu_affinity: false\n\
gpu_ids: all\n\
machine_rank: 0\n\
main_training_function: main\n\
mixed_precision: bf16\n\
num_machines: 1\n\
num_processes: 1\n\
rdzv_backend: static\n\
same_network: true\n\
tpu_env: []\n\
tpu_use_cluster: false\n\
tpu_use_sudo: false\n\
use_cpu: false' > /root/.cache/huggingface/accelerate/default_config.yaml

# Make scripts executable
RUN chmod +x /workspace/lora_training/docker/start.sh \
    /workspace/lora_training/docker/env.sh \
    /workspace/lora_training/scripts/*.sh 2>/dev/null || true

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch

# Expose tensorboard port
EXPOSE 6006

# Set entrypoint
ENTRYPOINT ["/bin/bash", "/workspace/lora_training/docker/start.sh"]

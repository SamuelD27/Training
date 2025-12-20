# Identity LoRA Training - FLUX.1-dev
# Based on DeepResearchReport.md (December 2025)
#
# Build: docker build -t lora-flux-trainer:latest -f Dockerfile .
# Run:   docker run --gpus all -it -v /path/to/dataset:/workspace/lora_training/data/subject lora-flux-trainer:latest
#
# RunPod Template: This image is designed to work with RunPod's volume mounts.
# Boot scripts are stored in /opt/lora-training/ to survive /workspace overwrites.
#
# GPU Support: CUDA 12.8 + Blackwell (sm_100/sm_120), Hopper (sm_90), Ada (sm_89), Ampere (sm_80/86)

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

LABEL maintainer="lora-training"
LABEL description="FLUX.1-dev Identity LoRA Training Environment"
LABEL version="1.2.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System packages (including SSH server for TCP access)
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
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH server
RUN mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && echo 'root:runpod' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    && sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config \
    && echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB5L7fhROAJ1aZkQs9qksq+Fjf0Qt5pRSNsDyST1dCOD samueldukmedjian@MacBook-Air-de-Samuel.local" > /root/.ssh/authorized_keys \
    && chmod 600 /root/.ssh/authorized_keys \
    && echo "export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:\$PATH" >> /root/.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH" >> /root/.bashrc

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

# Install PyTorch (CUDA 12.8 for Blackwell/Hopper/Ada/Ampere support)
RUN uv pip install --system \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Install core training dependencies (updated for Blackwell support)
RUN uv pip install --system \
    accelerate \
    bitsandbytes \
    transformers \
    safetensors \
    diffusers \
    huggingface-hub

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
    tomli \
    tqdm==4.66.5 \
    sentencepiece==0.2.0 \
    ftfy==6.3.1 \
    einops

# Install xformers (CUDA 12.8 for Blackwell support)
RUN uv pip install --system xformers --index-url https://download.pytorch.org/whl/cu128

# Install nvitop for GPU monitoring
RUN uv pip install --system nvitop

# Install viu (terminal image viewer) for viewing sample images in terminal
RUN curl -sSL https://github.com/atanunq/viu/releases/download/v1.5.0/viu-x86_64-unknown-linux-musl -o /usr/local/bin/viu \
    && chmod +x /usr/local/bin/viu

# Clone sd-scripts (sd3 branch for FLUX.1 support) to /opt so it survives volume mounts
# The sd3 branch contains flux_train_network.py required for FLUX.1 training
ENV SDSCRIPTS_BRANCH=sd3
RUN git clone https://github.com/kohya-ss/sd-scripts.git /opt/sd-scripts \
    && cd /opt/sd-scripts \
    && git checkout ${SDSCRIPTS_BRANCH} \
    && uv pip install --system -r requirements.txt

# Create /opt/lora-training for boot scripts (survives RunPod volume mounts)
RUN mkdir -p /opt/lora-training

# Copy boot scripts to /opt (these survive volume mounts to /workspace)
COPY docker/start.sh /opt/lora-training/start.sh
COPY docker/env.sh /opt/lora-training/env.sh
RUN chmod +x /opt/lora-training/*.sh

# Copy full repo to /opt as backup (will be synced to /workspace on boot)
COPY . /opt/lora-training/repo

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

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch
ENV SDSCRIPTS=/opt/sd-scripts

# Expose ports: SSH (22) and TensorBoard (6006)
EXPOSE 22 6006

# Set entrypoint to /opt path (survives volume mounts)
ENTRYPOINT ["/bin/bash", "/opt/lora-training/start.sh"]

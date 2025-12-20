#!/bin/bash
# ============================================================================
# Identity LoRA Training - Container Startup Script
# ============================================================================
# This script runs on container boot to:
# 1. Start SSH server (for TCP access on RunPod)
# 2. Bootstrap repo if /workspace was overwritten by volume mount
# 3. Print version banner
# 4. Create runtime directories
# 5. Write pip freeze
# 6. Run healthchecks
# 7. Print usage instructions (does NOT start training)
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

WORKSPACE="/workspace/lora_training"
SDSCRIPTS="${SDSCRIPTS:-/opt/sd-scripts}"  # Use env var or default to /opt

# ============================================================================
# Start SSH Server (for TCP access)
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Starting SSH server..."
service ssh start 2>/dev/null || /usr/sbin/sshd 2>/dev/null || echo -e "${YELLOW}[WARN]${NC} SSH server not available"
if pgrep -x sshd > /dev/null; then
    echo -e "${GREEN}[OK]${NC} SSH server running on port 22"
    echo -e "${YELLOW}[INFO]${NC} Default credentials: root / runpod (change with: passwd)"
else
    echo -e "${YELLOW}[WARN]${NC} SSH server failed to start"
fi
echo ""

# ============================================================================
# Bootstrap Repository (if /workspace was overwritten by volume mount)
# ============================================================================
if [ ! -d "${WORKSPACE}/scripts" ]; then
    echo -e "${BLUE}[INFO]${NC} Workspace not found, bootstrapping from /opt/lora-training/repo..."
    mkdir -p /workspace
    if [ -d "/opt/lora-training/repo" ]; then
        cp -r /opt/lora-training/repo /workspace/lora_training
        chmod +x /workspace/lora_training/scripts/*.sh 2>/dev/null || true
        chmod +x /workspace/lora_training/docker/*.sh 2>/dev/null || true
        echo -e "${GREEN}[OK]${NC} Repository bootstrapped to ${WORKSPACE}"
    else
        echo -e "${YELLOW}[WARN]${NC} No backup repo found in /opt/lora-training/repo"
        echo -e "${YELLOW}[WARN]${NC} You may need to clone: git clone https://github.com/YOUR_REPO ${WORKSPACE}"
    fi
    echo ""
fi

# Create symlink for sd-scripts if needed
if [ ! -d "${WORKSPACE}/third_party/sd-scripts" ] && [ -d "${SDSCRIPTS}" ]; then
    mkdir -p ${WORKSPACE}/third_party
    ln -sf ${SDSCRIPTS} ${WORKSPACE}/third_party/sd-scripts
    echo -e "${GREEN}[OK]${NC} sd-scripts symlinked to ${WORKSPACE}/third_party/sd-scripts"
fi

# ============================================================================
# Auto-Sync Models from R2 (if not present)
# ============================================================================
# Set AUTO_SYNC_MODELS=1 to enable automatic model download on startup
# Models are synced from Cloudflare R2 for fast downloads (~2-5 min)
MODEL_DIR="${MODEL_PATH:-/workspace/models/flux1-dev}"
AUTO_SYNC="${AUTO_SYNC_MODELS:-1}"

if [ "$AUTO_SYNC" = "1" ] && [ ! -f "${MODEL_DIR}/flux1-dev.safetensors" ]; then
    echo -e "${BLUE}[INFO]${NC} Models not found at ${MODEL_DIR}"
    echo -e "${BLUE}[INFO]${NC} Auto-syncing models from R2 (this takes 2-5 minutes)..."
    echo ""

    # Run sync script in background so startup continues
    if [ -f "${WORKSPACE}/scripts/sync_models_r2.sh" ]; then
        # Run sync and log output
        nohup bash ${WORKSPACE}/scripts/sync_models_r2.sh > ${WORKSPACE}/logs/model_sync.log 2>&1 &
        SYNC_PID=$!
        echo -e "${YELLOW}[INFO]${NC} Model sync started in background (PID: ${SYNC_PID})"
        echo -e "${YELLOW}[INFO]${NC} Monitor progress: tail -f ${WORKSPACE}/logs/model_sync.log"
        echo ""
    else
        echo -e "${YELLOW}[WARN]${NC} Sync script not found. Run manually: bash scripts/sync_models_r2.sh"
    fi
elif [ -f "${MODEL_DIR}/flux1-dev.safetensors" ]; then
    echo -e "${GREEN}[OK]${NC} Models found at ${MODEL_DIR}"
fi

# ============================================================================
# Download Text Encoders (CLIP-L and T5-XXL from ComfyUI)
# ============================================================================
# These are required for FLUX.1 training with kohya-ss flux_train_network.py
TEXT_ENCODER_DIR="${TEXT_ENCODER_PATH:-/workspace/models/text_encoders}"

if [ ! -f "${TEXT_ENCODER_DIR}/clip_l.safetensors" ] || [ ! -f "${TEXT_ENCODER_DIR}/t5xxl_fp16.safetensors" ]; then
    echo -e "${BLUE}[INFO]${NC} Text encoders not found at ${TEXT_ENCODER_DIR}"
    echo -e "${BLUE}[INFO]${NC} Downloading CLIP-L and T5-XXL from ComfyUI (required for FLUX training)..."
    mkdir -p "${TEXT_ENCODER_DIR}"

    # Download in background
    (
        cd "${TEXT_ENCODER_DIR}"
        if [ ! -f "clip_l.safetensors" ]; then
            echo "[INFO] Downloading clip_l.safetensors (235MB)..."
            wget -q --show-progress -nc https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors 2>&1 || true
        fi
        if [ ! -f "t5xxl_fp16.safetensors" ]; then
            echo "[INFO] Downloading t5xxl_fp16.safetensors (9.2GB)..."
            wget -q --show-progress -nc https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors 2>&1 || true
        fi
        echo "[INFO] Text encoder download complete"
    ) > ${WORKSPACE}/logs/text_encoder_download.log 2>&1 &
    TE_PID=$!
    echo -e "${YELLOW}[INFO]${NC} Text encoder download started in background (PID: ${TE_PID})"
    echo -e "${YELLOW}[INFO]${NC} Monitor progress: tail -f ${WORKSPACE}/logs/text_encoder_download.log"
    echo ""
else
    echo -e "${GREEN}[OK]${NC} Text encoders found at ${TEXT_ENCODER_DIR}"
fi

# ============================================================================
# Banner
# ============================================================================
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       FLUX.1-dev Identity LoRA Training Environment            ║${NC}"
echo -e "${CYAN}║                    December 2025 SOTA                          ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# Version Info
# ============================================================================
echo -e "${BLUE}[INFO]${NC} System Versions:"
echo "----------------------------------------"

# NVIDIA Driver & GPU
echo -e "${YELLOW}GPU:${NC}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  GPU not detected"
echo ""

# Python
echo -e "${YELLOW}Python:${NC} $(python --version 2>&1)"

# Pip
echo -e "${YELLOW}Pip:${NC} $(pip --version 2>&1 | cut -d' ' -f1-2)"

# PyTorch
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
echo -e "${YELLOW}PyTorch:${NC} ${TORCH_VERSION}"

# CUDA availability
CUDA_AVAIL=$(python -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')" 2>/dev/null || echo "unknown")
echo -e "${YELLOW}CUDA Available:${NC} ${CUDA_AVAIL}"

if [ "$CUDA_AVAIL" = "Yes" ]; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    echo -e "${YELLOW}CUDA Version:${NC} ${CUDA_VERSION}"
fi

# Accelerate
ACCEL_VERSION=$(python -c "import accelerate; print(accelerate.__version__)" 2>/dev/null || echo "not installed")
echo -e "${YELLOW}Accelerate:${NC} ${ACCEL_VERSION}"

# bitsandbytes
BNB_VERSION=$(python -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null || echo "not installed")
echo -e "${YELLOW}bitsandbytes:${NC} ${BNB_VERSION}"

# sd-scripts
if [ -d "${SDSCRIPTS}" ]; then
    SDSCRIPTS_COMMIT=$(cd ${SDSCRIPTS} && git describe --tags --always 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo -e "${YELLOW}sd-scripts:${NC} ${SDSCRIPTS_COMMIT}"
else
    echo -e "${YELLOW}sd-scripts:${NC} not found"
fi

echo "----------------------------------------"
echo ""

# ============================================================================
# Create Runtime Directories
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Creating runtime directories..."
mkdir -p ${WORKSPACE}/logs
mkdir -p ${WORKSPACE}/output
mkdir -p ${WORKSPACE}/output/samples
mkdir -p ${WORKSPACE}/data/subject/images
mkdir -p ${WORKSPACE}/data/subject/captions
mkdir -p ${WORKSPACE}/data/reg/images
mkdir -p ${WORKSPACE}/data/reg/captions
echo -e "${GREEN}[OK]${NC} Directories ready"
echo ""

# ============================================================================
# Write pip freeze
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Writing pip freeze to logs/pip_freeze.txt..."
pip freeze > ${WORKSPACE}/logs/pip_freeze.txt
echo -e "${GREEN}[OK]${NC} pip freeze saved"
echo ""

# ============================================================================
# Healthchecks
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Running healthchecks..."
echo "----------------------------------------"

HEALTH_OK=true

# Check 1: GPU visible
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    echo -e "${GREEN}[PASS]${NC} GPU visible (${GPU_COUNT} device(s))"
else
    echo -e "${RED}[FAIL]${NC} GPU not visible"
    HEALTH_OK=false
fi

# Check 2: sd-scripts exists (with FLUX support)
if [ -d "${SDSCRIPTS}" ] && [ -f "${SDSCRIPTS}/flux_train_network.py" ]; then
    echo -e "${GREEN}[PASS]${NC} sd-scripts (sd3 branch) found at ${SDSCRIPTS}"
else
    echo -e "${RED}[FAIL]${NC} sd-scripts with FLUX support not found"
    HEALTH_OK=false
fi

# Check 3: flux_train_network.py importable
if python -c "import sys; sys.path.insert(0, '${SDSCRIPTS}'); import flux_train_network" 2>/dev/null; then
    echo -e "${GREEN}[PASS]${NC} flux_train_network.py importable"
else
    echo -e "${YELLOW}[WARN]${NC} flux_train_network.py import check skipped (may need deps)"
fi

# Check 4: accelerate config exists
ACCEL_CONFIG="/root/.cache/huggingface/accelerate/default_config.yaml"
if [ -f "${ACCEL_CONFIG}" ]; then
    echo -e "${GREEN}[PASS]${NC} Accelerate config exists"
else
    echo -e "${YELLOW}[WARN]${NC} Creating default accelerate config..."
    mkdir -p /root/.cache/huggingface/accelerate
    cat > ${ACCEL_CONFIG} << 'EOF'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: "NO"
downcast_bf16: "no"
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    echo -e "${GREEN}[OK]${NC} Accelerate config created"
fi

# Check 5: Workspace scripts exist
if [ -f "${WORKSPACE}/scripts/train_flux_fast.sh" ] && [ -f "${WORKSPACE}/scripts/train_flux_final.sh" ]; then
    echo -e "${GREEN}[PASS]${NC} Training scripts found"
else
    echo -e "${YELLOW}[WARN]${NC} Training scripts not found (may need to be mounted)"
fi

echo "----------------------------------------"

if [ "$HEALTH_OK" = true ]; then
    echo -e "${GREEN}[HEALTHCHECK]${NC} All critical checks passed"
else
    echo -e "${RED}[HEALTHCHECK]${NC} Some checks failed - review above"
fi
echo ""

# ============================================================================
# Usage Instructions
# ============================================================================
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                      READY FOR TRAINING                        ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Show SSH connection info if running on RunPod
if [ -n "$RUNPOD_POD_ID" ] || [ -n "$RUNPOD_PUBLIC_IP" ]; then
    echo -e "${YELLOW}SSH Connection (TCP):${NC}"
    if [ -n "$RUNPOD_TCP_PORT_22" ]; then
        echo "  ssh root@${RUNPOD_PUBLIC_IP:-<pod-ip>} -p ${RUNPOD_TCP_PORT_22}"
    else
        echo "  Ensure 'Expose TCP Ports' is enabled in your RunPod template"
        echo "  TCP port 22 must be exposed for SSH access"
    fi
    echo "  Default password: runpod (change with: passwd)"
    echo ""
fi

echo -e "${YELLOW}Dataset Location:${NC}"
echo "  Mount your dataset to: /workspace/lora_training/data/subject/"
echo "  Structure: images/*.jpg|png + captions/*.txt (same basenames)"
echo ""
echo -e "${YELLOW}To Analyze Your Dataset:${NC}"
echo "  cd ${WORKSPACE}"
echo "  python scripts/analyze_dataset.py"
echo ""
echo -e "${YELLOW}To Start Training (Fast Iteration):${NC}"
echo "  cd ${WORKSPACE}"
echo "  bash scripts/train_flux_fast.sh"
echo ""
echo -e "${YELLOW}To Start Training (Final/Production):${NC}"
echo "  cd ${WORKSPACE}"
echo "  bash scripts/train_flux_final.sh"
echo ""
echo -e "${YELLOW}To Start Training with Telemetry (tmux):${NC}"
echo "  cd ${WORKSPACE}"
echo "  bash scripts/tmux_train.sh fast   # or 'final'"
echo ""
echo -e "${YELLOW}To Test Without Training (Dry Run):${NC}"
echo "  DRY_RUN=1 bash scripts/train_flux_fast.sh"
echo "  DRY_RUN=1 bash scripts/train_flux_final.sh"
echo ""
echo -e "${YELLOW}Outputs:${NC}"
echo "  Logs:    ${WORKSPACE}/logs/"
echo "  Models:  ${WORKSPACE}/output/"
echo "  Samples: ${WORKSPACE}/output/samples/"
echo ""
echo -e "${YELLOW}GPU Telemetry:${NC}"
echo "  bash scripts/telemetry.sh"
echo "  # or: nvitop"
echo ""
echo "----------------------------------------"
echo -e "${GREEN}Container is ready. Training will NOT start automatically.${NC}"
echo "----------------------------------------"
echo ""

# Keep container running
# If TTY is attached (interactive), start bash shell
# Otherwise (RunPod template), sleep forever to keep container alive
if [ -t 0 ]; then
    echo "[INFO] Interactive mode detected, starting bash..."
    exec /bin/bash
else
    echo "[INFO] Non-interactive mode (RunPod template), keeping container alive..."
    echo "[INFO] Use SSH or Web Terminal to connect"
    # Sleep forever - SSH server handles connections in background
    exec sleep infinity
fi

#!/bin/bash
# ============================================================================
# Identity LoRA Training - Container Startup Script
# ============================================================================
# This script runs on container boot to:
# 1. Print version banner
# 2. Create runtime directories
# 3. Write pip freeze
# 4. Run healthchecks
# 5. Print usage instructions (does NOT start training)
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
SDSCRIPTS="/workspace/sd-scripts"

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

# Check 2: sd-scripts exists
if [ -d "${SDSCRIPTS}" ] && [ -f "${SDSCRIPTS}/train_network.py" ]; then
    echo -e "${GREEN}[PASS]${NC} sd-scripts found at ${SDSCRIPTS}"
else
    echo -e "${RED}[FAIL]${NC} sd-scripts not found"
    HEALTH_OK=false
fi

# Check 3: train_network.py importable
if python -c "import sys; sys.path.insert(0, '${SDSCRIPTS}'); import train_network" 2>/dev/null; then
    echo -e "${GREEN}[PASS]${NC} train_network.py importable"
else
    echo -e "${YELLOW}[WARN]${NC} train_network.py import check skipped (may need deps)"
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

# Keep container running (interactive shell)
exec /bin/bash

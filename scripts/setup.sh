#!/bin/bash
# ============================================================================
# Setup Script for Identity LoRA Training
# ============================================================================
# Run this script to set up the training environment locally or in a container.
# This is typically run automatically by Docker, but can be run manually.
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       Identity LoRA Training - Setup Script                    ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# Check Python
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Checking Python..."
if ! command -v python &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python not found"
    exit 1
fi
PYTHON_VERSION=$(python --version 2>&1)
echo -e "${GREEN}[OK]${NC} ${PYTHON_VERSION}"

# ============================================================================
# Check CUDA
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Checking CUDA..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    echo -e "${GREEN}[OK]${NC} CUDA ${CUDA_VERSION} available (${GPU_COUNT} GPU(s))"
else
    echo -e "${YELLOW}[WARN]${NC} CUDA not available (CPU-only mode)"
fi

# ============================================================================
# Create directories
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Creating directories..."
mkdir -p "${WORKSPACE}/output"
mkdir -p "${WORKSPACE}/output/samples"
mkdir -p "${WORKSPACE}/logs"
mkdir -p "${WORKSPACE}/data/subject/images"
mkdir -p "${WORKSPACE}/data/subject/captions"
mkdir -p "${WORKSPACE}/data/reg/images"
mkdir -p "${WORKSPACE}/data/reg/captions"
mkdir -p "${WORKSPACE}/third_party"
echo -e "${GREEN}[OK]${NC} Directories created"

# ============================================================================
# Clone sd-scripts if not present
# ============================================================================
SDSCRIPTS_DIR="${WORKSPACE}/third_party/sd-scripts"
SDSCRIPTS_VERSION="v0.9.2"

if [ ! -d "${SDSCRIPTS_DIR}" ]; then
    echo -e "${BLUE}[INFO]${NC} Cloning sd-scripts (${SDSCRIPTS_VERSION})..."

    # Check if /workspace/sd-scripts exists (Docker case)
    if [ -d "/workspace/sd-scripts" ]; then
        echo -e "${BLUE}[INFO]${NC} Linking from /workspace/sd-scripts..."
        ln -sf /workspace/sd-scripts "${SDSCRIPTS_DIR}"
    else
        git clone https://github.com/kohya-ss/sd-scripts.git "${SDSCRIPTS_DIR}"
        cd "${SDSCRIPTS_DIR}"
        git checkout "${SDSCRIPTS_VERSION}"
        cd "${WORKSPACE}"
    fi
    echo -e "${GREEN}[OK]${NC} sd-scripts installed"
else
    echo -e "${GREEN}[OK]${NC} sd-scripts already present"
fi

# Verify sd-scripts
if [ -f "${SDSCRIPTS_DIR}/train_network.py" ]; then
    SDSCRIPTS_COMMIT=$(cd "${SDSCRIPTS_DIR}" && git describe --tags --always 2>/dev/null || echo "unknown")
    echo -e "${GREEN}[OK]${NC} sd-scripts version: ${SDSCRIPTS_COMMIT}"
else
    echo -e "${RED}[ERROR]${NC} sd-scripts train_network.py not found"
    exit 1
fi

# ============================================================================
# Install sd-scripts dependencies
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Checking sd-scripts dependencies..."
if [ -f "${SDSCRIPTS_DIR}/requirements.txt" ]; then
    pip install -q -r "${SDSCRIPTS_DIR}/requirements.txt" 2>/dev/null || true
    echo -e "${GREEN}[OK]${NC} Dependencies checked"
fi

# ============================================================================
# Create accelerate config
# ============================================================================
ACCEL_CONFIG_DIR="${HOME}/.cache/huggingface/accelerate"
ACCEL_CONFIG="${ACCEL_CONFIG_DIR}/default_config.yaml"

if [ ! -f "${ACCEL_CONFIG}" ]; then
    echo -e "${BLUE}[INFO]${NC} Creating accelerate config..."
    mkdir -p "${ACCEL_CONFIG_DIR}"
    cat > "${ACCEL_CONFIG}" << 'EOF'
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
else
    echo -e "${GREEN}[OK]${NC} Accelerate config exists"
fi

# ============================================================================
# Make scripts executable
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Making scripts executable..."
chmod +x "${WORKSPACE}/scripts/"*.sh 2>/dev/null || true
chmod +x "${WORKSPACE}/docker/"*.sh 2>/dev/null || true
echo -e "${GREEN}[OK]${NC} Scripts are executable"

# ============================================================================
# Write pip freeze
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Writing pip freeze..."
pip freeze > "${WORKSPACE}/logs/pip_freeze.txt"
echo -e "${GREEN}[OK]${NC} pip freeze saved to logs/pip_freeze.txt"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                      SETUP COMPLETE                            ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Place your dataset in:"
echo "   ${WORKSPACE}/data/subject/"
echo ""
echo "2. Analyze your dataset:"
echo "   python scripts/analyze_dataset.py --data-dir data/subject"
echo ""
echo "3. Start training:"
echo "   bash scripts/train_flux_fast.sh    # Fast iteration"
echo "   bash scripts/train_flux_final.sh   # Production"
echo ""
echo "4. Or use tmux for telemetry:"
echo "   bash scripts/tmux_train.sh fast"
echo "   bash scripts/tmux_train.sh final"
echo ""

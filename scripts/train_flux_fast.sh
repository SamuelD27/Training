#!/bin/bash
# ============================================================================
# FLUX.1-dev Identity LoRA - Fast Iteration Training
# ============================================================================
# Uses kohya-ss flux_train_network.py for FLUX.1 training
# Reference: https://github.com/kohya-ss/sd-scripts/blob/sd3/docs/flux_train_network.md
# Purpose: Validate dataset, detect early overfit, <1 hour runtime
#
# This is a thin wrapper around scripts/build_train_cmd.py (Single Source of Truth)
# Configuration is loaded from configs/flux_fast.toml
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Load environment for paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
export WORKSPACE

if [ -f "${WORKSPACE}/docker/env.sh" ]; then
    source "${WORKSPACE}/docker/env.sh"
else
    echo -e "${YELLOW}[WARN]${NC} env.sh not found at ${WORKSPACE}/docker/env.sh, using defaults"
fi

# Profile: FAST
PROFILE="fast"

# Generate run name with timestamp if not set
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export RUN_NAME="${RUN_NAME:-flux_fast_${TIMESTAMP}}"

# ============================================================================
# Banner
# ============================================================================
echo ""
echo -e "${CYAN}============================================================================${NC}"
echo -e "${CYAN}       FLUX.1-dev Identity LoRA - FAST ITERATION                ${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo ""

# ============================================================================
# Validate environment
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Validating environment..."

# Check model path
if [ ! -f "${MODEL_PATH}/flux1-dev.safetensors" ]; then
    echo -e "${RED}[ERROR]${NC} FLUX.1-dev model not found at: ${MODEL_PATH}/flux1-dev.safetensors"
    echo "  Run: bash scripts/sync_models_r2.sh"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} FLUX.1-dev model found"

# Check text encoders
if [ ! -f "${TEXT_ENCODER_PATH}/clip_l.safetensors" ] || [ ! -f "${TEXT_ENCODER_PATH}/t5xxl_fp16.safetensors" ]; then
    echo -e "${RED}[ERROR]${NC} Text encoders not found at: ${TEXT_ENCODER_PATH}"
    echo "  They should auto-download on container start, or download manually:"
    echo "  wget https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
    echo "  wget https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Text encoders found"

# Check dataset
if [ ! -d "${DATA_DIR}" ]; then
    echo -e "${RED}[ERROR]${NC} Dataset directory not found: ${DATA_DIR}"
    exit 1
fi

IMG_COUNT=$(find "${DATA_DIR}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) 2>/dev/null | wc -l | tr -d ' ')
if [ "$IMG_COUNT" -eq 0 ]; then
    echo -e "${RED}[ERROR]${NC} No images found in dataset: ${DATA_DIR}"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Dataset found: ${IMG_COUNT} images"

# Check sd-scripts (FLUX support)
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo -e "${RED}[ERROR]${NC} flux_train_network.py not found at: ${TRAIN_SCRIPT}"
    echo "  Make sure sd-scripts is on the sd3 branch"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} sd-scripts (sd3 branch) found"

# ============================================================================
# Build command using Single Source of Truth
# ============================================================================
echo ""
echo -e "${BLUE}[INFO]${NC} Building training command from configs/flux_fast.toml..."

# Check for Python and tomli
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python3 not found"
    exit 1
fi

# Build command using build_train_cmd.py (Single Source of Truth)
CMD=$(python3 "${SCRIPT_DIR}/build_train_cmd.py" --profile "${PROFILE}" --run-name "${RUN_NAME}")

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Failed to build training command"
    exit 1
fi

# Create output directories
mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/samples"
mkdir -p "${LOG_DIR}"

# ============================================================================
# Display configuration (from TOML via build_train_cmd.py)
# ============================================================================
echo ""
echo -e "${YELLOW}Configuration (from configs/flux_fast.toml):${NC}"
python3 "${SCRIPT_DIR}/build_train_cmd.py" --profile "${PROFILE}" --run-name "${RUN_NAME}" --dry-run 2>&1 | grep -E "^\s{2}" || true
echo ""

# ============================================================================
# Reproducibility artifacts (P1 fix)
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Saving reproducibility artifacts..."

# Save the executed command
echo "${CMD}" > "${LOG_DIR}/${RUN_NAME}_command.txt"

# Save reproducibility info
python3 "${SCRIPT_DIR}/build_train_cmd.py" --profile "${PROFILE}" --run-name "${RUN_NAME}" --show-repro > "${LOG_DIR}/${RUN_NAME}_repro.json" 2>/dev/null || true

# Save pip freeze (non-blocking, best effort)
pip freeze > "${LOG_DIR}/${RUN_NAME}_pip_freeze.txt" 2>/dev/null || true

# Save dataset hash
python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from build_train_cmd import compute_dataset_hash
print(compute_dataset_hash('${DATA_DIR}'))
" > "${LOG_DIR}/${RUN_NAME}_dataset_hash.txt" 2>/dev/null || echo "hash_unavailable" > "${LOG_DIR}/${RUN_NAME}_dataset_hash.txt"

echo -e "${GREEN}[OK]${NC} Artifacts saved to ${LOG_DIR}/"

# ============================================================================
# Execute or dry run
# ============================================================================
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

echo "----------------------------------------"
echo -e "${CYAN}Command:${NC}"
echo "${CMD}"
echo "----------------------------------------"
echo ""

if [ "${DRY_RUN}" = "1" ]; then
    echo -e "${YELLOW}[DRY RUN]${NC} Command printed above. Set DRY_RUN=0 to execute."
    exit 0
fi

echo -e "${GREEN}[START]${NC} Training started at $(date)"
echo -e "${BLUE}[INFO]${NC} Logging to: ${LOG_FILE}"
echo -e "${BLUE}[INFO]${NC} Samples will be saved to: ${OUT_DIR}/samples/${RUN_NAME}/"

# Log resume status if applicable
if [ -n "${RESUME_FROM}" ]; then
    echo -e "${YELLOW}[RESUME]${NC} Resuming from: ${RESUME_FROM}"
fi

echo ""

# Execute with tee for logging
eval "${CMD}" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[SUCCESS]${NC} Training completed at $(date)"
    echo -e "${GREEN}[OUTPUT]${NC} LoRA saved to: ${OUT_DIR}/${RUN_NAME}.safetensors"
else
    echo ""
    echo -e "${RED}[FAILED]${NC} Training failed with exit code: ${EXIT_CODE}"
    echo ""
    echo -e "${YELLOW}Troubleshooting hints:${NC}"
    echo "  - NaNs: Lower learning rate (LEARNING_RATE=5e-5)"
    echo "  - OOM: Reduce batch size or enable more aggressive checkpointing"
    echo "  - Check log: ${LOG_FILE}"
    exit $EXIT_CODE
fi

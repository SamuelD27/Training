#!/bin/bash
# ============================================================================
# FLUX.1-dev Identity LoRA - Final High-Fidelity Training (Production)
# ============================================================================
# Source: DeepResearchReport.md Section 4.2
# Purpose: Maximum realism, strong identity lock, robust generalization
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [ -f "${WORKSPACE}/docker/env.sh" ]; then
    source "${WORKSPACE}/docker/env.sh"
else
    echo -e "${RED}[ERROR]${NC} env.sh not found at ${WORKSPACE}/docker/env.sh"
    exit 1
fi

# Profile: FINAL
PROFILE="final"

# Set defaults from env.sh (FINAL profile)
RESOLUTION="${RESOLUTION:-${FINAL_RESOLUTION}}"
MAX_STEPS="${MAX_STEPS:-${FINAL_MAX_STEPS}}"
RANK="${RANK:-${FINAL_RANK}}"
ALPHA="${ALPHA:-${FINAL_ALPHA}}"
UNET_LR="${UNET_LR:-${FINAL_UNET_LR}}"
DROPOUT="${DROPOUT:-${FINAL_DROPOUT}}"
NOISE_OFFSET="${NOISE_OFFSET:-${FINAL_NOISE_OFFSET}}"
MIN_BUCKET="${MIN_BUCKET_RESO:-${FINAL_MIN_BUCKET}}"
MAX_BUCKET="${MAX_BUCKET_RESO:-${FINAL_MAX_BUCKET}}"
WARMUP="${WARMUP:-${FINAL_WARMUP}}"
SNR_GAMMA="${SNR_GAMMA:-${FINAL_SNR_GAMMA}}"
LR_SCHEDULER="cosine"

# Generate run name with timestamp if not set
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="${RUN_NAME:-flux_final_${TIMESTAMP}}"

# ============================================================================
# Banner
# ============================================================================
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       FLUX.1-dev Identity LoRA - FINAL / PRODUCTION            ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# Validate environment
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Validating environment..."

# Check model path
if [ ! -d "${MODEL_PATH}" ]; then
    echo -e "${RED}[ERROR]${NC} Model not found at: ${MODEL_PATH}"
    echo "  Set MODEL_PATH environment variable or download the model."
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Model found: ${MODEL_PATH}"

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

# Warn if small dataset for final profile
if [ "$IMG_COUNT" -lt 15 ]; then
    echo -e "${YELLOW}[WARN]${NC} Dataset has only ${IMG_COUNT} images. Recommended: 15-30 for final profile."
fi

# Check sd-scripts
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo -e "${RED}[ERROR]${NC} train_network.py not found at: ${TRAIN_SCRIPT}"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} sd-scripts found"

# ============================================================================
# Run dataset analysis
# ============================================================================
echo ""
echo -e "${BLUE}[INFO]${NC} Running dataset analysis..."
python "${WORKSPACE}/scripts/analyze_dataset.py" --data-dir "${DATA_DIR}" --output "${LOG_DIR}/dataset_report.json" 2>/dev/null || true

# Check for recommendations and potentially adjust resolution
if [ -f "${LOG_DIR}/dataset_report.json" ]; then
    SUGGESTED_RESO=$(python3 -c "import json; r=json.load(open('${LOG_DIR}/dataset_report.json')); print(r.get('recommendations',{}).get('base_resolution', 768))" 2>/dev/null || echo "768")
    SUGGESTED_MIN=$(python3 -c "import json; r=json.load(open('${LOG_DIR}/dataset_report.json')); print(r.get('recommendations',{}).get('min_bucket_reso', 384))" 2>/dev/null || echo "384")
    SUGGESTED_MAX=$(python3 -c "import json; r=json.load(open('${LOG_DIR}/dataset_report.json')); print(r.get('recommendations',{}).get('max_bucket_reso', 1024))" 2>/dev/null || echo "1024")

    echo -e "${BLUE}[INFO]${NC} Dataset analysis recommendations:"
    echo "  Base Resolution: ${SUGGESTED_RESO}"
    echo "  Bucket Range: ${SUGGESTED_MIN}-${SUGGESTED_MAX}"

    # Use suggested values if not explicitly overridden
    if [ -z "${RESOLUTION_OVERRIDE}" ]; then
        RESOLUTION="${SUGGESTED_RESO}"
        MIN_BUCKET="${SUGGESTED_MIN}"
        MAX_BUCKET="${SUGGESTED_MAX}"
    fi
fi

# ============================================================================
# Calculate TE learning rate (two-phase support)
# ============================================================================
if [ "${ENABLE_TE}" = "1" ]; then
    TE_LR=$(calc_te_lr "${UNET_LR}")
    echo -e "${YELLOW}[INFO]${NC} Text Encoder training ENABLED (LR: ${TE_LR})"

    if [ "${TWO_PHASE_TE}" = "1" ]; then
        # Calculate phase boundary
        PHASE1_STEPS=$(python3 -c "print(int(${MAX_STEPS} * ${TE_PHASE_RATIO}))")
        PHASE2_STEPS=$((MAX_STEPS - PHASE1_STEPS))
        echo -e "${YELLOW}[INFO]${NC} Two-phase TE training:"
        echo "  Phase 1 (TE on):  ${PHASE1_STEPS} steps"
        echo "  Phase 2 (TE off): ${PHASE2_STEPS} steps"
        echo ""
        echo -e "${YELLOW}[NOTE]${NC} Two-phase training requires manual checkpoint resume."
        echo "  1. Run with MAX_STEPS=${PHASE1_STEPS}"
        echo "  2. Resume from checkpoint with ENABLE_TE=0"
    fi
else
    TE_LR="0"
fi

# ============================================================================
# Build command
# ============================================================================
echo ""
echo -e "${BLUE}[INFO]${NC} Building training command..."
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Run Name:       ${RUN_NAME}"
echo "  Profile:        ${PROFILE}"
echo "  Resolution:     ${RESOLUTION}x${RESOLUTION}"
echo "  Max Steps:      ${MAX_STEPS}"
echo "  Rank/Alpha:     ${RANK}/${ALPHA}"
echo "  Dropout:        ${DROPOUT}"
echo "  UNet LR:        ${UNET_LR}"
echo "  TE LR:          ${TE_LR}"
echo "  Batch Size:     ${BATCH_SIZE}"
echo "  Grad Accum:     ${GRAD_ACCUM}"
echo "  Bucket Range:   ${MIN_BUCKET}-${MAX_BUCKET}"
echo "  Scheduler:      ${LR_SCHEDULER}"
echo "  Warmup:         ${WARMUP}"
echo "  Noise Offset:   ${NOISE_OFFSET}"
echo "  SNR Gamma:      ${SNR_GAMMA}"
echo ""

# Create output directories
mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/samples/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

# Build accelerate command
CMD="accelerate launch ${TRAIN_SCRIPT}"

# Model
CMD="${CMD} --pretrained_model_name_or_path=${MODEL_PATH}"

# Network
CMD="${CMD} --network_module=networks.lora"
CMD="${CMD} --network_dim=${RANK}"
CMD="${CMD} --network_alpha=${ALPHA}"
CMD="${CMD} --network_dropout=${DROPOUT}"

# Learning rates
CMD="${CMD} --unet_lr=${UNET_LR}"
CMD="${CMD} --text_encoder_lr=${TE_LR}"

# Optimizer
CMD="${CMD} --optimizer_type=Adafactor"
CMD="${CMD} --optimizer_args=\"relative_step=False\" \"scale_parameter=False\""

# Scheduler
CMD="${CMD} --lr_scheduler=${LR_SCHEDULER}"
CMD="${CMD} --lr_warmup_steps=${WARMUP}"

# Resolution and bucketing
CMD="${CMD} --resolution=${RESOLUTION},${RESOLUTION}"
CMD="${CMD} --enable_bucket"
CMD="${CMD} --min_bucket_reso=${MIN_BUCKET}"
CMD="${CMD} --max_bucket_reso=${MAX_BUCKET}"

# Precision
CMD="${CMD} --mixed_precision=bf16"
CMD="${CMD} --full_bf16"
if [ "${FP8_BASE}" = "1" ]; then
    CMD="${CMD} --fp8_base"
fi

# Memory
CMD="${CMD} --gradient_checkpointing"

# Training
CMD="${CMD} --train_data_dir=${DATA_DIR}"
CMD="${CMD} --train_batch_size=${BATCH_SIZE}"
CMD="${CMD} --gradient_accumulation_steps=${GRAD_ACCUM}"
CMD="${CMD} --max_train_steps=${MAX_STEPS}"

# Noise and loss
CMD="${CMD} --noise_offset=${NOISE_OFFSET}"
CMD="${CMD} --min_snr_gamma=${SNR_GAMMA}"

# Captions
CMD="${CMD} --caption_extension=.txt"
CMD="${CMD} --keep_tokens=1"

# Saving
CMD="${CMD} --output_dir=${OUT_DIR}"
CMD="${CMD} --output_name=${RUN_NAME}"
CMD="${CMD} --save_every_n_steps=${SAVE_EVERY_N_STEPS}"
CMD="${CMD} --save_model_as=safetensors"
CMD="${CMD} --save_precision=bf16"

# Sampling
CMD="${CMD} --sample_every_n_steps=${SAMPLE_EVERY_N_STEPS}"
CMD="${CMD} --sample_prompts=${SAMPLE_PROMPTS}"
CMD="${CMD} --sample_sampler=euler"

# Logging
CMD="${CMD} --logging_dir=${LOG_DIR}"

# Seed
CMD="${CMD} --seed=${SEED}"

# Regularization (if enabled)
if [ "${USE_REG}" = "1" ] && [ -d "${REG_DIR}" ]; then
    REG_COUNT=$(find "${REG_DIR}" -type f \( -iname "*.jpg" -o -iname "*.png" \) 2>/dev/null | wc -l | tr -d ' ')
    if [ "$REG_COUNT" -gt 0 ]; then
        CMD="${CMD} --reg_data_dir=${REG_DIR}"
        CMD="${CMD} --prior_loss_weight=1.0"
        echo -e "${GREEN}[OK]${NC} Regularization enabled: ${REG_COUNT} images"
    fi
fi

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
echo ""

# Execute with tee for logging
eval "${CMD}" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[SUCCESS]${NC} Training completed at $(date)"
    echo -e "${GREEN}[OUTPUT]${NC} LoRA saved to: ${OUT_DIR}/${RUN_NAME}.safetensors"
    echo ""
    echo -e "${BLUE}[NEXT STEPS]${NC}"
    echo "  1. Review samples in: ${OUT_DIR}/samples/${RUN_NAME}/"
    echo "  2. Test LoRA at different strengths: 0.5, 0.8, 1.0, 1.2"
    echo "  3. Check evaluation protocol in docs/EVAL_PROTOCOL.md"
else
    echo ""
    echo -e "${RED}[FAILED]${NC} Training failed with exit code: ${EXIT_CODE}"
    echo ""
    echo -e "${YELLOW}Troubleshooting hints (from report Section 9):${NC}"
    echo "  - NaNs: Lower learning rate (try 5e-5) or ensure bf16"
    echo "  - Weak identity: More steps or higher rank"
    echo "  - Sameface: Early stop, enable dropout, use regularization"
    echo "  - Waxy skin: Lower noise_offset (try 0.05)"
    echo "  - OOM: Reduce batch size or resolution"
    echo "  - Check log: ${LOG_FILE}"
    exit $EXIT_CODE
fi

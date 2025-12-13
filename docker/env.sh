#!/bin/bash
# ============================================================================
# Environment Variables for Identity LoRA Training
# ============================================================================
# Source this file before running training scripts:
#   source /workspace/lora_training/docker/env.sh
# ============================================================================

# Paths (can be overridden by environment)
export WORKSPACE="${WORKSPACE:-/workspace/lora_training}"
export SDSCRIPTS="${SDSCRIPTS:-/workspace/sd-scripts}"
export TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SDSCRIPTS}/train_network.py}"

# Default model path (override with MODEL_PATH env var)
export MODEL_PATH="${MODEL_PATH:-/workspace/models/flux1-dev}"

# Dataset paths
export DATA_DIR="${DATA_DIR:-${WORKSPACE}/data/subject}"
export REG_DIR="${REG_DIR:-${WORKSPACE}/data/reg}"
export OUT_DIR="${OUT_DIR:-${WORKSPACE}/output}"
export LOG_DIR="${LOG_DIR:-${WORKSPACE}/logs}"

# Training defaults - FAST profile (from report Section 4.1)
export FAST_RESOLUTION="${FAST_RESOLUTION:-512}"
export FAST_MAX_STEPS="${FAST_MAX_STEPS:-1500}"
export FAST_RANK="${FAST_RANK:-32}"
export FAST_ALPHA="${FAST_ALPHA:-32}"
export FAST_UNET_LR="${FAST_UNET_LR:-1e-4}"
export FAST_TE_LR="${FAST_TE_LR:-0}"
export FAST_NOISE_OFFSET="${FAST_NOISE_OFFSET:-0.05}"
export FAST_MIN_BUCKET="${FAST_MIN_BUCKET:-256}"
export FAST_MAX_BUCKET="${FAST_MAX_BUCKET:-768}"
export FAST_WARMUP="${FAST_WARMUP:-100}"

# Training defaults - FINAL profile (from report Section 4.2)
export FINAL_RESOLUTION="${FINAL_RESOLUTION:-768}"
export FINAL_MAX_STEPS="${FINAL_MAX_STEPS:-2500}"
export FINAL_RANK="${FINAL_RANK:-64}"
export FINAL_ALPHA="${FINAL_ALPHA:-64}"
export FINAL_UNET_LR="${FINAL_UNET_LR:-1e-4}"
export FINAL_TE_LR="${FINAL_TE_LR:-0}"
export FINAL_DROPOUT="${FINAL_DROPOUT:-0.1}"
export FINAL_NOISE_OFFSET="${FINAL_NOISE_OFFSET:-0.1}"
export FINAL_MIN_BUCKET="${FINAL_MIN_BUCKET:-384}"
export FINAL_MAX_BUCKET="${FINAL_MAX_BUCKET:-1024}"
export FINAL_WARMUP="${FINAL_WARMUP:-500}"
export FINAL_SNR_GAMMA="${FINAL_SNR_GAMMA:-5.0}"

# Common defaults
export SEED="${SEED:-42}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-500}"
export SAMPLE_EVERY_N_STEPS="${SAMPLE_EVERY_N_STEPS:-250}"

# Text encoder policy (from report Section 5)
# Default: TE frozen. Set ENABLE_TE=1 to enable.
export ENABLE_TE="${ENABLE_TE:-0}"
export TE_LR_RATIO="${TE_LR_RATIO:-0.4}"  # TE_LR = UNET_LR * TE_LR_RATIO

# Two-phase TE training (if ENABLE_TE=1 and TWO_PHASE_TE=1)
export TWO_PHASE_TE="${TWO_PHASE_TE:-0}"
export TE_PHASE_RATIO="${TE_PHASE_RATIO:-0.5}"  # TE enabled for first 50% of steps

# Override with single env vars if provided
export RUN_NAME="${RUN_NAME:-flux_lora}"
export RANK="${RANK:-}"
export ALPHA="${ALPHA:-}"
export UNET_LR="${UNET_LR:-}"
export MAX_STEPS="${MAX_STEPS:-}"
export RESOLUTION="${RESOLUTION:-}"

# Regularization
export USE_REG="${USE_REG:-0}"

# Sample generation
export SAMPLE_PROMPTS="${SAMPLE_PROMPTS:-${WORKSPACE}/configs/sample_prompts.txt}"

# DRY_RUN mode - print command but don't execute
export DRY_RUN="${DRY_RUN:-0}"

# Precision settings (from report Section 3)
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export FP8_BASE="${FP8_BASE:-1}"

# ============================================================================
# Helper Functions
# ============================================================================

# Calculate TE learning rate based on policy
calc_te_lr() {
    local unet_lr="$1"
    if [ "$ENABLE_TE" = "1" ]; then
        python3 -c "print(f'{float($unet_lr) * float($TE_LR_RATIO):.2e}')"
    else
        echo "0"
    fi
}

# Get timestamp for run naming
get_timestamp() {
    date +"%Y%m%d_%H%M%S"
}

# Check if dataset exists
check_dataset() {
    local data_dir="$1"
    if [ ! -d "$data_dir" ]; then
        echo "ERROR: Dataset directory not found: $data_dir"
        return 1
    fi

    local img_count=$(find "$data_dir" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) 2>/dev/null | wc -l)
    if [ "$img_count" -eq 0 ]; then
        echo "ERROR: No images found in $data_dir"
        return 1
    fi

    echo "Found $img_count images in dataset"
    return 0
}

echo "[env.sh] Environment loaded. Workspace: ${WORKSPACE}"

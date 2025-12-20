#!/bin/bash
# ============================================================================
# Remote LoRA Training - All-in-One Script
# ============================================================================
# Run this locally on your Mac to:
#   1. Select and upload your training dataset
#   2. Connect to your RunPod instance
#   3. Launch training with live dashboard
#
# Usage: bash scripts/train_remote.sh
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Default configuration
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_DATA_DIR="/workspace/lora_training/data/subject"
IMAGE_EXTENSIONS="jpg jpeg png webp"

# Load saved connection from last session
CONFIG_FILE="$HOME/.lora_training_config"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                      ║${NC}"
echo -e "${CYAN}║   ${BOLD}FLUX.1-dev Identity LoRA Training${NC}${CYAN}                                 ║${NC}"
echo -e "${CYAN}║   All-in-One Training Pipeline                                       ║${NC}"
echo -e "${CYAN}║                                                                      ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# Step 1: Remote Connection Details
# ============================================================================
echo -e "${BOLD}${BLUE}[1/4]${NC} ${BOLD}Remote Connection${NC}"
echo "────────────────────────────────────────────────────────────────────────"

# Load previous config if exists
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    echo -e "  Last connection: ${GREEN}${REMOTE_HOST}:${REMOTE_PORT}${NC}"
    read -p "  Use this connection? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        REMOTE_HOST=""
        REMOTE_PORT=""
    fi
fi

if [ -z "$REMOTE_HOST" ]; then
    read -p "  Remote host (e.g., 38.147.83.28): " REMOTE_HOST
fi

if [ -z "$REMOTE_PORT" ]; then
    read -p "  SSH port (e.g., 46243): " REMOTE_PORT
fi

# Save config for next time
echo "REMOTE_HOST=\"$REMOTE_HOST\"" > "$CONFIG_FILE"
echo "REMOTE_PORT=\"$REMOTE_PORT\"" >> "$CONFIG_FILE"

# Test connection
echo ""
echo -e "  Testing connection..."
if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$SSH_KEY" -p "$REMOTE_PORT" "${REMOTE_USER}@${REMOTE_HOST}" "echo 'ok'" &>/dev/null; then
    GPU_INFO=$(ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p "$REMOTE_PORT" "${REMOTE_USER}@${REMOTE_HOST}" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null || echo "Unknown GPU")
    echo -e "  ${GREEN}✓${NC} Connected to ${REMOTE_HOST}:${REMOTE_PORT}"
    echo -e "  ${GREEN}✓${NC} GPU: ${GPU_INFO}"
else
    echo -e "  ${RED}✗${NC} Connection failed"
    echo "  Check: ssh -i $SSH_KEY -p $REMOTE_PORT ${REMOTE_USER}@${REMOTE_HOST}"
    exit 1
fi
echo ""

# ============================================================================
# Step 2: Select Dataset
# ============================================================================
echo -e "${BOLD}${BLUE}[2/4]${NC} ${BOLD}Select Training Dataset${NC}"
echo "────────────────────────────────────────────────────────────────────────"
echo ""

# Build list of folders with images
declare -a FOLDER_OPTIONS
idx=1

echo "  Available folders with images:"
echo ""

for base_dir in "$HOME/Desktop" "$HOME/Downloads" "$HOME/Documents"; do
    if [ -d "$base_dir" ]; then
        while IFS= read -r dir; do
            if [ -d "$dir" ]; then
                img_count=0
                txt_count=0
                for ext in $IMAGE_EXTENSIONS; do
                    count=$(find "$dir" -maxdepth 1 -iname "*.$ext" 2>/dev/null | wc -l | tr -d ' ')
                    img_count=$((img_count + count))
                done
                txt_count=$(find "$dir" -maxdepth 1 -name "*.txt" 2>/dev/null | wc -l | tr -d ' ')

                if [ "$img_count" -gt 0 ]; then
                    FOLDER_OPTIONS[$idx]="$dir"
                    rel_path="${dir/#$HOME/~}"
                    if [ "$txt_count" -eq "$img_count" ]; then
                        match_status="${GREEN}✓ matched${NC}"
                    elif [ "$txt_count" -gt 0 ]; then
                        match_status="${YELLOW}! partial${NC}"
                    else
                        match_status="${RED}✗ no captions${NC}"
                    fi
                    printf "  ${BOLD}[%d]${NC} %-45s %d imgs, %d txts %b\n" "$idx" "$rel_path" "$img_count" "$txt_count" "$match_status"
                    idx=$((idx + 1))
                fi
            fi
        done < <(find "$base_dir" -maxdepth 2 -type d 2>/dev/null | head -30)
    fi
done

echo ""
echo "  ${BOLD}[p]${NC} Enter custom path"
echo "  ${BOLD}[q]${NC} Quit"
echo ""

read -p "  Select folder: " selection

case "$selection" in
    q|Q) echo "Cancelled."; exit 0 ;;
    p|P)
        read -p "  Enter full path: " LOCAL_FOLDER
        LOCAL_FOLDER="${LOCAL_FOLDER/#\~/$HOME}"
        ;;
    [0-9]*)
        if [ -n "${FOLDER_OPTIONS[$selection]}" ]; then
            LOCAL_FOLDER="${FOLDER_OPTIONS[$selection]}"
        else
            echo -e "  ${RED}Invalid selection${NC}"; exit 1
        fi
        ;;
    *) echo -e "  ${RED}Invalid selection${NC}"; exit 1 ;;
esac

if [ ! -d "$LOCAL_FOLDER" ]; then
    echo -e "  ${RED}Folder not found: $LOCAL_FOLDER${NC}"
    exit 1
fi

echo ""
echo -e "  Selected: ${GREEN}$LOCAL_FOLDER${NC}"
echo ""

# ============================================================================
# Step 3: Validate and Upload Dataset
# ============================================================================
echo -e "${BOLD}${BLUE}[3/4]${NC} ${BOLD}Validate & Upload Dataset${NC}"
echo "────────────────────────────────────────────────────────────────────────"
echo ""

# Find and match images with captions
declare -a MATCHED_PAIRS
declare -a UNMATCHED

for ext in $IMAGE_EXTENSIONS; do
    while IFS= read -r img; do
        if [ -n "$img" ] && [ -f "$img" ]; then
            basename="${img%.*}"
            caption="${basename}.txt"
            img_name=$(basename "$img")

            if [ -f "$caption" ]; then
                MATCHED_PAIRS+=("$img_name")
            else
                UNMATCHED+=("$img_name (no caption)")
            fi
        fi
    done < <(find "$LOCAL_FOLDER" -maxdepth 1 -iname "*.$ext" 2>/dev/null)
done

echo -e "  ${GREEN}Matched pairs:${NC} ${#MATCHED_PAIRS[@]}"

if [ ${#UNMATCHED[@]} -gt 0 ]; then
    echo -e "  ${YELLOW}Unmatched:${NC} ${#UNMATCHED[@]}"
    for item in "${UNMATCHED[@]:0:3}"; do
        echo -e "    ${YELLOW}!${NC} $item"
    done
    [ ${#UNMATCHED[@]} -gt 3 ] && echo "    ... and $((${#UNMATCHED[@]} - 3)) more"
fi

if [ ${#MATCHED_PAIRS[@]} -eq 0 ]; then
    echo -e "  ${RED}No matched image/caption pairs found!${NC}"
    echo "  Expected: image.jpg + image.txt (same basename)"
    exit 1
fi

echo ""

# Clear remote and upload
read -p "  Clear existing remote data and upload? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo ""
    echo -e "  Uploading ${#MATCHED_PAIRS[@]} pairs..."

    # Clear remote
    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p "$REMOTE_PORT" "${REMOTE_USER}@${REMOTE_HOST}" \
        "rm -rf ${REMOTE_DATA_DIR}/* && mkdir -p ${REMOTE_DATA_DIR}"

    # Upload files
    uploaded=0
    for img_name in "${MATCHED_PAIRS[@]}"; do
        basename="${img_name%.*}"
        img_path="$LOCAL_FOLDER/$img_name"
        txt_path="$LOCAL_FOLDER/${basename}.txt"

        scp -i "$SSH_KEY" -P "$REMOTE_PORT" -q "$img_path" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DATA_DIR}/"
        scp -i "$SSH_KEY" -P "$REMOTE_PORT" -q "$txt_path" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DATA_DIR}/"

        uploaded=$((uploaded + 1))
        printf "\r  Uploaded: %d/%d" "$uploaded" "${#MATCHED_PAIRS[@]}"
    done
    echo ""
    echo -e "  ${GREEN}✓${NC} Upload complete"
fi

echo ""

# ============================================================================
# Step 4: Configure and Launch Training
# ============================================================================
echo -e "${BOLD}${BLUE}[4/4]${NC} ${BOLD}Training Configuration${NC}"
echo "────────────────────────────────────────────────────────────────────────"
echo ""

# Training profile selection
echo "  Training profiles:"
echo ""
echo -e "  ${BOLD}[1]${NC} Fast iteration  - 512px, 1500 steps, ~1 hour   ${GREEN}(recommended for testing)${NC}"
echo -e "  ${BOLD}[2]${NC} Production      - 768px, 2500 steps, ~3 hours"
echo -e "  ${BOLD}[3]${NC} Custom settings"
echo ""

read -p "  Select profile [1]: " profile_choice
profile_choice="${profile_choice:-1}"

case "$profile_choice" in
    1) PROFILE="fast" ;;
    2) PROFILE="final" ;;
    3)
        PROFILE="custom"
        read -p "  Resolution (512/768/1024) [512]: " CUSTOM_RES
        read -p "  Max steps [1500]: " CUSTOM_STEPS
        read -p "  LoRA rank (32/64/128) [32]: " CUSTOM_RANK
        CUSTOM_RES="${CUSTOM_RES:-512}"
        CUSTOM_STEPS="${CUSTOM_STEPS:-1500}"
        CUSTOM_RANK="${CUSTOM_RANK:-32}"
        ;;
    *) PROFILE="fast" ;;
esac

# Run name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DEFAULT_NAME="lora_${TIMESTAMP}"
read -p "  Run name [${DEFAULT_NAME}]: " RUN_NAME
RUN_NAME="${RUN_NAME:-$DEFAULT_NAME}"

echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo -e "  ${BOLD}Summary:${NC}"
echo -e "    Profile:  ${CYAN}${PROFILE}${NC}"
echo -e "    Run name: ${CYAN}${RUN_NAME}${NC}"
echo -e "    Dataset:  ${CYAN}${#MATCHED_PAIRS[@]} images${NC}"
echo -e "    GPU:      ${CYAN}${GPU_INFO}${NC}"
echo "────────────────────────────────────────────────────────────────────────"
echo ""

read -p "  Start training? [Y/n] " -n 1 -r
echo

if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Launching Training Dashboard...                                     ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Build remote command
if [ "$PROFILE" = "custom" ]; then
    REMOTE_CMD="cd /workspace/lora_training && RESOLUTION=${CUSTOM_RES} MAX_STEPS=${CUSTOM_STEPS} RANK=${CUSTOM_RANK} RUN_NAME=${RUN_NAME} bash scripts/train_dashboard.sh"
else
    REMOTE_CMD="cd /workspace/lora_training && PROFILE=${PROFILE} RUN_NAME=${RUN_NAME} bash scripts/train_dashboard.sh"
fi

# Connect and launch
ssh -t -o StrictHostKeyChecking=no -i "$SSH_KEY" -p "$REMOTE_PORT" "${REMOTE_USER}@${REMOTE_HOST}" "$REMOTE_CMD"

#!/bin/bash
# ============================================================================
# Training Dashboard - Live Training Monitor
# ============================================================================
# Shows real-time training progress with:
#   - Step/loss progress bar
#   - GPU telemetry
#   - Sample images (via viu)
#   - ETA countdown
#   - Full logging
#
# Uses scripts/build_train_cmd.py for command generation (Single Source of Truth)
# Configuration is loaded from configs/flux_fast.toml or configs/flux_final.toml
#
# Usage: PROFILE=fast RUN_NAME=my_lora bash scripts/train_dashboard.sh
# ============================================================================

# Exit on Ctrl+C gracefully
trap cleanup EXIT INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
export WORKSPACE
source "${WORKSPACE}/docker/env.sh" 2>/dev/null || true

# ============================================================================
# Configuration
# ============================================================================
PROFILE="${PROFILE:-fast}"
RUN_NAME="${RUN_NAME:-flux_lora_$(date +%Y%m%d_%H%M%S)}"
export RUN_NAME

LOG_DIR="${WORKSPACE}/logs"
OUTPUT_DIR="${WORKSPACE}/output"
SAMPLES_DIR="${OUTPUT_DIR}/samples/${RUN_NAME}"

# Create directories
mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$SAMPLES_DIR"

# Log files
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}_train.log"
FULL_LOG="${LOG_DIR}/${RUN_NAME}_full.log"
METRICS_LOG="${LOG_DIR}/${RUN_NAME}_metrics.json"

# Training parameters read from TOML via build_train_cmd.py
# These are for display purposes; actual values come from TOML
if [ "$PROFILE" = "final" ]; then
    RESOLUTION="${RESOLUTION:-768}"
    MAX_STEPS="${MAX_STEPS:-2500}"
    RANK="${RANK:-64}"
    WARMUP="${WARMUP:-500}"
else
    RESOLUTION="${RESOLUTION:-512}"
    MAX_STEPS="${MAX_STEPS:-1500}"
    RANK="${RANK:-32}"
    WARMUP="${WARMUP:-100}"
fi

# ============================================================================
# Colors and UI Elements
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# Clear screen and hide cursor
clear_screen() { printf "\033[2J\033[H"; }
hide_cursor() { printf "\033[?25l"; }
show_cursor() { printf "\033[?25h"; }
move_cursor() { printf "\033[%d;%dH" "$1" "$2"; }

# Progress bar (ASCII for compatibility)
progress_bar() {
    local current=${1:-0}
    local total=${2:-1}
    local width=50

    # Handle edge cases
    [ "$current" -lt 0 ] && current=0
    [ "$total" -le 0 ] && total=1
    [ "$current" -gt "$total" ] && current=$total

    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    printf "${CYAN}["
    printf "%${filled}s" | tr ' ' '='
    if [ "$filled" -gt 0 ] && [ "$filled" -lt "$width" ]; then
        printf ">"
        empty=$((empty - 1))
    fi
    printf "%${empty}s" | tr ' ' '-'
    printf "]${NC} %3d%%" "$percent"
}

# ============================================================================
# Cleanup function
# ============================================================================
cleanup() {
    show_cursor
    # Clean up state file
    rm -f "$STATE_FILE" 2>/dev/null

    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo ""
        echo -e "${YELLOW}Stopping training...${NC}"
        kill "$TRAIN_PID" 2>/dev/null
        wait "$TRAIN_PID" 2>/dev/null
    fi
    echo ""
    echo -e "${GREEN}Training logs saved to:${NC}"
    echo "  $FULL_LOG"
    echo ""
}

# ============================================================================
# GPU Telemetry
# ============================================================================
get_gpu_stats() {
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1
}

format_gpu_stats() {
    local stats="$1"
    if [ -z "$stats" ]; then
        echo "GPU: N/A"
        return
    fi

    IFS=',' read -r util mem_used mem_total temp power <<< "$stats"
    util=$(echo "$util" | tr -d ' ')
    mem_used=$(echo "$mem_used" | tr -d ' ')
    mem_total=$(echo "$mem_total" | tr -d ' ')
    temp=$(echo "$temp" | tr -d ' ')
    power=$(echo "$power" | tr -d ' ')

    local mem_percent=$((mem_used * 100 / mem_total))

    # Color coding
    local util_color="${GREEN}"
    [ "$util" -gt 80 ] && util_color="${YELLOW}"
    [ "$util" -gt 95 ] && util_color="${RED}"

    local temp_color="${GREEN}"
    [ "$temp" -gt 70 ] && temp_color="${YELLOW}"
    [ "$temp" -gt 85 ] && temp_color="${RED}"

    printf "GPU: ${util_color}%3d%%${NC} | VRAM: ${CYAN}%d${NC}/${CYAN}%d${NC}MB (%d%%) | Temp: ${temp_color}%d C${NC} | Power: ${DIM}%sW${NC}" \
        "$util" "$mem_used" "$mem_total" "$mem_percent" "$temp" "$power"
}

# ============================================================================
# State File (for cross-process communication)
# ============================================================================
STATE_FILE="/tmp/train_dashboard_$$_state"

write_state() {
    echo "CURRENT_STEP=$CURRENT_STEP" > "$STATE_FILE"
    echo "CURRENT_LOSS=$CURRENT_LOSS" >> "$STATE_FILE"
    echo "GENERATING_SAMPLES=$GENERATING_SAMPLES" >> "$STATE_FILE"
    echo "SAVING_CHECKPOINT=$SAVING_CHECKPOINT" >> "$STATE_FILE"
}

read_state() {
    if [ -f "$STATE_FILE" ]; then
        source "$STATE_FILE"
    fi
}

# ============================================================================
# Parse Training Output
# ============================================================================
# Kohya-ss tqdm output format:
# steps:  10%|...| 150/1500 [00:45<06:45,  3.33it/s, avr_loss=0.234]
parse_training_log() {
    local log_file="$1"

    if [ ! -f "$log_file" ]; then
        return
    fi

    # Get the last 100 lines to find progress info
    local recent_lines=$(tail -100 "$log_file" 2>/dev/null)

    # Parse step/total from tqdm progress bar
    # Formats seen:
    #   steps:  10%|...| 150/1500 [time]
    #   10%|...| 15/150 [time]
    # Look for pattern: NUMBER/NUMBER followed by space and [
    local step_match=$(echo "$recent_lines" | grep -oE '[0-9]+/[0-9]+[[:space:]]*\[' | tail -1)
    if [ -n "$step_match" ]; then
        # Extract "150/1500" part
        local nums=$(echo "$step_match" | grep -oE '[0-9]+/[0-9]+')
        CURRENT_STEP=$(echo "$nums" | cut -d'/' -f1)
        local total=$(echo "$nums" | cut -d'/' -f2)
        if [ -n "$total" ] && [ "$total" -gt 0 ]; then
            TOTAL_STEPS="$total"
        fi
    fi

    # Parse loss from tqdm: "avr_loss=0.234" or "loss=0.234"
    local loss_match=$(echo "$recent_lines" | grep -oE 'avr_loss=[0-9]+\.[0-9]+' | tail -1)
    if [ -n "$loss_match" ]; then
        CURRENT_LOSS=$(echo "$loss_match" | cut -d'=' -f2)
    fi

    # Detect current phase from recent output (last 20 lines for status)
    local status_lines=$(echo "$recent_lines" | tail -20)

    # Reset status flags
    GENERATING_SAMPLES=0
    SAVING_CHECKPOINT=0
    LOADING_MODEL=0
    CACHING=0

    # Check for specific status messages (be very specific to avoid false positives)
    if echo "$status_lines" | grep -qi "generating sample\|sample images saved"; then
        GENERATING_SAMPLES=1
    elif echo "$status_lines" | grep -qi "saving checkpoint:\|model saved"; then
        SAVING_CHECKPOINT=1
    elif echo "$status_lines" | grep -qi "caching latents\|caching text encoder"; then
        CACHING=1
    elif echo "$status_lines" | grep -qi "loading model\|loading weights\|Loading pipe"; then
        LOADING_MODEL=1
    fi
}

# ============================================================================
# Display Sample Images
# ============================================================================
display_latest_samples() {
    local samples_path="${OUTPUT_DIR}/samples"
    local latest_samples=$(find "$samples_path" -name "*.png" -mmin -1 2>/dev/null | sort -r | head -3)

    if [ -n "$latest_samples" ]; then
        echo ""
        echo -e "${MAGENTA}--- Latest Samples ---${NC}"
        for img in $latest_samples; do
            local img_name=$(basename "$img")
            echo -e "${DIM}$img_name${NC}"
            viu -w 60 -h 15 "$img" 2>/dev/null || echo "[Could not display image]"
            echo ""
        done
        LAST_SAMPLE_SHOWN=$(date +%s)
    fi
}

# ============================================================================
# Main Dashboard Display
# ============================================================================
display_dashboard() {
    local gpu_stats=$(get_gpu_stats)
    local elapsed=$(($(date +%s) - START_TIME))
    local elapsed_fmt=$(printf '%02d:%02d:%02d' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))

    # Ensure numeric values (default to 0 if empty/invalid)
    local step=${CURRENT_STEP:-0}
    local total=${TOTAL_STEPS:-$MAX_STEPS}
    local loss=${CURRENT_LOSS:-0.0000}

    # Ensure step is numeric
    [[ ! "$step" =~ ^[0-9]+$ ]] && step=0
    [[ ! "$total" =~ ^[0-9]+$ ]] && total=$MAX_STEPS

    # Calculate ETA
    local eta_str="--:--:--"
    if [ "$step" -gt 0 ] && [ "$step" -lt "$total" ] && [ "$elapsed" -gt 0 ]; then
        local steps_remaining=$((total - step))
        local secs_per_step=$((elapsed / step))
        local eta_secs=$((steps_remaining * secs_per_step))
        eta_str=$(printf '%02d:%02d:%02d' $((eta_secs/3600)) $((eta_secs%3600/60)) $((eta_secs%60)))
    fi

    # Clear and draw
    clear_screen

    # Header
    echo -e "${CYAN}+============================================================================+${NC}"
    echo -e "${CYAN}|${NC}  ${BOLD}FLUX.1-dev LoRA Training${NC}                                               ${CYAN}|${NC}"
    printf "${CYAN}|${NC}  ${DIM}Run: %-60s${NC}  ${CYAN}|${NC}\n" "${RUN_NAME}"
    echo -e "${CYAN}+============================================================================+${NC}"

    # Progress section
    echo -e "${CYAN}|${NC}                                                                            ${CYAN}|${NC}"
    printf "${CYAN}|${NC}  "
    progress_bar "$step" "$total"
    printf "                    ${CYAN}|${NC}\n"
    echo -e "${CYAN}|${NC}                                                                            ${CYAN}|${NC}"

    # Stats line
    printf "${CYAN}|${NC}  Step: ${BOLD}%5d${NC}/${BOLD}%-5d${NC} | Loss: ${YELLOW}%-8s${NC} | Elapsed: ${GREEN}%s${NC} | ETA: ${MAGENTA}%s${NC}  ${CYAN}|${NC}\n" \
        "$step" "$total" "$loss" "$elapsed_fmt" "$eta_str"

    echo -e "${CYAN}|${NC}                                                                            ${CYAN}|${NC}"
    echo -e "${CYAN}+============================================================================+${NC}"

    # GPU stats
    printf "${CYAN}|${NC}  "
    format_gpu_stats "$gpu_stats"
    printf "  ${CYAN}|${NC}\n"

    echo -e "${CYAN}+============================================================================+${NC}"

    # Status line - determine current phase
    local status_msg="Training..."
    local status_color="${GREEN}"
    if [ "$GENERATING_SAMPLES" = "1" ]; then
        status_msg="Generating samples..."
        status_color="${MAGENTA}"
    elif [ "$SAVING_CHECKPOINT" = "1" ]; then
        status_msg="Saving checkpoint..."
        status_color="${YELLOW}"
    elif [ "$CACHING" = "1" ]; then
        status_msg="Caching latents..."
        status_color="${BLUE}"
    elif [ "$LOADING_MODEL" = "1" ]; then
        status_msg="Loading models..."
        status_color="${BLUE}"
    elif [ "$step" -eq 0 ]; then
        status_msg="Initializing..."
        status_color="${YELLOW}"
    fi
    printf "${CYAN}|${NC}  Status: ${status_color}%-20s${NC}  Profile: ${CYAN}%-10s${NC}  Rank: ${CYAN}%d${NC}          ${CYAN}|${NC}\n" \
        "$status_msg" "$PROFILE" "$RANK"

    echo -e "${CYAN}+============================================================================+${NC}"
    echo ""

    # Show recent samples if any
    display_latest_samples

    # Footer
    echo ""
    echo -e "${DIM}Press Ctrl+C to stop training${NC}"
    echo -e "${DIM}Full log: tail -f ${FULL_LOG}${NC}"
}

# ============================================================================
# Build Training Command using Single Source of Truth
# ============================================================================
# P1 Fix: Use build_train_cmd.py so dashboard uses same command as launch scripts
build_training_command() {
    # Build command using the canonical command builder (Single Source of Truth)
    python3 "${SCRIPT_DIR}/build_train_cmd.py" --profile "${PROFILE}" --run-name "${RUN_NAME}"
}

# ============================================================================
# Main Execution
# ============================================================================
main() {
    # Initialize variables
    CURRENT_STEP=0
    CURRENT_LOSS=0
    GENERATING_SAMPLES=0
    SAVING_CHECKPOINT=0
    LOADING_MODEL=0
    CACHING=0
    START_TIME=$(date +%s)
    TOTAL_STEPS=$MAX_STEPS

    # Clean up state file on exit
    trap "rm -f $STATE_FILE" EXIT

    # Log start
    {
        echo "========================================"
        echo "Training started: $(date)"
        echo "Run name: $RUN_NAME"
        echo "Profile: $PROFILE"
        echo "Resolution: $RESOLUTION"
        echo "Max steps: $MAX_STEPS"
        echo "Rank: $RANK"
        echo "Config source: configs/flux_${PROFILE}.toml (via build_train_cmd.py)"
        echo "========================================"
    } >> "$FULL_LOG"

    hide_cursor

    # Build command using Single Source of Truth (P1 fix)
    echo -e "${BLUE}[INFO]${NC} Building command from configs/flux_${PROFILE}.toml..."
    TRAIN_CMD=$(build_training_command)

    if [ $? -ne 0 ]; then
        show_cursor
        echo -e "${RED}[ERROR]${NC} Failed to build training command"
        exit 1
    fi

    echo "$TRAIN_CMD" >> "$FULL_LOG"

    # ============================================================================
    # Save reproducibility artifacts (P1 fix)
    # ============================================================================
    echo "${TRAIN_CMD}" > "${LOG_DIR}/${RUN_NAME}_command.txt"
    python3 "${SCRIPT_DIR}/build_train_cmd.py" --profile "${PROFILE}" --run-name "${RUN_NAME}" --show-repro > "${LOG_DIR}/${RUN_NAME}_repro.json" 2>/dev/null || true
    pip freeze > "${LOG_DIR}/${RUN_NAME}_pip_freeze.txt" 2>/dev/null || true

    # Start training in background, logging to file
    echo "Starting training..." >> "$FULL_LOG"

    # Run training with unbuffered Python output
    # PYTHONUNBUFFERED=1 ensures tqdm progress appears immediately
    PYTHONUNBUFFERED=1 bash -c "$TRAIN_CMD" >> "$FULL_LOG" 2>&1 &
    TRAIN_PID=$!

    # Also create a symlink for the train log
    ln -sf "$FULL_LOG" "$TRAIN_LOG" 2>/dev/null || cp "$FULL_LOG" "$TRAIN_LOG" 2>/dev/null

    # Wait a moment for training to start
    sleep 3

    # Monitor and display dashboard
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        # Parse the log file to extract current state
        parse_training_log "$FULL_LOG"

        # Display the dashboard
        display_dashboard

        # Update interval (2 seconds)
        sleep 2
    done

    # Training finished - get exit code
    wait "$TRAIN_PID"
    EXIT_CODE=$?

    # Final parse to get last state
    parse_training_log "$FULL_LOG"

    show_cursor

    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}+============================================================================+${NC}"
        echo -e "${GREEN}|                        TRAINING COMPLETE                                   |${NC}"
        echo -e "${GREEN}+============================================================================+${NC}"
    else
        echo -e "${RED}+============================================================================+${NC}"
        echo -e "${RED}|                        TRAINING FAILED                                     |${NC}"
        echo -e "${RED}+============================================================================+${NC}"
    fi
    echo ""
    echo -e "  ${BOLD}Final Step:${NC}  ${CURRENT_STEP}/${TOTAL_STEPS}"
    echo -e "  ${BOLD}Final Loss:${NC}  ${CURRENT_LOSS}"
    echo -e "  ${BOLD}Output:${NC}      ${OUTPUT_DIR}/${RUN_NAME}.safetensors"
    echo -e "  ${BOLD}Samples:${NC}     ${OUTPUT_DIR}/samples/"
    echo -e "  ${BOLD}Logs:${NC}        ${LOG_DIR}/"
    echo ""

    # Show final samples
    echo -e "${MAGENTA}Final samples:${NC}"
    for img in $(ls -t "${OUTPUT_DIR}/samples/"*.png 2>/dev/null | head -5); do
        echo -e "${DIM}$(basename $img)${NC}"
        viu -w 60 "$img" 2>/dev/null
        echo ""
    done

    # Log completion
    {
        echo "========================================"
        echo "Training completed: $(date)"
        echo "Exit code: $EXIT_CODE"
        echo "========================================"
    } >> "$FULL_LOG"

    exit $EXIT_CODE
}

# Run
main

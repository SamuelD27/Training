#!/bin/bash
# ============================================================================
# Watch and Display Sample Images
# ============================================================================
# Monitors the samples directory and displays new images in terminal using viu.
# Run this in a separate terminal/tmux pane during training.
#
# Usage: bash scripts/watch_samples.sh [samples_dir]
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SAMPLES_DIR="${1:-/workspace/lora_training/output/samples}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"  # Check every 5 seconds

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       Sample Image Viewer                                      ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Watching:${NC} ${SAMPLES_DIR}"
echo -e "${BLUE}Interval:${NC} ${POLL_INTERVAL}s"
echo ""

# Check for viu
if ! command -v viu &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} viu not found. Install with:"
    echo "  curl -sSL https://github.com/atanunq/viu/releases/download/v1.5.0/viu-x86_64-unknown-linux-musl -o /usr/local/bin/viu && chmod +x /usr/local/bin/viu"
    exit 1
fi

# Create samples directory if it doesn't exist
mkdir -p "$SAMPLES_DIR"

# Track displayed files
DISPLAYED_FILE="/tmp/displayed_samples.txt"
touch "$DISPLAYED_FILE"

echo -e "${YELLOW}Waiting for sample images...${NC}"
echo "(Press Ctrl+C to stop)"
echo ""

# Function to display a single image
display_image() {
    local img="$1"
    local basename=$(basename "$img")

    # Extract step number from filename if present
    local step_info=""
    if [[ $basename =~ ([0-9]+) ]]; then
        step_info=" (Step ${BASH_REMATCH[1]})"
    fi

    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}NEW SAMPLE:${NC} ${basename}${step_info}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Display image with viu (width auto-detected, or specify with -w)
    viu -w 80 "$img" 2>/dev/null || viu "$img"

    echo ""
}

# Function to display all images in grid view
display_latest() {
    local count="${1:-5}"

    echo ""
    echo -e "${CYAN}Latest ${count} samples:${NC}"
    echo ""

    # Get latest images
    local images=($(ls -t "$SAMPLES_DIR"/*.png "$SAMPLES_DIR"/*.jpg 2>/dev/null | head -n "$count"))

    if [ ${#images[@]} -eq 0 ]; then
        echo -e "${YELLOW}No samples yet.${NC}"
        return
    fi

    for img in "${images[@]}"; do
        display_image "$img"
    done
}

# Main watch loop
while true; do
    # Find new PNG/JPG files
    for img in "$SAMPLES_DIR"/*.png "$SAMPLES_DIR"/*.jpg 2>/dev/null; do
        if [ -f "$img" ]; then
            # Check if already displayed
            if ! grep -qF "$img" "$DISPLAYED_FILE" 2>/dev/null; then
                display_image "$img"
                echo "$img" >> "$DISPLAYED_FILE"
            fi
        fi
    done

    sleep "$POLL_INTERVAL"
done

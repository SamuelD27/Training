#!/bin/bash
# ============================================================================
# tmux Training Orchestrator
# ============================================================================
# Creates a tmux session with 3 panes:
#   1. Training command (fast or final)
#   2. GPU telemetry
#   3. Log tail
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

# Default profile
PROFILE="${1:-fast}"
SESSION_NAME="${SESSION_NAME:-lora_training}"

# Validate profile
if [ "$PROFILE" != "fast" ] && [ "$PROFILE" != "final" ]; then
    echo -e "${RED}[ERROR]${NC} Invalid profile: ${PROFILE}"
    echo "Usage: $0 [fast|final]"
    exit 1
fi

# Generate run name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="${RUN_NAME:-flux_${PROFILE}_${TIMESTAMP}}"
LOG_FILE="${WORKSPACE}/logs/${RUN_NAME}.log"

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       tmux Training Orchestrator                               ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Profile:${NC}      ${PROFILE}"
echo -e "${BLUE}Run Name:${NC}     ${RUN_NAME}"
echo -e "${BLUE}Session:${NC}      ${SESSION_NAME}"
echo -e "${BLUE}Log File:${NC}     ${LOG_FILE}"
echo ""

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} tmux is not installed"
    echo "Install with: apt-get install tmux"
    exit 1
fi

# Check if session already exists
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo -e "${YELLOW}[WARN]${NC} Session '${SESSION_NAME}' already exists"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t ${SESSION_NAME}"
    echo "  2. Kill existing session:      tmux kill-session -t ${SESSION_NAME}"
    echo "  3. Use different session name: SESSION_NAME=my_session $0 ${PROFILE}"
    echo ""
    read -p "Attach to existing session? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux attach -t "${SESSION_NAME}"
        exit 0
    fi
    exit 1
fi

# Create log directory
mkdir -p "${WORKSPACE}/logs"

# Create empty log file (for tail to watch)
touch "${LOG_FILE}"

# Training command based on profile
if [ "$PROFILE" = "fast" ]; then
    TRAIN_CMD="cd ${WORKSPACE} && RUN_NAME=${RUN_NAME} bash scripts/train_flux_fast.sh"
else
    TRAIN_CMD="cd ${WORKSPACE} && RUN_NAME=${RUN_NAME} bash scripts/train_flux_final.sh"
fi

# Create tmux session
echo -e "${BLUE}[INFO]${NC} Creating tmux session '${SESSION_NAME}'..."

# Create session with training pane
tmux new-session -d -s "${SESSION_NAME}" -n "training"

# Set up panes
# Main pane (0): Training
tmux send-keys -t "${SESSION_NAME}:0.0" "echo 'Training will start shortly...'; sleep 2; ${TRAIN_CMD}" Enter

# Split horizontally for telemetry
tmux split-window -h -t "${SESSION_NAME}:0"
tmux send-keys -t "${SESSION_NAME}:0.1" "cd ${WORKSPACE} && bash scripts/telemetry.sh" Enter

# Split the right pane vertically for log tail
tmux split-window -v -t "${SESSION_NAME}:0.1"
tmux send-keys -t "${SESSION_NAME}:0.2" "echo 'Waiting for log file...'; sleep 3; tail -f ${LOG_FILE}" Enter

# Set pane layout (left = training, right top = telemetry, right bottom = log)
tmux select-layout -t "${SESSION_NAME}:0" main-vertical

# Resize panes (training gets more space)
tmux resize-pane -t "${SESSION_NAME}:0.0" -x 100

# Select the training pane
tmux select-pane -t "${SESSION_NAME}:0.0"

echo ""
echo -e "${GREEN}[SUCCESS]${NC} tmux session '${SESSION_NAME}' created"
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                      PANE LAYOUT                               ║${NC}"
echo -e "${CYAN}╠════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║  ┌─────────────────────┬───────────────────┐                   ║${NC}"
echo -e "${CYAN}║  │                     │    TELEMETRY      │                   ║${NC}"
echo -e "${CYAN}║  │     TRAINING        │   (GPU stats)     │                   ║${NC}"
echo -e "${CYAN}║  │     (main)          ├───────────────────┤                   ║${NC}"
echo -e "${CYAN}║  │                     │    LOG TAIL       │                   ║${NC}"
echo -e "${CYAN}║  │                     │   (live log)      │                   ║${NC}"
echo -e "${CYAN}║  └─────────────────────┴───────────────────┘                   ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}To attach to the session:${NC}"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
echo -e "${YELLOW}tmux key bindings:${NC}"
echo "  Ctrl+b d       - Detach from session (training continues)"
echo "  Ctrl+b arrow   - Switch between panes"
echo "  Ctrl+b z       - Zoom current pane (toggle)"
echo "  Ctrl+b [       - Enter scroll mode (q to exit)"
echo "  Ctrl+b x       - Kill current pane"
echo ""
echo -e "${YELLOW}To kill the session:${NC}"
echo "  tmux kill-session -t ${SESSION_NAME}"
echo ""

# Ask to attach
read -p "Attach to session now? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    tmux attach -t "${SESSION_NAME}"
fi

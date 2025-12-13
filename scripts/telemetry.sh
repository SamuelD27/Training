#!/bin/bash
# ============================================================================
# GPU Telemetry Script
# ============================================================================
# Displays real-time GPU utilization, memory, and top processes.
# Prefers nvitop if available, falls back to nvidia-smi.
# ============================================================================

# Colors
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}Starting GPU Telemetry...${NC}"
echo "Press Ctrl+C to exit"
echo ""

# Check for nvitop (better interface)
if command -v nvitop &> /dev/null; then
    echo "Using nvitop..."
    nvitop
# Check for nvtop
elif command -v nvtop &> /dev/null; then
    echo "Using nvtop..."
    nvtop
# Fallback to nvidia-smi watch
else
    echo "Using nvidia-smi (install nvitop for better interface: pip install nvitop)"
    echo ""

    # Custom nvidia-smi display with watch
    watch -n 1 'echo "=== GPU Telemetry ===" && \
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits | \
        awk -F", " "{printf \"GPU %s: %s\\n  Temp: %s°C | GPU Util: %s%% | Mem Util: %s%%\\n  VRAM: %s / %s MiB | Power: %sW\\n\\n\", \$1, \$2, \$3, \$4, \$5, \$6, \$7, \$8}" && \
        echo "=== Top GPU Processes ===" && \
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -5 || echo "No GPU processes"'
fi

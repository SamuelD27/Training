#!/bin/bash
# ============================================================================
# Sync Models from R2 to Pod
# ============================================================================
# Run this on your RunPod instance to sync models from R2
# Much faster than downloading from HuggingFace (~10x speedup)
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# R2 Configuration
R2_BUCKET="${R2_BUCKET:-storage-training}"
R2_ENDPOINT="${R2_ENDPOINT:-https://e6b3925ef3896465b73c442be466db90.r2.cloudflarestorage.com}"
R2_ACCESS_KEY="${R2_ACCESS_KEY:-4fcb7a2f5b18934a841f1c45860c1343}"
R2_SECRET_KEY="${R2_SECRET_KEY:-75f51f60a84d6e4d554ec876bbd8b9d2dbae114d2298085d91655afbd75a8897}"

# Destination
MODEL_DIR="${MODEL_DIR:-/workspace/models/flux1-dev}"

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       Sync FLUX.1-dev from R2                                  ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# Install rclone if needed
# ============================================================================
if ! command -v rclone &> /dev/null; then
    echo -e "${BLUE}[INFO]${NC} Installing rclone..."
    curl -s https://rclone.org/install.sh | bash
fi

# ============================================================================
# Configure rclone via environment (no config file needed)
# ============================================================================
export RCLONE_CONFIG_R2_TYPE=s3
export RCLONE_CONFIG_R2_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="${R2_ACCESS_KEY}"
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="${R2_SECRET_KEY}"
export RCLONE_CONFIG_R2_ENDPOINT="${R2_ENDPOINT}"

# ============================================================================
# Check R2 connection
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Checking R2 connection..."
if rclone lsd r2: &>/dev/null; then
    echo -e "${GREEN}  ✓${NC} R2 connection OK"
else
    echo -e "${RED}[ERROR]${NC} Cannot connect to R2"
    exit 1
fi

# Check if model exists in R2
echo -e "${BLUE}[INFO]${NC} Checking model in R2..."
R2_SIZE=$(rclone size "r2:${R2_BUCKET}/models/flux1-dev" 2>/dev/null | grep "Total size" || echo "")
if [ -z "$R2_SIZE" ]; then
    echo -e "${RED}[ERROR]${NC} Model not found in r2:${R2_BUCKET}/models/flux1-dev"
    echo "Run scripts/download_models.sh on your local machine first"
    exit 1
fi
echo -e "${GREEN}  ✓${NC} Model found: ${R2_SIZE}"

# ============================================================================
# Sync model
# ============================================================================
echo ""
echo -e "${BLUE}[INFO]${NC} Syncing model to ${MODEL_DIR}..."
echo -e "${YELLOW}  This should take 2-5 minutes on a fast connection${NC}"
echo ""

mkdir -p "${MODEL_DIR}"

# Optimized rclone settings for fast download
rclone sync "r2:${R2_BUCKET}/models/flux1-dev" "${MODEL_DIR}" \
    --progress \
    --transfers 16 \
    --checkers 32 \
    --buffer-size 256M \
    --s3-chunk-size 64M \
    --fast-list

echo ""
echo -e "${GREEN}  ✓${NC} Sync complete!"
echo ""

# Verify
echo -e "${BLUE}[INFO]${NC} Verifying..."
if [ -f "${MODEL_DIR}/flux1-dev.safetensors" ]; then
    SIZE=$(du -h "${MODEL_DIR}/flux1-dev.safetensors" | cut -f1)
    echo -e "${GREEN}  ✓${NC} flux1-dev.safetensors (${SIZE})"
else
    echo -e "${RED}  ✗${NC} flux1-dev.safetensors missing!"
    exit 1
fi

TOTAL=$(du -sh "${MODEL_DIR}" | cut -f1)
echo -e "${GREEN}  ✓${NC} Total model size: ${TOTAL}"
echo ""
echo -e "${CYAN}Model ready at: ${MODEL_DIR}${NC}"
echo ""

#!/bin/bash
# ============================================================================
# FLUX.1-dev Model Download & R2 Upload Script
# ============================================================================
# Downloads models using aria2 (parallel) + hf_transfer (fast HF downloads)
# Then uploads to Cloudflare R2 bucket for fast pod sync
#
# Prerequisites:
#   brew install aria2 rclone jq
#   pip install huggingface_hub hf_transfer
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
HF_TOKEN="${HF_TOKEN:-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}"
R2_BUCKET="storage-training"
R2_ENDPOINT="https://e6b3925ef3896465b73c442be466db90.r2.cloudflarestorage.com"
R2_ACCESS_KEY="4fcb7a2f5b18934a841f1c45860c1343"
R2_SECRET_KEY="75f51f60a84d6e4d554ec876bbd8b9d2dbae114d2298085d91655afbd75a8897"

# Local download directory
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$HOME/models}"
MODEL_NAME="black-forest-labs/FLUX.1-dev"
LOCAL_MODEL_DIR="${DOWNLOAD_DIR}/flux1-dev"

# Aria2 optimization for 2Gbps connection
ARIA2_OPTS="-x 16 -s 16 -k 1M --max-connection-per-server=16 --min-split-size=1M --file-allocation=none"

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       FLUX.1-dev Model Download & R2 Upload                    ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# Step 0: Check prerequisites
# ============================================================================
echo -e "${BLUE}[0/5]${NC} Checking prerequisites..."

check_cmd() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}[ERROR]${NC} $1 not found. Install with: $2"
        exit 1
    fi
    echo -e "${GREEN}  ✓${NC} $1"
}

check_cmd "aria2c" "brew install aria2"
check_cmd "rclone" "brew install rclone"
check_cmd "jq" "brew install jq"

# Check Python packages
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo -e "${YELLOW}[WARN]${NC} huggingface_hub not found, installing..."
    pip3 install huggingface_hub hf_transfer
fi

echo ""

# ============================================================================
# Step 1: Configure rclone for R2
# ============================================================================
echo -e "${BLUE}[1/5]${NC} Configuring rclone for R2..."

RCLONE_CONFIG_DIR="${HOME}/.config/rclone"
mkdir -p "${RCLONE_CONFIG_DIR}"

# Add/update R2 config
if ! grep -q "\[r2\]" "${RCLONE_CONFIG_DIR}/rclone.conf" 2>/dev/null; then
    cat >> "${RCLONE_CONFIG_DIR}/rclone.conf" << EOF

[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY}
secret_access_key = ${R2_SECRET_KEY}
endpoint = ${R2_ENDPOINT}
acl = private
EOF
    echo -e "${GREEN}  ✓${NC} rclone R2 config added"
else
    echo -e "${GREEN}  ✓${NC} rclone R2 config exists"
fi

# Test R2 connection
if rclone lsd r2: &>/dev/null; then
    echo -e "${GREEN}  ✓${NC} R2 connection verified"
else
    echo -e "${RED}[ERROR]${NC} Cannot connect to R2. Check credentials."
    exit 1
fi

echo ""

# ============================================================================
# Step 2: Create R2 bucket if needed
# ============================================================================
echo -e "${BLUE}[2/5]${NC} Checking R2 bucket..."

if rclone lsd r2: | grep -q "${R2_BUCKET}"; then
    echo -e "${GREEN}  ✓${NC} Bucket '${R2_BUCKET}' exists"
else
    echo -e "${YELLOW}  Creating bucket '${R2_BUCKET}'...${NC}"
    rclone mkdir "r2:${R2_BUCKET}"
    echo -e "${GREEN}  ✓${NC} Bucket created"
fi

echo ""

# ============================================================================
# Step 3: Download FLUX.1-dev using hf_transfer (fastest method)
# ============================================================================
echo -e "${BLUE}[3/5]${NC} Downloading FLUX.1-dev model..."
echo -e "${YELLOW}  Model: ${MODEL_NAME}${NC}"
echo -e "${YELLOW}  Destination: ${LOCAL_MODEL_DIR}${NC}"
echo -e "${YELLOW}  Size: ~34GB${NC}"
echo ""

mkdir -p "${LOCAL_MODEL_DIR}"

# Enable hf_transfer for maximum speed
export HF_HUB_ENABLE_HF_TRANSFER=1
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

# Download with huggingface-cli (uses hf_transfer under the hood)
echo -e "${BLUE}[INFO]${NC} Using hf_transfer for optimized download..."

huggingface-cli download "${MODEL_NAME}" \
    --local-dir "${LOCAL_MODEL_DIR}" \
    --local-dir-use-symlinks False \
    --token "${HF_TOKEN}"

echo ""
echo -e "${GREEN}  ✓${NC} Download complete"
echo ""

# Verify download
echo -e "${BLUE}[INFO]${NC} Verifying download..."
EXPECTED_FILES=("flux1-dev.safetensors" "ae.safetensors" "model_index.json")
for f in "${EXPECTED_FILES[@]}"; do
    if [ -f "${LOCAL_MODEL_DIR}/${f}" ]; then
        SIZE=$(du -h "${LOCAL_MODEL_DIR}/${f}" | cut -f1)
        echo -e "${GREEN}  ✓${NC} ${f} (${SIZE})"
    else
        echo -e "${RED}  ✗${NC} ${f} missing!"
    fi
done

echo ""

# ============================================================================
# Step 4: Upload to R2
# ============================================================================
echo -e "${BLUE}[4/5]${NC} Uploading to R2 bucket..."
echo -e "${YELLOW}  Source: ${LOCAL_MODEL_DIR}${NC}"
echo -e "${YELLOW}  Destination: r2:${R2_BUCKET}/models/flux1-dev/${NC}"
echo ""

# Use rclone with optimized settings for upload
rclone sync "${LOCAL_MODEL_DIR}" "r2:${R2_BUCKET}/models/flux1-dev" \
    --progress \
    --transfers 8 \
    --checkers 16 \
    --buffer-size 128M \
    --s3-upload-concurrency 8 \
    --s3-chunk-size 64M

echo ""
echo -e "${GREEN}  ✓${NC} Upload complete"
echo ""

# ============================================================================
# Step 5: Verify R2 upload
# ============================================================================
echo -e "${BLUE}[5/5]${NC} Verifying R2 upload..."

R2_SIZE=$(rclone size "r2:${R2_BUCKET}/models/flux1-dev" 2>/dev/null | grep "Total size" | awk '{print $3, $4}')
echo -e "${GREEN}  ✓${NC} R2 total size: ${R2_SIZE}"

rclone ls "r2:${R2_BUCKET}/models/flux1-dev" | head -10
echo "  ..."

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                      DOWNLOAD COMPLETE                         ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}To sync to a RunPod instance:${NC}"
echo ""
echo "  rclone sync r2:${R2_BUCKET}/models/flux1-dev /workspace/models/flux1-dev \\"
echo "    --progress --transfers 16 --checkers 32"
echo ""
echo -e "${YELLOW}Or add to pod startup script:${NC}"
echo ""
echo "  export RCLONE_CONFIG_R2_TYPE=s3"
echo "  export RCLONE_CONFIG_R2_PROVIDER=Cloudflare"
echo "  export RCLONE_CONFIG_R2_ACCESS_KEY_ID=${R2_ACCESS_KEY}"
echo "  export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=${R2_SECRET_KEY}"
echo "  export RCLONE_CONFIG_R2_ENDPOINT=${R2_ENDPOINT}"
echo "  rclone sync r2:${R2_BUCKET}/models/flux1-dev /workspace/models/flux1-dev --progress"
echo ""

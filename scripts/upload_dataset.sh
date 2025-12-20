#!/bin/bash
# ============================================================================
# Upload Dataset to RunPod Instance
# ============================================================================
# Run this locally on your Mac to select and upload training images/captions
# to your RunPod instance.
#
# Usage: bash scripts/upload_dataset.sh [local_folder] [remote_host] [remote_port]
#
# Examples:
#   bash scripts/upload_dataset.sh                          # Interactive mode
#   bash scripts/upload_dataset.sh ~/Desktop/my_dataset     # Specify folder
#   bash scripts/upload_dataset.sh ~/data host.runpod.io 22 # Full specification
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default SSH key
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-/workspace/lora_training/data/subject}"

# Image extensions to look for
IMAGE_EXTENSIONS="jpg jpeg png webp"

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       Dataset Upload to RunPod                                 ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# Step 1: Select local folder
# ============================================================================
LOCAL_FOLDER="${1:-}"

if [ -z "$LOCAL_FOLDER" ]; then
    echo -e "${BLUE}[1/4]${NC} Select local dataset folder"
    echo ""

    # Show recent folders in common locations
    echo "Recent folders:"
    echo "----------------------------------------"

    # List folders in Desktop and Downloads
    idx=1
    declare -a FOLDER_OPTIONS

    for base_dir in "$HOME/Desktop" "$HOME/Downloads" "$HOME/Documents"; do
        if [ -d "$base_dir" ]; then
            while IFS= read -r dir; do
                if [ -d "$dir" ]; then
                    # Check if folder has images
                    img_count=0
                    for ext in $IMAGE_EXTENSIONS; do
                        count=$(find "$dir" -maxdepth 1 -iname "*.$ext" 2>/dev/null | wc -l)
                        img_count=$((img_count + count))
                    done

                    if [ "$img_count" -gt 0 ]; then
                        FOLDER_OPTIONS[$idx]="$dir"
                        rel_path="${dir/#$HOME/~}"
                        echo "  [$idx] $rel_path ($img_count images)"
                        idx=$((idx + 1))
                    fi
                fi
            done < <(find "$base_dir" -maxdepth 2 -type d 2>/dev/null | head -20)
        fi
    done

    echo ""
    echo "  [p] Enter custom path"
    echo "  [q] Quit"
    echo ""

    read -p "Select folder [1-$((idx-1))/p/q]: " selection

    case "$selection" in
        q|Q)
            echo "Cancelled."
            exit 0
            ;;
        p|P)
            read -p "Enter full path: " LOCAL_FOLDER
            LOCAL_FOLDER="${LOCAL_FOLDER/#\~/$HOME}"
            ;;
        [0-9]*)
            if [ -n "${FOLDER_OPTIONS[$selection]}" ]; then
                LOCAL_FOLDER="${FOLDER_OPTIONS[$selection]}"
            else
                echo -e "${RED}[ERROR]${NC} Invalid selection"
                exit 1
            fi
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Invalid selection"
            exit 1
            ;;
    esac
fi

# Expand ~ if present
LOCAL_FOLDER="${LOCAL_FOLDER/#\~/$HOME}"

if [ ! -d "$LOCAL_FOLDER" ]; then
    echo -e "${RED}[ERROR]${NC} Folder not found: $LOCAL_FOLDER"
    exit 1
fi

echo -e "${GREEN}[OK]${NC} Selected folder: $LOCAL_FOLDER"
echo ""

# ============================================================================
# Step 2: Validate dataset (image/caption matching)
# ============================================================================
echo -e "${BLUE}[2/4]${NC} Validating dataset..."
echo ""

# Find all images
declare -a IMAGES
declare -a CAPTIONS
declare -a MATCHED
declare -a UNMATCHED_IMAGES
declare -a UNMATCHED_CAPTIONS

for ext in $IMAGE_EXTENSIONS; do
    while IFS= read -r img; do
        if [ -n "$img" ]; then
            IMAGES+=("$img")
        fi
    done < <(find "$LOCAL_FOLDER" -maxdepth 1 -iname "*.$ext" 2>/dev/null)
done

# Find all captions
while IFS= read -r txt; do
    if [ -n "$txt" ]; then
        CAPTIONS+=("$txt")
    fi
done < <(find "$LOCAL_FOLDER" -maxdepth 1 -name "*.txt" 2>/dev/null)

# Match images with captions
for img in "${IMAGES[@]}"; do
    basename="${img%.*}"
    caption="${basename}.txt"

    if [ -f "$caption" ]; then
        MATCHED+=("$(basename "$img")")
    else
        UNMATCHED_IMAGES+=("$(basename "$img")")
    fi
done

# Check for orphan captions
for txt in "${CAPTIONS[@]}"; do
    basename="${txt%.txt}"
    found=0
    for ext in $IMAGE_EXTENSIONS; do
        if [ -f "${basename}.$ext" ] || [ -f "${basename}.${ext^^}" ]; then
            found=1
            break
        fi
    done
    if [ $found -eq 0 ]; then
        UNMATCHED_CAPTIONS+=("$(basename "$txt")")
    fi
done

echo "Dataset Summary:"
echo "----------------------------------------"
echo -e "  Total images:     ${#IMAGES[@]}"
echo -e "  Total captions:   ${#CAPTIONS[@]}"
echo -e "  ${GREEN}Matched pairs:    ${#MATCHED[@]}${NC}"

if [ ${#UNMATCHED_IMAGES[@]} -gt 0 ]; then
    echo -e "  ${YELLOW}Images without captions: ${#UNMATCHED_IMAGES[@]}${NC}"
    for img in "${UNMATCHED_IMAGES[@]}"; do
        echo -e "    ${YELLOW}!${NC} $img"
    done
fi

if [ ${#UNMATCHED_CAPTIONS[@]} -gt 0 ]; then
    echo -e "  ${YELLOW}Captions without images: ${#UNMATCHED_CAPTIONS[@]}${NC}"
    for txt in "${UNMATCHED_CAPTIONS[@]}"; do
        echo -e "    ${YELLOW}!${NC} $txt"
    done
fi

echo "----------------------------------------"
echo ""

if [ ${#MATCHED[@]} -eq 0 ]; then
    echo -e "${RED}[ERROR]${NC} No matched image/caption pairs found!"
    echo ""
    echo "Expected format:"
    echo "  image_01.jpg + image_01.txt"
    echo "  image_02.png + image_02.txt"
    echo ""
    exit 1
fi

if [ ${#UNMATCHED_IMAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}[WARN]${NC} Some images don't have matching captions."
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
fi

# ============================================================================
# Step 3: Get remote connection details
# ============================================================================
echo -e "${BLUE}[3/4]${NC} Remote connection details"
echo ""

REMOTE_HOST="${2:-}"
REMOTE_PORT="${3:-}"

if [ -z "$REMOTE_HOST" ]; then
    read -p "Remote host (e.g., 38.147.83.28): " REMOTE_HOST
fi

if [ -z "$REMOTE_PORT" ]; then
    read -p "SSH port (e.g., 46243): " REMOTE_PORT
fi

# Test connection
echo ""
echo "Testing SSH connection..."
if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$SSH_KEY" -p "$REMOTE_PORT" "root@$REMOTE_HOST" "echo 'Connected'" &>/dev/null; then
    echo -e "${GREEN}[OK]${NC} SSH connection successful"
else
    echo -e "${RED}[ERROR]${NC} SSH connection failed"
    echo "Check: ssh -i $SSH_KEY -p $REMOTE_PORT root@$REMOTE_HOST"
    exit 1
fi

echo ""

# ============================================================================
# Step 4: Upload files
# ============================================================================
echo -e "${BLUE}[4/4]${NC} Uploading dataset..."
echo ""

# Clear existing data on remote (optional)
read -p "Clear existing data on remote before upload? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Clearing remote data directory..."
    ssh -i "$SSH_KEY" -p "$REMOTE_PORT" "root@$REMOTE_HOST" "rm -rf ${REMOTE_DATA_DIR}/* && mkdir -p ${REMOTE_DATA_DIR}"
fi

# Create remote directory
ssh -i "$SSH_KEY" -p "$REMOTE_PORT" "root@$REMOTE_HOST" "mkdir -p ${REMOTE_DATA_DIR}"

# Upload matched pairs only
echo "Uploading ${#MATCHED[@]} image/caption pairs..."
echo ""

uploaded=0
for img_name in "${MATCHED[@]}"; do
    basename="${img_name%.*}"
    img_path="$LOCAL_FOLDER/$img_name"
    txt_path="$LOCAL_FOLDER/${basename}.txt"

    # Upload image
    scp -i "$SSH_KEY" -P "$REMOTE_PORT" -q "$img_path" "root@$REMOTE_HOST:${REMOTE_DATA_DIR}/"

    # Upload caption
    scp -i "$SSH_KEY" -P "$REMOTE_PORT" -q "$txt_path" "root@$REMOTE_HOST:${REMOTE_DATA_DIR}/"

    uploaded=$((uploaded + 1))
    echo -e "  [${uploaded}/${#MATCHED[@]}] $img_name + ${basename}.txt"
done

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    UPLOAD COMPLETE                             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Uploaded: ${uploaded} image/caption pairs"
echo "Location: root@${REMOTE_HOST}:${REMOTE_DATA_DIR}/"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. SSH to pod: ssh -i $SSH_KEY -p $REMOTE_PORT root@$REMOTE_HOST"
echo "  2. Run analysis: cd /workspace/lora_training && python scripts/analyze_dataset.py"
echo "  3. Start training: bash scripts/tmux_train.sh fast"
echo ""

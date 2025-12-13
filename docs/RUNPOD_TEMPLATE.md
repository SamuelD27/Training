# RunPod Template Setup Guide

Instructions for deploying the Identity LoRA training environment on RunPod.

---

## Prerequisites

1. RunPod account with GPU credits
2. Docker Hub account (or other container registry)
3. Your dataset prepared locally

---

## Building & Pushing the Docker Image

### Option 1: Build Locally and Push

```bash
# Navigate to repo root
cd lora_training

# Build the image
docker build -t YOUR_DOCKERHUB_USER/lora-flux-trainer:latest -f docker/Dockerfile .

# Login to Docker Hub
docker login

# Push to registry
docker push YOUR_DOCKERHUB_USER/lora-flux-trainer:latest
```

### Option 2: Build on RunPod (slower but no local GPU needed)

1. Start a basic RunPod instance
2. Clone this repo
3. Build the image there
4. Push to registry

---

## Creating the RunPod Template

1. Go to **RunPod Console** → **Templates** → **New Template**

2. Fill in:
   - **Template Name**: `FLUX LoRA Trainer`
   - **Container Image**: `YOUR_DOCKERHUB_USER/lora-flux-trainer:latest`
   - **Docker Command**: (leave empty - uses ENTRYPOINT)
   - **Container Disk**: `50 GB` minimum (for models + outputs)
   - **Volume Disk**: `100 GB` (for caching models)
   - **Volume Mount Path**: `/workspace`

3. **Environment Variables** (optional, can override at launch):
   ```
   MODEL_PATH=/workspace/models/flux1-dev
   HF_TOKEN=your_huggingface_token
   ```

4. **Exposed Ports**:
   - `6006` (TensorBoard)
   - `22` (SSH)

5. Click **Save Template**

---

## Expected Mount Points

| Path | Purpose |
|------|---------|
| `/workspace` | Main volume (persistent across restarts) |
| `/workspace/lora_training` | Training repo (copied from image) |
| `/workspace/lora_training/data/subject` | **Your dataset (mount or copy here)** |
| `/workspace/lora_training/data/reg` | Regularization images (optional) |
| `/workspace/lora_training/output` | Training outputs |
| `/workspace/lora_training/logs` | Logs |
| `/workspace/models` | Base models (FLUX.1-dev) |

---

## Environment Variables

Override defaults by setting these when launching the pod:

### Paths
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/workspace/models/flux1-dev` | Path to FLUX.1-dev model |
| `DATA_DIR` | `/workspace/lora_training/data/subject` | Subject dataset path |
| `REG_DIR` | `/workspace/lora_training/data/reg` | Regularization images |
| `OUT_DIR` | `/workspace/lora_training/output` | Output directory |

### Training Parameters
| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_NAME` | `flux_lora` | Name for this training run |
| `SEED` | `42` | Random seed |
| `BATCH_SIZE` | `1` | Batch size |
| `GRAD_ACCUM` | `4` | Gradient accumulation steps |
| `MAX_STEPS` | (profile default) | Override max training steps |
| `RANK` | (profile default) | LoRA rank |
| `ALPHA` | (profile default) | LoRA alpha |
| `UNET_LR` | (profile default) | UNet learning rate |

### Text Encoder
| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_TE` | `0` | Enable text encoder training |
| `TE_LR` | `0` | Text encoder LR (or auto-calc) |
| `TWO_PHASE_TE` | `0` | Enable two-phase TE training |

### Debug
| Variable | Default | Description |
|----------|---------|-------------|
| `DRY_RUN` | `0` | Print command without executing |

---

## Launching a Pod

### From RunPod Console

1. Go to **Pods** → **New Pod**
2. Select **GPU Type**: RTX Pro 6000 / A6000 / H100 / B200
3. Select your template
4. Add environment variables if needed
5. Click **Deploy**

### From CLI (runpodctl)

```bash
# Install runpodctl
pip install runpodctl

# Configure
runpodctl config --apiKey YOUR_API_KEY

# Launch pod
runpodctl pod create \
  --name "flux-lora-training" \
  --gpu "NVIDIA RTX 6000 Ada" \
  --template "YOUR_TEMPLATE_ID" \
  --volume 100 \
  --env "RUN_NAME=my_subject" \
  --env "MODEL_PATH=/workspace/models/flux1-dev"
```

---

## First-Time Setup After Launch

### 1. SSH into the Pod

```bash
ssh root@YOUR_POD_IP -p YOUR_SSH_PORT
# or use RunPod web terminal
```

### 2. Download FLUX.1-dev Model (if not cached)

```bash
# Using huggingface-cli
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir /workspace/models/flux1-dev \
  --token $HF_TOKEN

# Or using git lfs
git lfs install
git clone https://huggingface.co/black-forest-labs/FLUX.1-dev /workspace/models/flux1-dev
```

### 3. Upload Your Dataset

```bash
# From your local machine
rsync -avz --progress \
  /path/to/your/dataset/ \
  root@YOUR_POD_IP:/workspace/lora_training/data/subject/ \
  -e "ssh -p YOUR_SSH_PORT"
```

Or use RunPod's file upload feature.

### 4. Verify Setup

```bash
cd /workspace/lora_training

# Analyze dataset
python scripts/analyze_dataset.py

# Dry run to see the command
DRY_RUN=1 bash scripts/train_flux_fast.sh
```

---

## One-Line Start Commands

### Fast Iteration (Sanity Check)
```bash
cd /workspace/lora_training && bash scripts/train_flux_fast.sh
```

### Final/Production Training
```bash
cd /workspace/lora_training && bash scripts/train_flux_final.sh
```

### With Telemetry (tmux)
```bash
cd /workspace/lora_training && bash scripts/tmux_train.sh fast
# or
cd /workspace/lora_training && bash scripts/tmux_train.sh final
```

### Custom Run Name
```bash
RUN_NAME=john_doe bash scripts/train_flux_final.sh
```

---

## Accessing Outputs

### During Training
- Logs: `tail -f /workspace/lora_training/logs/RUN_NAME.log`
- Samples: `ls /workspace/lora_training/output/samples/RUN_NAME/`
- TensorBoard: Connect to port 6006

### After Training
```bash
# Download LoRA
rsync -avz \
  root@YOUR_POD_IP:/workspace/lora_training/output/*.safetensors \
  /local/path/ \
  -e "ssh -p YOUR_SSH_PORT"
```

---

## Cost Optimization Tips

1. **Use Fast profile first** to validate dataset (~1 hour)
2. **Pause pod** when not training to save credits
3. **Use volume storage** for models (persist across sessions)
4. **Download outputs** before terminating pod

---

## Troubleshooting

### Pod won't start
- Check container image exists and is public (or you have auth)
- Verify enough disk space requested

### GPU not detected
- Try restarting the pod
- Check NVIDIA driver compatibility

### Out of VRAM
- Reduce `BATCH_SIZE` to 1
- Enable gradient checkpointing (default)
- Use FP8 base model (default)
- Use block swapping if available

### Training NaNs
- Lower learning rate
- Ensure bf16 mixed precision
- Check dataset for corrupted images

See `docs/TRAINING_GUIDE.md` for more troubleshooting.

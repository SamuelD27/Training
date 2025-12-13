# Identity LoRA Training (FLUX.1-dev)

Production-ready pipeline for training identity-preserving LoRAs with FLUX.1-dev.

> Based on **DeepResearchReport.md** (State of the Art – December 2025)

---

## Quick Start

### 1. Build Docker Image

```bash
docker build -t lora-flux-trainer:latest -f docker/Dockerfile .
docker push YOUR_REGISTRY/lora-flux-trainer:latest
```

### 2. Launch on RunPod

See [docs/RUNPOD_TEMPLATE.md](docs/RUNPOD_TEMPLATE.md) for detailed setup.

### 3. SSH and Prepare

```bash
# Download FLUX.1-dev model
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir /workspace/models/flux1-dev

# Upload your dataset
# (from local machine)
rsync -avz /path/to/your/dataset/ root@POD_IP:/workspace/lora_training/data/subject/
```

### 4. Analyze Dataset

```bash
cd /workspace/lora_training
python scripts/analyze_dataset.py
```

### 5. Train

**Fast Iteration (validate dataset)**:
```bash
bash scripts/train_flux_fast.sh
```

**Production Quality**:
```bash
bash scripts/train_flux_final.sh
```

**With Telemetry (tmux)**:
```bash
bash scripts/tmux_train.sh final
```

---

## Training Commands

| Command | Purpose | Runtime |
|---------|---------|---------|
| `bash scripts/train_flux_fast.sh` | Quick validation | ~1 hour |
| `bash scripts/train_flux_final.sh` | Production LoRA | ~3-4 hours |
| `bash scripts/tmux_train.sh fast` | Fast + telemetry | ~1 hour |
| `bash scripts/tmux_train.sh final` | Final + telemetry | ~3-4 hours |

### Dry Run (Preview Command)

```bash
DRY_RUN=1 bash scripts/train_flux_fast.sh
DRY_RUN=1 bash scripts/train_flux_final.sh
```

### Custom Parameters

```bash
RUN_NAME=john_doe \
MAX_STEPS=3000 \
RANK=48 \
bash scripts/train_flux_final.sh
```

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for all options.

---

## Output Locations

| Content | Location |
|---------|----------|
| LoRA files | `output/*.safetensors` |
| Training samples | `output/samples/<run_name>/` |
| Logs | `logs/<run_name>.log` |
| Dataset analysis | `logs/dataset_report.json` |
| pip freeze | `logs/pip_freeze.txt` |

---

## Dataset Requirements

**Structure**:
```
data/subject/
├── images/
│   ├── photo1.jpg
│   └── photo2.png
└── captions/
    ├── photo1.txt
    └── photo2.txt
```

**Requirements**:
- 15-30 images (minimum 15)
- Mixed angles (front, ¾, profile)
- Varied lighting and environments
- Captions with trigger token: `<token>, description`

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for details.

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| **NaNs** | LR too high | `UNET_LR=5e-5` |
| **Weak identity** | Undertraining | More steps or higher rank |
| **Sameface** | Overtraining | Early stop, use regularization |
| **Waxy skin** | High noise offset | `NOISE_OFFSET=0.05` |
| **Angle failure** | Poor dataset | Add diverse angles |
| **OOM** | Insufficient VRAM | Reduce batch size or resolution |

Full troubleshooting: [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md#troubleshooting)

---

## Documentation

| Document | Content |
|----------|---------|
| [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | Complete training guide |
| [EVAL_PROTOCOL.md](docs/EVAL_PROTOCOL.md) | How to evaluate LoRAs |
| [REPORT_SUMMARY.md](docs/REPORT_SUMMARY.md) | Report defaults summary |
| [ENV_PINNING.md](docs/ENV_PINNING.md) | Version pinning |
| [RUNPOD_TEMPLATE.md](docs/RUNPOD_TEMPLATE.md) | RunPod setup |
| [patches.md](docs/patches.md) | Code patches (if any) |

---

## Repository Structure

```
lora_training/
├── README.md                 # This file
├── DeepResearchReport.md     # Source of truth
├── docker/
│   ├── Dockerfile            # Container definition
│   ├── start.sh              # Container entrypoint
│   └── env.sh                # Environment defaults
├── scripts/
│   ├── analyze_dataset.py    # Dataset analysis
│   ├── train_flux_fast.sh    # Fast iteration training
│   ├── train_flux_final.sh   # Production training
│   ├── telemetry.sh          # GPU monitoring
│   ├── tmux_train.sh         # tmux orchestrator
│   └── make_regularization_set.md
├── configs/
│   ├── flux_fast.toml        # Fast profile config
│   ├── flux_final.toml       # Final profile config
│   └── sample_prompts.txt    # Evaluation prompts
├── data/
│   ├── subject/              # Your dataset (mount here)
│   └── reg/                  # Regularization images
├── output/                   # Training outputs
├── logs/                     # Training logs
├── docs/                     # Documentation
└── third_party/
    └── sd-scripts/           # kohya-ss sd-scripts (pinned)
```

---

## Environment Variables

### Essential
```bash
MODEL_PATH      # Path to FLUX.1-dev model
DATA_DIR        # Path to subject dataset
RUN_NAME        # Name for this training run
```

### Training
```bash
MAX_STEPS       # Override max training steps
RANK            # LoRA rank (32 fast, 64 final)
ALPHA           # LoRA alpha (usually = rank)
UNET_LR         # UNet learning rate
```

### Features
```bash
ENABLE_TE=1     # Enable text encoder training
USE_REG=1       # Enable regularization
DRY_RUN=1       # Print command without running
```

---

## Hardware Requirements

| GPU | VRAM | Supported | Notes |
|-----|------|-----------|-------|
| RTX 3090 | 24GB | Yes | Minimum |
| RTX 4090 | 24GB | Yes | Good |
| RTX Pro 6000 | 48GB | Yes | Recommended |
| A6000 | 48GB | Yes | Recommended |
| H100 | 80GB | Yes | Best |
| B200 | 80GB | Yes | Best |

---

## Acknowledgments

- **kohya-ss/sd-scripts** - Training engine
- **Black Forest Labs** - FLUX.1-dev model
- **DeepResearchReport.md** - Training methodology

# FLUX.1-dev Identity LoRA Training Pipeline

## Project Overview

Production-ready FLUX.1-dev LoRA training pipeline using kohya-ss sd-scripts (sd3 branch). Uses `flux_train_network.py` for native FLUX.1 support with identity preservation training.

## Quick Reference

### Entry Points
| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/train_dashboard.sh` | Live dashboard training | `PROFILE=fast bash scripts/train_dashboard.sh` |
| `scripts/train_flux_fast.sh` | Fast iteration (<1hr) | `bash scripts/train_flux_fast.sh` |
| `scripts/train_flux_final.sh` | Production training | `bash scripts/train_flux_final.sh` |
| `scripts/train_remote.sh` | Local Mac orchestrator | `bash scripts/train_remote.sh` |
| `scripts/upload_dataset.sh` | Upload dataset to pod | `bash scripts/upload_dataset.sh` |
| `scripts/watch_samples.sh` | Watch sample images | `bash scripts/watch_samples.sh` |
| `scripts/analyze_dataset.py` | Dataset validation | `python scripts/analyze_dataset.py` |
| `scripts/sync_models_r2.sh` | Sync models from R2 | `bash scripts/sync_models_r2.sh` |

### Environment Variables
```bash
# Paths
MODEL_PATH=/workspace/models/flux1-dev       # FLUX.1-dev model location
TEXT_ENCODER_PATH=/workspace/models/text_encoders  # CLIP-L and T5-XXL
DATA_DIR=/workspace/lora_training/data/subject     # Training dataset

# Training config
RUN_NAME=my_lora              # Output name
PROFILE=fast                  # fast or final
MAX_STEPS=1500               # Override max steps
RESOLUTION=512               # Override resolution (auto-adjusts bucket sizes)
RANK=32                      # LoRA rank (ALPHA auto-syncs to match)

# Sample generation
SAMPLE_EVERY_N_STEPS=250     # Generate samples every N steps
SAMPLE_PROMPTS=/workspace/lora_training/configs/sample_prompts.txt

# Checkpoints
SAVE_EVERY_N_STEPS=500       # Save checkpoint every N steps

# Debug
DRY_RUN=1                    # Preview command without executing
```

### Auto-Adjustments (build_train_cmd.py)

The training script automatically handles:
- **ALPHA = RANK**: When you set `RANK=64`, alpha auto-syncs to 64
- **Bucket sizes**: When `RESOLUTION > max_bucket_reso`, buckets auto-adjust
- **Empty values**: Empty env vars are ignored (use profile defaults)

### Training Profiles
| Profile | Resolution | Steps | Rank | LR Scheduler | Runtime |
|---------|------------|-------|------|--------------|---------|
| fast | 512x512 | 1500 | 32 | constant_with_warmup | ~1 hour |
| final | 768x768 | 2500 | 64 | cosine_with_restarts | ~3-4 hours |

## Dashboard Features

The `train_dashboard.sh` provides a live training monitor with:
- ASCII progress bar with percentage
- Live step/loss tracking
- ETA calculation
- GPU telemetry (utilization, VRAM, temp, power)
- Status detection (Initializing, Caching, Training, Generating samples, Saving)
- Sample image display (requires `viu`)

```bash
# Quick test with samples
MAX_STEPS=30 SAMPLE_EVERY_N_STEPS=10 bash scripts/train_dashboard.sh
```

## Directory Structure
```
lora_training/
├── scripts/              # Training and utility scripts
├── configs/              # Sample prompts (TOK Woman trigger)
├── docker/               # Docker env.sh and start.sh
├── data/
│   └── subject/          # Your identity dataset
│       └── 10_TOK/       # 10 repeats, TOK trigger token
│           ├── image1.jpg
│           ├── image1.txt
│           └── ...
├── output/               # Trained LoRA models
│   └── sample/           # Generated sample images
└── logs/                 # Training logs
```

## Dataset Requirements
- Minimum: 15-30 high-quality images
- Each image needs matching `.txt` caption file
- Directory format: `data/subject/10_TOK/` (10 repeats, TOK trigger token)
- Caption format: `TOK Woman, description of the image`
- Mixed angles, lighting, and environments

## Sample Prompts

Located at `configs/sample_prompts.txt`. Uses `TOK Woman` trigger. Lines starting with `#` are comments.

Default prompts test:
1. Clean portrait (baseline)
2. Dramatic lighting
3. Outdoor environment
4. Cinematic style
5. Expressive pose

## Key Configuration (docker/env.sh)
- `SDSCRIPTS=/opt/sd-scripts` - sd-scripts location (sd3 branch)
- `TRAIN_SCRIPT=flux_train_network.py` - FLUX training script
- `TEXT_ENCODER_PATH` - CLIP-L and T5-XXL from ComfyUI
- `FAST_*` / `FINAL_*` - Profile-specific parameters

## sd-scripts Branch

Uses **sd3 branch** which includes:
- `flux_train_network.py` - Native FLUX.1 training
- `networks/lora_flux.py` - FLUX-specific LoRA module
- FLUX-specific settings: `--guidance_scale=1.0`, `--timestep_sampling=flux_shift`

## Required Models

1. **FLUX.1-dev**: `flux1-dev.safetensors`, `ae.safetensors`
2. **Text Encoders** (from ComfyUI):
   - `clip_l.safetensors` (235MB)
   - `t5xxl_fp16.safetensors` (9.2GB)

Text encoders auto-download on container start from:
`https://huggingface.co/comfyanonymous/flux_text_encoders`

## FLUX Training Parameters

Key parameters used (from kohya-ss docs):
```bash
--network_module=networks.lora_flux    # FLUX-specific LoRA
--guidance_scale=1.0                   # FLUX guidance
--timestep_sampling=flux_shift         # FLUX timestep sampling
--model_prediction_type=raw            # FLUX prediction type
--discrete_flow_shift=1.0              # Flow matching shift
--blocks_to_swap=18                    # Memory optimization
--cache_text_encoder_outputs           # Cache TE outputs
--cache_latents_to_disk                # Cache latents to disk
```

## Common Issues

| Issue | Solution |
|-------|----------|
| NaN during training | Lower learning rate (try 5e-5) |
| OOM errors | Increase `--blocks_to_swap` (default 18) |
| Weak identity | More steps or higher rank |
| Captions not found | Ensure `--caption_extension=.txt` |
| `shuffle_caption` error | Don't use with `cache_text_encoder_outputs` |
| `constant` scheduler error | Use `constant_with_warmup` instead |

## Versions (CUDA 12.8 / Blackwell Support)
- PyTorch: 2.9.x (CUDA 12.8) - supports Blackwell sm_100/sm_120
- transformers: latest
- diffusers: latest
- xformers: latest (cu128)
- bitsandbytes: latest
- sd-scripts: sd3 branch (FLUX support)

## Docker Build & Run

```bash
# Build
docker build -t lora-flux-trainer:latest -f Dockerfile .

# Run (interactive)
docker run --gpus all -it \
  -v /path/to/dataset:/workspace/lora_training/data/subject \
  -v /path/to/models:/workspace/models \
  lora-flux-trainer:latest
```

## RunPod Deployment

1. Create pod with GPU (A6000 48GB recommended)
2. Use custom Docker image or RunPod PyTorch template
3. SSH into pod and clone/sync repo
4. Run `bash scripts/sync_models_r2.sh` to get models
5. Upload dataset to `data/subject/10_TOK/`
6. Run training with dashboard

## Remote Workflow (from Mac)

```bash
# 1. Upload dataset
bash scripts/upload_dataset.sh

# 2. Start training with dashboard (SSH into pod)
ssh -p PORT root@POD_IP
cd /workspace/lora_training
PROFILE=fast bash scripts/train_dashboard.sh

# 3. Watch samples locally
bash scripts/watch_samples.sh
```

## Debugging
- Check logs: `logs/<run_name>.log`
- Full log: `logs/<run_name>_full.log`
- GPU monitoring: `nvitop` or dashboard
- Dry run: `DRY_RUN=1 bash scripts/train_flux_fast.sh`
- Sample output: `output/sample/<run_name>_*.png`

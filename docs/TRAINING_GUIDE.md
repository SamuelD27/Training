# Training Guide

Complete guide for training Identity LoRAs with FLUX.1-dev.

> Source: DeepResearchReport.md

---

## Table of Contents

1. [Configuration (Single Source of Truth)](#configuration-single-source-of-truth)
2. [Dataset Contract](#dataset-contract)
3. [Training Profiles](#training-profiles)
4. [Environment Variables](#environment-variables)
5. [Step-by-Step Training](#step-by-step-training)
6. [Resume Training](#resume-training)
7. [Troubleshooting](#troubleshooting)

---

## Configuration (Single Source of Truth)

**Important**: All training configuration is now centralized in TOML files:

- `configs/flux_fast.toml` - Fast iteration profile
- `configs/flux_final.toml` - Production profile

These files are loaded by `scripts/build_train_cmd.py` which generates the exact command used by all training scripts.

### How It Works

1. TOML files define profile-specific hyperparameters
2. `build_train_cmd.py` reads TOML and applies env var overrides
3. Shell scripts (`train_flux_fast.sh`, `train_flux_final.sh`, `train_dashboard.sh`) call `build_train_cmd.py`
4. All scripts execute the exact same command for consistency

### Validated Parameters (P0/P1 Fixes)

The following critical parameters are now properly wired:

| Parameter | Fast Profile | Final Profile | Status |
|-----------|--------------|---------------|--------|
| `network_alpha` | 32 (= rank) | 64 (= rank) | P0 Fixed |
| `noise_offset` | 0.05 | 0.1 | P0 Fixed |
| `network_dropout` | 0.0 | 0.1 | P0 Fixed |
| `min_snr_gamma` | 0 (disabled) | 5.0 | P0 Fixed |
| `gradient_accumulation_steps` | 4 | 4 | P1 Fixed |

---

## Dataset Contract

### Required Structure

Your dataset must be mounted to `/workspace/lora_training/data/subject/` with one of these structures:

**Option A: Separate directories**
```
data/subject/
├── images/
│   ├── photo1.jpg
│   ├── photo2.png
│   └── photo3.jpg
└── captions/
    ├── photo1.txt
    ├── photo2.txt
    └── photo3.txt
```

**Option B: Files alongside each other**
```
data/subject/
├── photo1.jpg
├── photo1.txt
├── photo2.png
├── photo2.txt
├── photo3.jpg
└── photo3.txt
```

### Image Requirements

| Requirement | Value | Source |
|-------------|-------|--------|
| Minimum count | 15 images | Section 6 |
| Recommended count | 15-30 images | Section 6 |
| Format | JPG, PNG, WebP | - |
| Angles | Mixed (front, 3/4, profile) | Section 6 |
| Lighting | Varied | Section 6 |
| Framing | Portrait + mid/full body | Section 6 |

### Caption Requirements

Each image must have a corresponding `.txt` caption file with the same basename.

**Format**:
```
<trigger_token>, description of the image
```

**Examples**:
```
ohwx, a man standing outdoors, sunny day
ohwx wearing a blue shirt, indoor setting
ohwx, portrait photo, neutral background
```

**Rules**:
1. Use a unique trigger token (e.g., `ohwx`, `sks`, or a made-up word)
2. Trigger token appears in EVERY caption
3. Keep captions concise (avoid verbosity)
4. Vary the descriptions (don't repeat identical captions)
5. Don't describe permanent traits inconsistently

---

## Training Profiles

### Fast Iteration Profile

**Purpose**: Validate dataset, detect early issues

| Setting | Value |
|---------|-------|
| Resolution | 512x512 |
| Steps | 1500 |
| Rank/Alpha | 32/32 |
| LR | 1e-4 |
| Scheduler | constant_with_warmup |
| Warmup | 100 |
| Noise Offset | 0.05 |
| Bucket Range | 256-768 |

**Command**:
```bash
bash scripts/train_flux_fast.sh
```

### Final/Production Profile

**Purpose**: Maximum quality, production use

| Setting | Value |
|---------|-------|
| Resolution | 768x768 |
| Steps | 2500 |
| Rank/Alpha | 64/64 |
| Dropout | 0.1 |
| LR | 1e-4 |
| Scheduler | cosine_with_restarts |
| Warmup | 500 |
| Noise Offset | 0.1 |
| SNR Gamma | 5.0 |
| Bucket Range | 384-1024 |

**Command**:
```bash
bash scripts/train_flux_final.sh
```

### Dashboard Mode

Use the dashboard for live progress visualization:

```bash
PROFILE=fast bash scripts/train_dashboard.sh
PROFILE=final bash scripts/train_dashboard.sh
```

---

## Environment Variables

Override any default by setting environment variables.

### Paths
```bash
MODEL_PATH=/path/to/flux1-dev
DATA_DIR=/path/to/dataset
OUT_DIR=/path/to/output
```

### Training Parameters
```bash
RUN_NAME=my_subject_lora    # Name for this run
SEED=42                     # Random seed
MAX_STEPS=2500              # Override max steps
```

### Network
```bash
RANK=64                     # LoRA rank (alpha will match)
ALPHA=64                    # Explicit alpha (optional)
LEARNING_RATE=1e-4          # Learning rate
NOISE_OFFSET=0.1            # Noise offset
DROPOUT=0.1                 # Network dropout
SNR_GAMMA=5.0               # Min SNR gamma
GRAD_ACCUM=4                # Gradient accumulation steps
```

### Features
```bash
FP8_BASE=1                  # Enable FP8 base model quantization
RESUME_FROM=/path/to/ckpt   # Resume from checkpoint
DRY_RUN=1                   # Print command without executing
```

### Example: Custom Training
```bash
RUN_NAME=john_doe \
RANK=48 \
MAX_STEPS=2000 \
NOISE_OFFSET=0.08 \
bash scripts/train_flux_final.sh
```

---

## Step-by-Step Training

### 1. Prepare Dataset

1. Collect 15-30 high-quality images
2. Create captions with trigger token
3. Upload to pod:
   ```bash
   bash scripts/upload_dataset.sh
   # or manually:
   rsync -avz /local/dataset/ root@POD_IP:/workspace/lora_training/data/subject/
   ```

### 2. Analyze Dataset

```bash
cd /workspace/lora_training
python scripts/analyze_dataset.py
```

Review recommendations and warnings.

### 3. Dry Run (Preview Command)

```bash
DRY_RUN=1 bash scripts/train_flux_fast.sh
```

This shows the exact command that will be executed.

### 4. Run Fast Iteration

```bash
bash scripts/train_flux_fast.sh
```

Check samples in `output/samples/` - if identity is recognizable, proceed to final.

### 5. Run Final Training

```bash
bash scripts/train_flux_final.sh
```

Or with live dashboard:
```bash
PROFILE=final bash scripts/train_dashboard.sh
```

### 6. Evaluate Results

1. Check samples in `output/samples/`
2. Test at different LoRA strengths (0.5, 0.8, 1.0, 1.2)
3. Run evaluation prompts from `configs/sample_prompts.txt`
4. See `docs/EVAL_PROTOCOL.md` for full evaluation

---

## Resume Training

Training can be resumed from a checkpoint using the `RESUME_FROM` environment variable:

```bash
# Resume from a specific checkpoint
RESUME_FROM=/workspace/lora_training/output/flux_final_20231215_checkpoint.safetensors \
bash scripts/train_flux_final.sh

# Resume with additional overrides
RESUME_FROM=/path/to/checkpoint.safetensors \
MAX_STEPS=3000 \
bash scripts/train_flux_final.sh
```

The resume feature:
- Loads network weights from the specified checkpoint
- Continues training with the same or modified parameters
- Logs resume status in training output

---

## Troubleshooting

> Source: DeepResearchReport.md Section 9

### NaNs During Training

**Cause**: Learning rate too high or fp16 overflow

**Fix**:
```bash
LEARNING_RATE=5e-5 bash scripts/train_flux_final.sh
```
Ensure `bf16` mixed precision (default).

### Weak Identity

**Cause**: Undertraining

**Fix**:
- Increase steps: `MAX_STEPS=3500`
- Increase rank: `RANK=96`
- Add more diverse training images

### Sameface / Carbon Copy

**Cause**: Overtraining

**Fix**:
- Early stop (reduce steps)
- Enable dropout (default in final profile)
- Use regularization: `USE_REG=1`
- Check for duplicate images in dataset

### Waxy / Plastic Skin

**Cause**: Over-smoothing from high noise offset

**Fix**:
```bash
NOISE_OFFSET=0.05 bash scripts/train_flux_final.sh
```

### Angle Failure (Only Works for Certain Angles)

**Cause**: Poor dataset diversity

**Fix**:
- Add more images from different angles
- Ensure dataset has front, 3/4, and profile views
- Run `analyze_dataset.py` to check aspect ratio diversity

### Prompt Brittleness (Only Works with Specific Prompts)

**Cause**: Text encoder overfit

**Fix**:
- Freeze TE (default)
- If using TE, freeze early with two-phase training

### Out of Memory (OOM)

**Fix**:
1. Reduce batch size: `BATCH_SIZE=1`
2. Reduce resolution: `RESOLUTION=512`
3. FP8 base model: `FP8_BASE=1`
4. Gradient checkpointing (enabled by default)

---

## Reproducibility Artifacts

Every training run automatically saves:

| File | Content |
|------|---------|
| `logs/<run>_command.txt` | Exact command executed |
| `logs/<run>_repro.json` | Git hash, GPU info, hyperparameters |
| `logs/<run>_pip_freeze.txt` | Python package versions |
| `logs/<run>_dataset_hash.txt` | Dataset fingerprint |

These files enable exact reproduction of any training run.

---

## Validation

To validate the training pipeline is correctly configured:

```bash
python scripts/validate_config_usage.py
```

This checks:
- Scripts reference `build_train_cmd.py`
- TOML configs are valid
- P0 fixes are in place (alpha=rank, noise_offset, etc.)
- FLUX-required parameters are present

---

## Canonical Rule

> **If identity realism degrades, fix the dataset first - not the optimizer.**

Before adjusting training parameters, ensure your dataset:
- Has enough images (15-30)
- Has good angle diversity
- Has consistent trigger token usage
- Has no duplicates or low-quality images

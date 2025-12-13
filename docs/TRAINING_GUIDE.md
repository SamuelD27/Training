# Training Guide

Complete guide for training Identity LoRAs with FLUX.1-dev.

> Source: DeepResearchReport.md

---

## Table of Contents

1. [Dataset Contract](#dataset-contract)
2. [Training Profiles](#training-profiles)
3. [Text Encoder Policy](#text-encoder-policy)
4. [Environment Variables](#environment-variables)
5. [Step-by-Step Training](#step-by-step-training)
6. [Troubleshooting](#troubleshooting)

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
| Angles | Mixed (front, ¾, profile) | Section 6 |
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
| Text Encoder | Frozen |

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
| Scheduler | cosine |
| Warmup | 500 |
| Noise Offset | 0.1 |
| SNR Gamma | 5.0 |
| Bucket Range | 384-1024 |
| Text Encoder | Frozen (default) |

**Command**:
```bash
bash scripts/train_flux_final.sh
```

---

## Text Encoder Policy

> Source: DeepResearchReport.md Section 5

### Default: Frozen (Recommended)

Text encoder is frozen by default. This is correct for most cases.

**Keep TE frozen when**:
- Using a unique trigger token
- Dataset < 15 images
- Identity already appears clearly
- Prompt flexibility matters

### When to Enable TE Training

**Enable TE when**:
- Identity is weak or inconsistent
- Trigger token has semantic ambiguity
- Need stronger name→face binding

### How to Enable

```bash
ENABLE_TE=1 bash scripts/train_flux_final.sh
```

### TE Training Rules

When TE is enabled:
- TE learning rate = 0.4 × UNet LR (automatic)
- Use regularization images
- Consider two-phase training

### Two-Phase Training

For best results with TE, use two phases:

**Phase 1**: Train with TE for first 50% of steps
```bash
ENABLE_TE=1 MAX_STEPS=1250 RUN_NAME=subject_phase1 bash scripts/train_flux_final.sh
```

**Phase 2**: Resume with TE frozen for remaining steps
```bash
ENABLE_TE=0 MAX_STEPS=1250 RESUME_FROM=output/subject_phase1.safetensors bash scripts/train_flux_final.sh
```

---

## Environment Variables

Override any default by setting environment variables:

### Paths
```bash
MODEL_PATH=/path/to/flux1-dev
DATA_DIR=/path/to/dataset
REG_DIR=/path/to/regularization
OUT_DIR=/path/to/output
```

### Training Parameters
```bash
RUN_NAME=my_subject_lora
SEED=42
MAX_STEPS=2500
BATCH_SIZE=1
GRAD_ACCUM=4
```

### Network
```bash
RANK=64
ALPHA=64
UNET_LR=1e-4
```

### Resolution
```bash
RESOLUTION=768
MIN_BUCKET_RESO=384
MAX_BUCKET_RESO=1024
```

### Features
```bash
ENABLE_TE=1           # Enable text encoder training
USE_REG=1             # Enable regularization
TWO_PHASE_TE=1        # Two-phase TE training
DRY_RUN=1             # Print command without executing
```

### Example: Custom Training
```bash
RUN_NAME=john_doe \
RANK=48 \
MAX_STEPS=2000 \
ENABLE_TE=1 \
USE_REG=1 \
bash scripts/train_flux_final.sh
```

---

## Step-by-Step Training

### 1. Prepare Dataset

1. Collect 15-30 high-quality images
2. Create captions with trigger token
3. Upload to pod:
   ```bash
   rsync -avz /local/dataset/ root@POD_IP:/workspace/lora_training/data/subject/
   ```

### 2. Analyze Dataset

```bash
cd /workspace/lora_training
python scripts/analyze_dataset.py
```

Review recommendations and warnings.

### 3. Run Fast Iteration

```bash
bash scripts/train_flux_fast.sh
```

Check samples in `output/samples/` - if identity is recognizable, proceed to final.

### 4. Run Final Training

```bash
bash scripts/train_flux_final.sh
```

Or with telemetry:
```bash
bash scripts/tmux_train.sh final
```

### 5. Evaluate Results

1. Check samples in `output/samples/`
2. Test at different LoRA strengths (0.5, 0.8, 1.0, 1.2)
3. Run evaluation prompts from `configs/sample_prompts.txt`
4. See `docs/EVAL_PROTOCOL.md` for full evaluation

---

## Troubleshooting

> Source: DeepResearchReport.md Section 9

### NaNs During Training

**Cause**: Learning rate too high or fp16 overflow

**Fix**:
```bash
UNET_LR=5e-5 bash scripts/train_flux_final.sh
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
- Ensure dataset has front, ¾, and profile views
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
3. FP8 base model (enabled by default)
4. Gradient checkpointing (enabled by default)

---

## Canonical Rule

> **If identity realism degrades, fix the dataset first — not the optimizer.**

Before adjusting training parameters, ensure your dataset:
- Has enough images (15-30)
- Has good angle diversity
- Has consistent trigger token usage
- Has no duplicates or low-quality images

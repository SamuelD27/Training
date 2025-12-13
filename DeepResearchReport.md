# Identity LoRA Training Playbook (FLUX.1-dev)
**State of the Art – December 2025**

> Scope: Identity LoRA training only (max realism + identity consistency)
> Primary target: **FLUX.1-dev**
> Secondary target: **HunyuanImage / Hunyuan DiT** (only if production-ready)
> Audience: Autonomous training agents (Claude Code), advanced practitioners
> Hardware assumption: 24GB minimum, optimized for 48–80GB GPUs (A6000 / H100 / B200)

---

## 1. Executive Recommendation

### ✅ Best Pipeline — FLUX.1-dev Identity LoRA
**kohya-ss `sd-scripts` (CLI-first)**

**Why**
- Highest training quality ceiling for identity realism
- Most mature FLUX support (FP8, block swap, SNR loss, bucketing)
- Fully scriptable, deterministic, automation-friendly
- Supports prior regularization (critical for faces)

**Tradeoffs**
- Requires precise configuration
- Steeper learning curve than GUI tools

---

### ⚠️ HunyuanImage / Hunyuan DiT Identity LoRA
**Status: Experimental / Emerging**

- LoRA training *is supported* via:
  - `sd-scripts` (UNet-only)
  - `diffusion-pipe`
- Documentation + community validation are still limited

**Recommendation**
> Use **FLUX.1-dev** for production identity LoRAs
> Experiment with Hunyuan only if high-res (>2K) output is mission-critical

---

## 2. Pipeline Decision Matrix (Identity LoRA Only)

| Pipeline | Engine or Wrapper | Quality Ceiling | Stability | VRAM Efficiency | Automation | FLUX Support | Hunyuan Support |
|--------|------------------|-----------------|-----------|-----------------|------------|--------------|-----------------|
| **sd-scripts (kohya)** | Engine | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Mature** | Emerging |
| Kohya GUI (bmaltais) | Wrapper | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Mature | Partial |
| Ostris AI-Toolkit | Wrapper | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Mature | Partial |
| diffusion-pipe | Engine | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Good | **Best** |
| OneTrainer | Wrapper | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Good | No |
| ComfyUI Flux Trainer | Wrapper | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Flux only | No |

**Key Distinction**
- **Engines**: `sd-scripts`, `diffusion-pipe`
- **Wrappers / UX layers**: AI-Toolkit, Kohya GUI, OneTrainer, ComfyUI

---

## 3. kohya-ss / sd-scripts — How It Works (Practically)

### Entry Points
- `train_network.py` → LoRA training
- LoRA injected into:
  - UNet attention & cross-attention layers
  - (Optional) text encoder layers

### Dataset Flow
1. Read images + `.txt` captions
2. Aspect-ratio bucketing (64px steps)
3. Optional:
   - Latent caching (VAE)
   - Text embedding caching (CLIP / T5)
4. Batched training in latent space

### Precision & Memory
- `bf16` preferred
- FP8 base weights (`fp8_base=true`)
- Gradient checkpointing
- Optional block swapping (CPU offload)

### Common Flux Gotchas
- LR too high → washed outputs / NaNs
- Missing `model_prediction_type="raw"`
- No bucketing → identity distortion
- Overtraining beyond saturation

---

## 4. SOTA Identity LoRA Recipes (FLUX.1-dev)

### 4.1 Fast Iteration Profile (Sanity Check)

**Purpose**
- Validate dataset
- Detect early overfit
- < 1 hour runtime

**Core Settings**
- Resolution: `512x512`
- Steps: `1500`
- Rank / Alpha: `32 / 32`
- Optimizer: `Adafactor`
- LR: `1e-4`
- Text Encoder: ❌ frozen
- Noise offset: `0.05`

#### CLI Script
```bash
accelerate launch train_network.py \
  --pretrained_model_name_or_path=/models/flux1-dev \
  --network_module=networks.lora \
  --network_dim=32 --network_alpha=32 \
  --unet_lr=1e-4 --text_encoder_lr=0 \
  --optimizer_type=Adafactor \
  --optimizer_args="relative_step=False scale_parameter=False" \
  --lr_scheduler=constant_with_warmup \
  --lr_warmup_steps=100 \
  --resolution=512,512 \
  --enable_bucket --min_bucket_reso=256 --max_bucket_reso=768 \
  --mixed_precision=bf16 --fp8_base \
  --gradient_checkpointing \
  --train_data_dir=data/subject \
  --caption_extension=.txt \
  --max_train_epochs=100 \
  --save_every_n_epochs=10 \
  --output_dir=output --output_name=subject_fast \
  --seed=42
```

---

### 4.2 Final High-Fidelity Profile (Production)

**Purpose**
- Maximum realism
- Strong identity lock
- Robust generalization

**Core Settings**
- Resolution: 768 base, buckets up to 1024
- Steps: 2000–3000
- Rank / Alpha: 64 / 64
- Optimizer: Adafactor
- LR: 1e-4 → cosine decay
- Dropout: 0.1
- Prior images: ✅ recommended
- Noise offset: 0.1
- Loss: SNR / Huber

#### High-Fidelity Config (Excerpt)
```yaml
network:
  dim: 64
  alpha: 64
  dropout: 0.1

optimizer:
  type: Adafactor
  lr: 1e-4
  scheduler: cosine
  warmup_steps: 500

training:
  resolution: 768
  min_bucket_reso: 384
  max_bucket_reso: 1024
  gradient_checkpointing: true
  noise_offset: 0.1
  min_snr_gamma: 5.0
```

---

## 5. Text Encoder Training Policy

**❌ Do NOT train TE when:**
- Using a unique trigger token
- Dataset < ~15 images
- Identity already appears clearly
- Prompt flexibility matters

**✅ Train TE when:**
- Identity weak / inconsistent
- Token has semantic ambiguity
- You need stronger name→face binding

**Rules**
- LR = 0.3–0.5 × UNet LR
- Train TE for ≤50% of steps
- Freeze TE for final refinement
- Use prior regularization

---

## 6. Dataset & Captioning (Gold Standard)

### Image Selection
- 15–30 images minimum
- Mixed angles (front / ¾ / profile)
- Varied lighting & environments
- Mix of portrait + mid/full body
- No repeated background dominance

### Caption Rules

**Good**
```
<token>, a man standing outdoors
<token> wearing a black jacket
```

**Avoid**
- Overly verbose captions
- Repeating identical phrasing
- Describing permanent traits inconsistently
- Negative phrasing

### Trigger Token
- Use a unique, uncommon string
- Appear in every subject caption
- Never appear in regularization captions

### Regularization Images
- 20–50 generic people
- Caption: `a person` / `a man`
- Prevents concept collapse

---

## 7. Evaluation Protocol

### Fixed Prompt Set
- Neutral portrait
- New environment
- New outfit
- New style
- Control (no token)

### LoRA Strength Sweep
- 0.5 / 0.8 / 1.0 / 1.2

### Identity Metrics
- Human inspection
- Face-embedding similarity (ArcFace)
- Diversity across seeds

### Overfit Signals
- Same pose every time
- Carbon-copy images
- Loss still dropping, visuals not improving

---

## 8. HunyuanImage / Hunyuan DiT — Status

**Supported Models**
- Hunyuan-DiT v1.2
- HunyuanImage-2.1

**Supported Pipelines**
- `sd-scripts` (UNet-only LoRA)
- `diffusion-pipe` (best current option)

**Recommendation**
> Not yet as mature as FLUX for identity LoRA
> Use experimentally only

---

## 9. Failure Modes & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| NaNs | LR too high, fp16 overflow | Lower LR, use bf16 |
| Weak identity | Undertraining | More steps, higher rank |
| Sameface | Overtraining | Early stop, dropout, priors |
| Waxy skin | Over-smoothing | Lower noise offset |
| Angle failure | Poor dataset | Add side/profile images |
| Prompt brittleness | TE overfit | Freeze TE early |

---

## 10. Reproducibility & Pinning

### Known-Good Environment
- Python: 3.10
- PyTorch: 2.5.x
- CUDA: 11.8 / 12.x
- Accelerate: >=0.21
- bitsandbytes: >=0.42

### Recommended Commits
- `sd-scripts`: stable post-FLUX merge (late 2025)
- `diffusion-pipe`: fallback `6940992` if HEAD breaks

### Always Record
- Tool commit hashes
- Training config
- Dataset hash
- Seed
- LoRA rank / LR / steps

---

## Canonical Rule

> **If identity realism degrades, fix the dataset first — not the optimizer.**

# Report Summary: Identity LoRA Training Defaults

> Extracted from `DeepResearchReport.md` (State of the Art – December 2025)

---

## Pipeline Choice

**Engine**: `kohya-ss/sd-scripts` (CLI-first)

> Source: Section 1 "Executive Recommendation"
> - Highest training quality ceiling for identity realism
> - Most mature FLUX support (FP8, block swap, SNR loss, bucketing)
> - Fully scriptable, deterministic, automation-friendly

---

## Fast Iteration Profile (Section 4.1)

**Purpose**: Validate dataset, detect early overfit, <1 hour runtime

| Parameter | Value | Source |
|-----------|-------|--------|
| Resolution | `512x512` | Section 4.1 Core Settings |
| Steps | `1500` | Section 4.1 Core Settings |
| Rank | `32` | Section 4.1 Core Settings |
| Alpha | `32` | Section 4.1 Core Settings |
| Optimizer | `Adafactor` | Section 4.1 Core Settings |
| Optimizer Args | `relative_step=False scale_parameter=False` | Section 4.1 CLI Script |
| Learning Rate (UNet) | `1e-4` | Section 4.1 Core Settings |
| Learning Rate (TE) | `0` (frozen) | Section 4.1 Core Settings |
| LR Scheduler | `constant_with_warmup` | Section 4.1 CLI Script |
| Warmup Steps | `100` | Section 4.1 CLI Script |
| Mixed Precision | `bf16` | Section 3 Precision & Memory |
| FP8 Base | `true` | Section 3 Precision & Memory |
| Bucketing | `enabled` | Section 4.1 CLI Script |
| Min Bucket Reso | `256` | Section 4.1 CLI Script |
| Max Bucket Reso | `768` | Section 4.1 CLI Script |
| Noise Offset | `0.05` | Section 4.1 Core Settings |
| Gradient Checkpointing | `true` | Section 4.1 CLI Script |

---

## Final High-Fidelity Profile (Section 4.2)

**Purpose**: Maximum realism, strong identity lock, robust generalization

| Parameter | Value | Source |
|-----------|-------|--------|
| Resolution | `768` base | Section 4.2 Core Settings |
| Steps | `2000–3000` | Section 4.2 Core Settings |
| Rank | `64` | Section 4.2 Core Settings |
| Alpha | `64` | Section 4.2 Core Settings |
| Dropout | `0.1` | Section 4.2 Core Settings |
| Optimizer | `Adafactor` | Section 4.2 Core Settings |
| Learning Rate | `1e-4` | Section 4.2 High-Fidelity Config |
| LR Scheduler | `cosine` | Section 4.2 Core Settings |
| Warmup Steps | `500` | Section 4.2 High-Fidelity Config |
| Mixed Precision | `bf16` | Section 3 Precision & Memory |
| FP8 Base | `true` | Section 3 Precision & Memory |
| Bucketing | `enabled` | Section 4.2 High-Fidelity Config |
| Min Bucket Reso | `384` | Section 4.2 High-Fidelity Config |
| Max Bucket Reso | `1024` | Section 4.2 High-Fidelity Config |
| Noise Offset | `0.1` | Section 4.2 Core Settings |
| Min SNR Gamma | `5.0` | Section 4.2 High-Fidelity Config |
| Prior Regularization | Recommended | Section 4.2 Core Settings |
| Gradient Checkpointing | `true` | Section 4.2 High-Fidelity Config |

---

## Text Encoder Training Policy (Section 5)

### Do NOT Train TE When:
- Using a unique trigger token
- Dataset < ~15 images
- Identity already appears clearly
- Prompt flexibility matters

### Train TE When:
- Identity weak / inconsistent
- Token has semantic ambiguity
- Need stronger name→face binding

### TE Training Rules:
| Rule | Value | Source |
|------|-------|--------|
| TE Learning Rate | `0.3–0.5 × UNet LR` | Section 5 Rules |
| TE Duration | `≤50% of steps` | Section 5 Rules |
| Final Phase | Freeze TE | Section 5 Rules |
| Regularization | Required when training TE | Section 5 Rules |

**Implementation**: Two-phase training
1. Phase 1: TE enabled for first 50% of steps
2. Phase 2: TE frozen for remaining steps

---

## Dataset Requirements (Section 6)

### Image Selection
- 15–30 images minimum
- Mixed angles (front / ¾ / profile)
- Varied lighting & environments
- Mix of portrait + mid/full body
- No repeated background dominance

### Caption Rules
- Use unique trigger token in every subject caption
- Format: `<token>, a man standing outdoors`
- Avoid verbose captions, identical phrasing, negative phrasing

### Regularization Images
- 20–50 generic people
- Caption: `a person` / `a man` (NO trigger token)
- Prevents concept collapse

---

## Evaluation Protocol (Section 7)

### Fixed Prompt Set
1. Neutral portrait
2. New environment
3. New outfit
4. New style
5. Control (no token)

### LoRA Strength Sweep
- Test at: `0.5`, `0.8`, `1.0`, `1.2`

### Identity Metrics
- Human inspection
- Face-embedding similarity (ArcFace)
- Diversity across seeds

### Overfit Signals
- Same pose every time
- Carbon-copy images
- Loss still dropping, visuals not improving

---

## Failure Modes & Fixes (Section 9)

| Issue | Cause | Fix |
|-------|-------|-----|
| NaNs | LR too high, fp16 overflow | Lower LR, use bf16 |
| Weak identity | Undertraining | More steps, higher rank |
| Sameface | Overtraining | Early stop, dropout, priors |
| Waxy skin | Over-smoothing | Lower noise offset |
| Angle failure | Poor dataset | Add side/profile images |
| Prompt brittleness | TE overfit | Freeze TE early |

---

## Environment Pinning Requirements (Section 10)

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.5.x |
| CUDA | 11.8 / 12.x |
| Accelerate | >=0.21 |
| bitsandbytes | >=0.42 |

### sd-scripts Commit
- Use stable post-FLUX merge (late 2025)
- Pin to specific commit hash

---

## Canonical Rule (Section 10)

> **If identity realism degrades, fix the dataset first — not the optimizer.**

# Repository Structure & Responsibility Map

**Generated**: 2025-12-18
**Purpose**: Evidence-based analysis of repo organization, source-of-truth locations, and maintenance risks
**Scope**: Read-only audit for ChatGPT critique alongside `flux_lora_pipeline_diagnostic.md`

---

## 1. Executive Overview

This repository provides a **FLUX.1-dev identity LoRA training pipeline** using kohya-ss `sd-scripts`. The canonical user journey is: (1) build/pull Docker image → (2) launch RunPod pod → (3) upload dataset → (4) run analysis → (5) execute training script → (6) evaluate samples.

**Critical Finding**: The repo has **two configuration systems that are not connected**:
- **Shell scripts** (source of truth): `train_flux_fast.sh`, `train_flux_final.sh` build training commands directly
- **TOML configs** (unused): `configs/flux_fast.toml`, `configs/flux_final.toml` exist but are **never loaded**

This creates maintenance risk where changes to TOML configs have no effect, and documentation references config files that don't actually control training.

---

## 2. High-Level Tree (Top 4 Levels)

```
lora_training/
├── CLAUDE.md                     # Claude Code project instructions
├── DeepResearchReport.md         # RESEARCH SOURCE OF TRUTH (training methodology)
├── Dockerfile                    # Root-level build file (copy of docker/Dockerfile)
├── README.md                     # User-facing documentation
├── configs/
│   ├── flux_fast.toml            # ⚠️ UNUSED - Not loaded by any script
│   ├── flux_final.toml           # ⚠️ UNUSED - Not loaded by any script
│   └── sample_prompts.txt        # ✓ USED - Referenced by training scripts
├── data/
│   ├── subject/                  # User dataset mount point
│   │   ├── images/.gitkeep
│   │   └── captions/.gitkeep
│   └── reg/                      # Regularization images (optional)
│       ├── images/.gitkeep
│       └── captions/.gitkeep
├── docker/
│   ├── Dockerfile                # Container definition
│   ├── env.sh                    # ✓ USED - Sourced by training scripts
│   └── start.sh                  # Container entrypoint
├── docs/
│   ├── ENV_PINNING.md
│   ├── EVAL_PROTOCOL.md
│   ├── REPORT_SUMMARY.md
│   ├── RUNPOD_TEMPLATE.md
│   ├── TRAINING_GUIDE.md         # User guide (has doc/impl mismatches)
│   └── patches.md
├── logs/                         # Runtime logs output
├── output/                       # Training artifacts output
│   └── samples/                  # Generated sample images
├── reports/                      # Diagnostic reports
├── scripts/
│   ├── analyze_dataset.py        # ✓ Dataset validation tool
│   ├── download_models.sh        # Local: HF download + R2 upload
│   ├── setup.sh                  # Local setup (clones sd-scripts)
│   ├── sync_models_r2.sh         # Pod: Fast R2 model sync
│   ├── telemetry.sh              # GPU monitoring loop
│   ├── tmux_train.sh             # tmux orchestrator
│   ├── train_dashboard.sh        # ✓ PRIMARY - Dashboard training UI
│   ├── train_flux_fast.sh        # ✓ PRIMARY - Fast iteration script
│   ├── train_flux_final.sh       # ✓ PRIMARY - Production script
│   ├── train_remote.sh           # Mac: All-in-one remote training
│   ├── upload_dataset.sh         # Mac: Dataset upload utility
│   └── watch_samples.sh          # Sample image watcher
└── third_party/                  # sd-scripts symlink target
```

---

## 3. Responsibility Map (Core Surfaces)

| Component | Files | Responsibility | Inputs | Outputs | Source of Truth | Notes/Risks |
|-----------|-------|----------------|--------|---------|-----------------|-------------|
| **Training Engine** | `scripts/train_flux_fast.sh`, `scripts/train_flux_final.sh` | Build & execute accelerate command | `env.sh`, `MODEL_PATH`, `DATA_DIR` | LoRA `.safetensors`, logs | **Scripts are SoT** | Hardcodes all params; ignores TOML |
| **Dashboard Training** | `scripts/train_dashboard.sh` | Interactive training with progress UI | Profile selection, env vars | Same as above + live progress | Script is SoT | Parallel implementation to fast/final |
| **Environment Config** | `docker/env.sh` | Define paths and training defaults | None (static) | Exports env vars | Partial SoT | Many vars defined but **not used** |
| **TOML Configs** | `configs/flux_fast.toml`, `configs/flux_final.toml` | (Intended) Training configuration | N/A | N/A | **UNUSED** | Zero scripts reference these |
| **Sample Prompts** | `configs/sample_prompts.txt` | Evaluation prompts during training | None | Passed to `--sample_prompts` | Used correctly | 5 prompts, trigger "TOK Woman" |
| **Container Boot** | `docker/start.sh` | SSH, model sync, healthcheck, dirs | Docker env | Running services, dirs | Used correctly | Calls `sync_models_r2.sh` |
| **Dataset Validation** | `scripts/analyze_dataset.py` | Quality checks, blur/duplicate detection | `DATA_DIR` or `--data-dir` | `logs/dataset_report.json` | Used correctly | Comprehensive analysis |
| **Model Download** | `scripts/download_models.sh` | HF download + R2 upload | `HF_TOKEN`, R2 creds | Models in `~/models/` + R2 | Used correctly | Local Mac script |
| **Model Sync** | `scripts/sync_models_r2.sh` | Fast R2→pod model download | R2 creds | `/workspace/models/flux1-dev` | Used correctly | 2-5 min vs HF hours |
| **Research Docs** | `DeepResearchReport.md` | SOTA methodology reference | N/A | N/A | Research SoT | Scripts don't fully implement |
| **User Docs** | `README.md`, `docs/TRAINING_GUIDE.md` | User instructions | N/A | N/A | Doc SoT | Contains inaccuracies |

---

## 4. Entrypoints and How They Compose

### User-Facing Entrypoints (Run Directly)

| Script | Purpose | Calls/Uses |
|--------|---------|------------|
| `scripts/train_flux_fast.sh` | Fast iteration training (~1hr) | Sources `docker/env.sh`, calls `accelerate launch flux_train_network.py` |
| `scripts/train_flux_final.sh` | Production training (~3-4hr) | Sources `docker/env.sh`, calls `accelerate launch flux_train_network.py` |
| `scripts/train_dashboard.sh` | Dashboard with progress bar | Sources `docker/env.sh`, builds command internally |
| `scripts/tmux_train.sh` | tmux orchestrator | Calls `train_flux_fast.sh` or `train_flux_final.sh` + `telemetry.sh` |
| `scripts/train_remote.sh` | Mac all-in-one remote | Calls `upload_dataset.sh` logic + SSH + `train_dashboard.sh` |
| `scripts/analyze_dataset.py` | Dataset validation | Standalone Python |
| `scripts/setup.sh` | Local environment setup | Clones sd-scripts, creates dirs |

### Internal/Support Scripts

| Script | Called By | Purpose |
|--------|-----------|---------|
| `docker/env.sh` | `train_flux_fast.sh`, `train_flux_final.sh`, `train_dashboard.sh` | Exports env vars |
| `docker/start.sh` | Docker ENTRYPOINT | Container initialization |
| `scripts/telemetry.sh` | `tmux_train.sh` | GPU monitoring loop |
| `scripts/sync_models_r2.sh` | `start.sh` (auto), user (manual) | R2 model download |
| `scripts/download_models.sh` | User (local Mac) | HF download + R2 upload |
| `scripts/upload_dataset.sh` | User (local Mac) | Dataset upload to pod |
| `scripts/watch_samples.sh` | User | Watch sample dir for new images |

### Call Graph

```
[User on Mac]
    │
    ├─→ scripts/train_remote.sh
    │       ├─→ SSH connection test
    │       ├─→ Dataset upload (scp)
    │       └─→ SSH exec: scripts/train_dashboard.sh
    │
[User on Pod]
    │
    ├─→ scripts/train_flux_fast.sh
    │       └─→ source docker/env.sh
    │       └─→ accelerate launch flux_train_network.py
    │
    ├─→ scripts/train_flux_final.sh
    │       └─→ source docker/env.sh
    │       └─→ accelerate launch flux_train_network.py
    │
    ├─→ scripts/tmux_train.sh [fast|final]
    │       ├─→ scripts/train_flux_fast.sh OR train_flux_final.sh
    │       ├─→ scripts/telemetry.sh
    │       └─→ tail -f logs/<run>.log
    │
    └─→ scripts/train_dashboard.sh
            └─→ source docker/env.sh
            └─→ accelerate launch flux_train_network.py
```

---

## 5. Configuration Topology

### Config File Status

| File | Type | Status | Evidence |
|------|------|--------|----------|
| `docker/env.sh` | Shell exports | **PARTIALLY USED** | Sourced by scripts, but many vars ignored |
| `configs/flux_fast.toml` | TOML | **UNUSED** | `grep -rn "flux_fast.toml" scripts/` returns empty |
| `configs/flux_final.toml` | TOML | **UNUSED** | `grep -rn "flux_final.toml" scripts/` returns empty |
| `configs/sample_prompts.txt` | Text | **USED** | Referenced in `train_flux_fast.sh:186`, `train_flux_final.sh:192` |
| `.claude/settings.local.json` | JSON | Used by Claude Code | IDE config |
| `logs/test_report.json` | JSON | Output artifact | Generated by analyze_dataset.py |

### env.sh Variable Usage Analysis

| Variable | Defined In | Used In Scripts | Status |
|----------|------------|-----------------|--------|
| `WORKSPACE` | env.sh:10 | All scripts | ✓ Used |
| `SDSCRIPTS` | env.sh:11 | train_*.sh | ✓ Used |
| `MODEL_PATH` | env.sh:17 | train_*.sh | ✓ Used |
| `DATA_DIR` | env.sh:23 | train_*.sh | ✓ Used |
| `FAST_RANK` | env.sh:31 | train_flux_fast.sh:37 | ✓ Used |
| `FINAL_RANK` | env.sh:41 | train_flux_final.sh:37 | ✓ Used |
| `FAST_NOISE_OFFSET` | env.sh:35 | **NONE** | ⚠️ **NOT USED** |
| `FINAL_NOISE_OFFSET` | env.sh:48 | **NONE** | ⚠️ **NOT USED** |
| `FINAL_DROPOUT` | env.sh:47 | **NONE** | ⚠️ **NOT USED** |
| `FINAL_SNR_GAMMA` | env.sh:52 | **NONE** | ⚠️ **NOT USED** |
| `GRAD_ACCUM` | env.sh:57 | **NONE** | ⚠️ **NOT USED** |
| `FP8_BASE` | env.sh:89 | **NONE** | ⚠️ **NOT USED** |
| `ENABLE_TE` | env.sh:63 | **NONE** | ⚠️ **NOT USED** |
| `SAMPLE_PROMPTS` | env.sh:82 | train_*.sh | ✓ Used |

### Evidence: TOML Files Not Referenced

```bash
# Command executed:
grep -rn "\.toml" scripts/*.sh

# Output: (empty - no matches)
```

```bash
# Command executed:
grep -r "flux_fast.toml\|flux_final.toml" . --include="*.sh"

# Output: (empty - no matches)
```

### Config Drift: TOML vs Scripts

| Parameter | flux_fast.toml | train_flux_fast.sh | Drift? |
|-----------|----------------|-------------------|--------|
| `network_module` | `networks.lora` | `networks.lora_flux` | **YES** |
| `optimizer_type` | `Adafactor` | `AdamW8bit` | **YES** |
| `network_alpha` | `32` | `1` (hardcoded) | **YES** |
| `noise_offset` | `0.05` | Not set | **YES** |
| `gradient_accumulation_steps` | `4` | Not set | **YES** |
| `fp8_base` | `true` | Not set | **YES** |

| Parameter | flux_final.toml | train_flux_final.sh | Drift? |
|-----------|-----------------|---------------------|--------|
| `network_module` | `networks.lora` | `networks.lora_flux` | **YES** |
| `optimizer_type` | `Adafactor` | `AdamW8bit` | **YES** |
| `network_alpha` | `64` | `1` (hardcoded) | **YES** |
| `network_dropout` | `0.1` | Not set | **YES** |
| `noise_offset` | `0.1` | Not set | **YES** |
| `min_snr_gamma` | `5.0` | Not set | **YES** |
| `gradient_accumulation_steps` | `4` | Not set | **YES** |
| `fp8_base` | `true` | Not set | **YES** |

---

## 6. Data & Output Contracts

### Dataset Input Contract

**Location**: `/workspace/lora_training/data/subject/` (or `DATA_DIR` env var)

**Accepted Structures**:

```
# Option A: Flat structure (scripts prefer this)
data/subject/
├── photo1.jpg
├── photo1.txt
├── photo2.png
└── photo2.txt

# Option B: Separate directories (analyze_dataset.py handles both)
data/subject/
├── images/
│   └── photo1.jpg
└── captions/
    └── photo1.txt
```

**File Requirements** (from `analyze_dataset.py`):
- Image formats: `.jpg`, `.jpeg`, `.png`, `.webp`
- Caption format: `.txt` with same basename as image
- Caption content: `<trigger_token>, description`
- Minimum: 15 images (warning below this)
- Recommended: 15-30 images

**Evidence** (train_flux_fast.sh:83-88):
```bash
IMG_COUNT=$(find "${DATA_DIR}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) 2>/dev/null | wc -l | tr -d ' ')
if [ "$IMG_COUNT" -eq 0 ]; then
    echo -e "${RED}[ERROR]${NC} No images found in dataset: ${DATA_DIR}"
    exit 1
fi
```

### Output Contract

| Output Type | Location | Naming Convention |
|-------------|----------|-------------------|
| LoRA model | `output/<RUN_NAME>.safetensors` | Based on `RUN_NAME` env var |
| Checkpoints | `output/<RUN_NAME>-step<N>.safetensors` | Every `SAVE_EVERY_N_STEPS` |
| Sample images | `output/samples/<RUN_NAME>/` | Generated every `SAMPLE_EVERY_N_STEPS` |
| Training log | `logs/<RUN_NAME>.log` | Full training output |
| Dataset report | `logs/dataset_report.json` | From `analyze_dataset.py` |
| pip freeze | `logs/pip_freeze.txt` | From `start.sh` |

---

## 7. Documentation as Source of Truth

### Document Inventory

| Document | Claims | Status |
|----------|--------|--------|
| `DeepResearchReport.md` | SOTA methodology, hyperparameters | **Research SoT** - Scripts don't fully implement |
| `README.md` | Quick start, structure overview | Contains inaccuracies |
| `docs/TRAINING_GUIDE.md` | Detailed training guide | Contains inaccuracies |
| `CLAUDE.md` | Project instructions | Accurate |

### Documentation vs Implementation Mismatches

| Document | Claim | Reality | Evidence |
|----------|-------|---------|----------|
| `README.md:170-172` | "flux_fast.toml - Fast profile config" | TOML files are not used | `grep` returns empty |
| `docs/TRAINING_GUIDE.md:96-101` | "Rank/Alpha: 32/32" for fast | Scripts use `network_alpha=1` | train_flux_fast.sh:139 |
| `docs/TRAINING_GUIDE.md:100` | "Noise Offset: 0.05" for fast | Not set in script | `grep noise_offset train_flux_fast.sh` empty |
| `docs/TRAINING_GUIDE.md:118` | "Dropout: 0.1" for final | Not set in script | `grep dropout train_flux_final.sh` empty |
| `docs/TRAINING_GUIDE.md:123` | "SNR Gamma: 5.0" for final | Not set in script | `grep snr train_flux_final.sh` empty |
| `docs/TRAINING_GUIDE.md:326-327` | `NOISE_OFFSET=0.05` as fix | Variable not used by scripts | env.sh defines but not wired |

---

## 8. Maintenance Risk Register

| Risk | Impact | Evidence | Recommended Fix | Priority |
|------|--------|----------|-----------------|----------|
| **TOML configs exist but unused** | High - Misleading, changes have no effect | `grep "\.toml" scripts/` empty | Either: (a) Delete TOMLs + update docs, OR (b) Refactor scripts to load TOMLs | **P0** |
| **env.sh vars defined but not wired** | High - 8+ vars defined but ignored | `FAST_NOISE_OFFSET`, `FINAL_DROPOUT`, etc. not in script commands | Wire vars: `--noise_offset=${FAST_NOISE_OFFSET}` | **P0** |
| **network_alpha=1 hardcoded** | High - Wrong LR scaling | train_flux_fast.sh:139, train_flux_final.sh:144 | Change to `--network_alpha=${RANK}` | **P0** |
| **Three parallel training scripts** | Medium - Maintenance burden | `train_flux_fast.sh`, `train_flux_final.sh`, `train_dashboard.sh` | Consolidate into single parameterized script | **P1** |
| **Duplicate RANK definitions** | Medium - Inconsistency risk | env.sh, train_flux_fast.sh, train_flux_final.sh, train_dashboard.sh | Single source in env.sh | **P1** |
| **Docs claim TOML usage** | Medium - User confusion | README.md, TRAINING_GUIDE.md | Update docs to reflect reality | **P1** |
| **Root Dockerfile duplicates docker/Dockerfile** | Low - Sync risk | Two Dockerfiles with same content | Symlink or delete one | **P2** |
| **Hardcoded secrets in scripts** | Low - Security | R2 keys in download_models.sh, sync_models_r2.sh | Move to env vars or secrets manager | **P2** |
| **sd-scripts version inconsistency** | Low - Potential issues | Dockerfile: sd3 branch, setup.sh: v0.9.2 tag | Standardize on one version reference | **P2** |

---

## 9. Appendix

### Commands Executed

```bash
# Tree inventory
find . -maxdepth 4 -type f -o -type d 2>/dev/null | grep -v '__pycache__' | grep -v '.git/'

# Script listing
ls -la scripts/*.sh scripts/*.py

# Config file discovery
find . -name "*.toml" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "env.sh" -o -name "*.txt"

# TOML reference check (CRITICAL - proves non-use)
grep -rn "\.toml" scripts/*.sh  # Output: empty
grep -r "flux_fast.toml\|flux_final.toml" . --include="*.sh"  # Output: empty

# env.sh reference check
grep -r "env.sh" . --include="*.sh" --include="*.md"

# sample_prompts reference check
grep -r "sample_prompts" . --include="*.sh" --include="*.py"

# Duplicate definition check
grep -r "network_dim" . --include="*.sh" --include="*.toml"
grep -r "network_alpha" . --include="*.sh" --include="*.toml"
grep -r "network_module" . --include="*.sh" --include="*.toml"
grep -r "optimizer_type" . --include="*.sh" --include="*.toml"
grep -r "noise_offset" . --include="*.sh" --include="*.toml" --include="*.md"
grep -r "RANK" scripts/*.sh
```

### Key Evidence Excerpts

**train_flux_fast.sh:137-139** (network_alpha hardcoded):
```bash
CMD="${CMD} --network_module=networks.lora_flux"
CMD="${CMD} --network_dim=${RANK}"
CMD="${CMD} --network_alpha=1"
```

**configs/flux_fast.toml:17-20** (different values):
```toml
[network]
network_module = "networks.lora"
network_dim = 32
network_alpha = 32
```

**docker/env.sh:35,47-48** (unused vars):
```bash
export FAST_NOISE_OFFSET="${FAST_NOISE_OFFSET:-0.05}"
export FINAL_DROPOUT="${FINAL_DROPOUT:-0.1}"
export FINAL_NOISE_OFFSET="${FINAL_NOISE_OFFSET:-0.1}"
```

### Assumptions

1. Scripts in `scripts/` are the intended user entrypoints (verified by README.md and CLAUDE.md)
2. `docker/env.sh` is intended to be the single source of configurable defaults (verified by comments)
3. TOML configs were intended for use but never integrated (based on existence and content)
4. `DeepResearchReport.md` represents the intended "correct" configuration (based on references in docs)

### Unknowns

1. **Why TOMLs exist but aren't used**: Possibly planned feature, possibly legacy from different training approach
2. **Why three parallel training scripts**: Historical evolution unclear; could consolidate
3. **Intended use of `ENABLE_TE`**: Defined in env.sh with two-phase logic, but no script implements it

---

## Next Actions (Structural Improvements)

1. **Decide TOML fate** (P0): Either delete `configs/flux_*.toml` and update docs, OR refactor scripts to load them with `--config_file` parameter
2. **Wire env.sh variables** (P0): Add `--noise_offset`, `--network_dropout`, `--min_snr_gamma`, `--gradient_accumulation_steps` to training commands using env.sh values
3. **Fix network_alpha** (P0): Change `--network_alpha=1` to `--network_alpha=${RANK}` in all three training scripts
4. **Consolidate training scripts** (P1): Merge `train_flux_fast.sh`, `train_flux_final.sh`, `train_dashboard.sh` into single parameterized script
5. **Update documentation** (P1): Remove references to TOML configs OR add TOML loading; fix claimed vs actual parameter values
6. **Deduplicate Dockerfiles** (P2): Keep only `docker/Dockerfile` or create symlink from root

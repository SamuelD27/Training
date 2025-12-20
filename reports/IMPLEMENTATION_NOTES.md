# Implementation Notes: Configuration Drift Fixes

**Date**: 2025-12-18
**Author**: Claude Code (Opus 4.5)
**Audit Sources**: `reports/flux_lora_pipeline_diagnostic.md`, `reports/repo_structure_map.md`

---

## Summary

This document describes the changes made to eliminate configuration drift and make training execution match documented intent in the FLUX.1-dev Identity LoRA training pipeline.

---

## Changes by Priority

### P0 Fixes (Critical)

#### 1. Fixed `network_alpha` Misconfiguration

**Problem**: Scripts hardcoded `--network_alpha=1` while rank was 32/64. This creates a 32x/64x effective learning rate multiplier.

**Solution**:
- `build_train_cmd.py:213-217` - Validates alpha matches rank, errors if mismatch
- `configs/flux_fast.toml:24-25` - `network_alpha = 32` (matches dim)
- `configs/flux_final.toml:28-29` - `network_alpha = 64` (matches dim)

**Audit Reference**: `flux_lora_pipeline_diagnostic.md` Section 3.5, Line 175

#### 2. Wired Missing `noise_offset`

**Problem**: Parameter defined in env.sh but never passed to training command.

**Solution**:
- `build_train_cmd.py:275-277` - Adds `--noise_offset` if > 0
- `configs/flux_fast.toml:62` - `noise_offset = 0.05`
- `configs/flux_final.toml:68` - `noise_offset = 0.1`

**Audit Reference**: `flux_lora_pipeline_diagnostic.md` Section 4, P0 table

#### 3. Wired Missing `network_dropout`

**Problem**: Dropout not passed to final profile command.

**Solution**:
- `build_train_cmd.py:221-223` - Adds `--network_dropout` if > 0
- `configs/flux_final.toml:30` - `network_dropout = 0.1`

**Audit Reference**: `flux_lora_pipeline_diagnostic.md` Section 3.5, Line 176-177

#### 4. Wired Missing `min_snr_gamma`

**Problem**: SNR gamma not passed to final profile command.

**Solution**:
- `build_train_cmd.py:279-281` - Adds `--min_snr_gamma` if > 0
- `configs/flux_final.toml:71` - `min_snr_gamma = 5.0`

**Audit Reference**: `flux_lora_pipeline_diagnostic.md` Section 4, P1 table

#### 5. Single Source of Truth for Configuration

**Problem**: TOML configs existed but were never loaded. Scripts hardcoded all parameters.

**Solution**:
- Created `scripts/build_train_cmd.py` - Canonical command builder
- TOML files are now the authoritative source for profile configs
- Shell scripts are thin wrappers that call `build_train_cmd.py`
- All three training paths (fast.sh, final.sh, dashboard.sh) use same command builder

**Files Changed**:
- `scripts/build_train_cmd.py` (NEW)
- `scripts/train_flux_fast.sh` (refactored to use build_train_cmd.py)
- `scripts/train_flux_final.sh` (refactored to use build_train_cmd.py)
- `scripts/train_dashboard.sh` (refactored to use build_train_cmd.py)
- `configs/flux_fast.toml` (updated with correct params)
- `configs/flux_final.toml` (updated with correct params)

**Audit Reference**: `repo_structure_map.md` Section 5, entire "Config Drift" table

---

### P1 Fixes (High Priority)

#### 1. Wired `gradient_accumulation_steps`

**Problem**: `GRAD_ACCUM=4` defined in env.sh but never used.

**Solution**:
- `build_train_cmd.py:235-237` - Adds `--gradient_accumulation_steps` if > 1
- `configs/flux_*.toml` - `gradient_accumulation_steps = 4`

**Audit Reference**: `flux_lora_pipeline_diagnostic.md` Section 4, P1 table

#### 2. Added `fp8_base` Toggle

**Problem**: FP8_BASE env var existed but was never wired.

**Solution**:
- `build_train_cmd.py:244-245` - Adds `--fp8_base` if `FP8_BASE=1`
- Enabled via environment: `FP8_BASE=1 bash scripts/train_flux_fast.sh`

**Audit Reference**: `flux_lora_pipeline_diagnostic.md` Section 3.7, Line 220

#### 3. Reproducibility Artifacts

**Problem**: No automatic capture of run configuration for audit.

**Solution**: Each training run now saves:
- `logs/<run>_command.txt` - Exact executed command
- `logs/<run>_repro.json` - Git hash, GPU info, hyperparameters
- `logs/<run>_pip_freeze.txt` - Python packages
- `logs/<run>_dataset_hash.txt` - Dataset fingerprint

**Implementation**:
- `build_train_cmd.py:generate_repro_info()` - Generates repro JSON
- `build_train_cmd.py:compute_dataset_hash()` - Dataset fingerprinting
- Shell scripts call these during execution

**Audit Reference**: `flux_lora_pipeline_diagnostic.md` Section 6 "Reproducibility Checklist"

#### 4. Resume Support

**Problem**: RESUME_FROM mentioned in docs but not implemented.

**Solution**:
- `build_train_cmd.py:302-307` - Adds `--network_weights` flag for resume
- Enabled via: `RESUME_FROM=/path/to/ckpt.safetensors bash scripts/train_flux_final.sh`

**Audit Reference**: `flux_lora_pipeline_diagnostic.md` Section 3.8, Line 242-243

#### 5. Dashboard Command Sync

**Problem**: Dashboard built command internally, could drift from launch scripts.

**Solution**:
- `train_dashboard.sh:349-352` - Now calls `build_train_cmd.py`
- Exact same command generation as `train_flux_fast.sh` and `train_flux_final.sh`

**Audit Reference**: `repo_structure_map.md` Section 3, "Three parallel training scripts"

---

### P2 Fixes (Minor)

#### 1. Added Dataloader Workers

**Solution**: `configs/flux_*.toml` - `max_data_loader_n_workers = 2`

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `scripts/build_train_cmd.py` | NEW | Single source of truth command builder |
| `scripts/validate_config_usage.py` | NEW | Validation script for P0/P1 checks |
| `scripts/train_flux_fast.sh` | MODIFIED | Thin wrapper using build_train_cmd.py |
| `scripts/train_flux_final.sh` | MODIFIED | Thin wrapper using build_train_cmd.py |
| `scripts/train_dashboard.sh` | MODIFIED | Uses build_train_cmd.py for command |
| `configs/flux_fast.toml` | MODIFIED | Fixed network_module, alpha, added params |
| `configs/flux_final.toml` | MODIFIED | Fixed network_module, alpha, added params |
| `docs/TRAINING_GUIDE.md` | MODIFIED | Updated for new SoT system |
| `README.md` | MODIFIED | Added configuration system section |
| `reports/IMPLEMENTATION_NOTES.md` | NEW | This file |

---

## Backward Compatibility

- **Same commands still work**: `bash scripts/train_flux_fast.sh`, `bash scripts/train_flux_final.sh`
- **Env var overrides work**: `RANK=48 MAX_STEPS=2000 bash scripts/train_flux_final.sh`
- **DRY_RUN mode works**: `DRY_RUN=1 bash scripts/train_flux_fast.sh`
- **Dashboard UX preserved**: Progress bar, GPU telemetry, sample display unchanged

---

## Non-Changes (Intentionally Preserved)

1. **kohya-ss sd-scripts** - Training engine unchanged, only flags passed differently
2. **Docker setup** - env.sh still defines paths, Dockerfile unchanged
3. **Dataset structure** - Same format expected
4. **Output locations** - Same paths for models, samples, logs

---

## Validation

Run the validation script to verify all P0/P1 fixes are in place:

```bash
python scripts/validate_config_usage.py
```

This checks:
1. Scripts reference `build_train_cmd.py`
2. TOML files are valid
3. Generated commands contain required flags
4. network_alpha equals network_dim
5. FLUX-specific parameters present

---

## Rollback Plan

If issues arise, revert these commits/files:

```bash
# Restore original scripts
git checkout HEAD~1 -- scripts/train_flux_fast.sh
git checkout HEAD~1 -- scripts/train_flux_final.sh
git checkout HEAD~1 -- scripts/train_dashboard.sh

# Remove new files
rm scripts/build_train_cmd.py
rm scripts/validate_config_usage.py

# Restore original TOML configs
git checkout HEAD~1 -- configs/flux_fast.toml
git checkout HEAD~1 -- configs/flux_final.toml
```

Or simply:
```bash
git revert HEAD
```

---

## Testing Commands

### Dry Run (Fast Profile)
```bash
DRY_RUN=1 bash scripts/train_flux_fast.sh
```

### Dry Run (Final Profile)
```bash
DRY_RUN=1 bash scripts/train_flux_final.sh
```

### Direct Command Builder Test
```bash
python scripts/build_train_cmd.py --profile fast --dry-run
python scripts/build_train_cmd.py --profile final --dry-run
```

### Validation
```bash
python scripts/validate_config_usage.py --verbose
```

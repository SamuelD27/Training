#!/usr/bin/env python3
"""
build_train_cmd.py - Single Source of Truth for FLUX.1 LoRA Training Commands

This script loads training configuration from TOML files and builds the exact
accelerate launch command for kohya-ss flux_train_network.py.

Audit Fixes Implemented:
- P0: network_alpha must match rank (validates and enforces)
- P0: noise_offset now wired from TOML/env
- P0: network_dropout now wired from TOML/env
- P0: min_snr_gamma now wired from TOML/env
- P1: gradient_accumulation_steps now wired
- P1: fp8_base toggle support
- P1: Resume from checkpoint support

Usage:
    python scripts/build_train_cmd.py --profile fast
    python scripts/build_train_cmd.py --profile final --dry-run
    python scripts/build_train_cmd.py --profile fast --output-json

    # With env overrides:
    RANK=48 MAX_STEPS=2000 python scripts/build_train_cmd.py --profile fast

    # Resume from checkpoint:
    RESUME_FROM=/path/to/checkpoint.safetensors python scripts/build_train_cmd.py --profile fast
"""

import argparse
import json
import os
import sys
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Try tomllib (Python 3.11+) or fall back to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("ERROR: No TOML library found. Install with: pip install tomli", file=sys.stderr)
        sys.exit(1)


# ============================================================================
# Default Paths (from docker/env.sh)
# ============================================================================
DEFAULT_WORKSPACE = os.environ.get("WORKSPACE", "/workspace/lora_training")
DEFAULT_SDSCRIPTS = os.environ.get("SDSCRIPTS", "/opt/sd-scripts")
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/models/flux1-dev")
DEFAULT_TEXT_ENCODER_PATH = os.environ.get("TEXT_ENCODER_PATH", "/workspace/models/text_encoders")
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", f"{DEFAULT_WORKSPACE}/data/subject")
DEFAULT_OUTPUT_DIR = os.environ.get("OUT_DIR", f"{DEFAULT_WORKSPACE}/output")
DEFAULT_LOG_DIR = os.environ.get("LOG_DIR", f"{DEFAULT_WORKSPACE}/logs")
DEFAULT_SAMPLE_PROMPTS = os.environ.get("SAMPLE_PROMPTS", f"{DEFAULT_WORKSPACE}/configs/sample_prompts.txt")


# ============================================================================
# Profile Defaults (fallbacks if TOML missing values)
# ============================================================================
PROFILE_DEFAULTS = {
    "fast": {
        "resolution": 512,
        "max_train_steps": 1500,
        "network_dim": 32,
        "network_alpha": 32,  # P0: alpha = rank
        "lr_warmup_steps": 100,
        "learning_rate": 1e-4,
        "lr_scheduler": "constant_with_warmup",
        "noise_offset": 0.05,  # P0: wired from report Section 4.1
        "network_dropout": 0.0,
        "min_snr_gamma": 0,  # disabled for fast
        "gradient_accumulation_steps": 4,  # P1: wired
        "min_bucket_reso": 256,
        "max_bucket_reso": 768,
        "optimizer_type": "AdamW8bit",
    },
    "final": {
        "resolution": 768,
        "max_train_steps": 2500,
        "network_dim": 64,
        "network_alpha": 64,  # P0: alpha = rank
        "lr_warmup_steps": 500,
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine_with_restarts",
        "lr_scheduler_num_cycles": 1,
        "noise_offset": 0.1,  # P0: wired from report Section 4.2
        "network_dropout": 0.1,  # P0: wired from report Section 4.2
        "min_snr_gamma": 5.0,  # P0: wired from report Section 4.2
        "gradient_accumulation_steps": 4,  # P1: wired
        "min_bucket_reso": 384,
        "max_bucket_reso": 1024,
        "optimizer_type": "AdamW8bit",
    },
}

# FLUX-specific parameters (always required, from kohya-ss docs)
FLUX_REQUIRED_PARAMS = {
    "guidance_scale": 1.0,
    "timestep_sampling": "flux_shift",
    "model_prediction_type": "raw",
    "discrete_flow_shift": 1.0,
}


def load_toml_config(profile: str, workspace: str) -> dict:
    """Load TOML config for the given profile."""
    config_path = Path(workspace) / "configs" / f"flux_{profile}.toml"

    if not config_path.exists():
        print(f"WARNING: TOML config not found: {config_path}", file=sys.stderr)
        print(f"WARNING: Using profile defaults for '{profile}'", file=sys.stderr)
        return {}

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def flatten_toml_config(toml_config: dict) -> dict:
    """Flatten nested TOML sections into a single dict."""
    flat = {}
    for section, values in toml_config.items():
        if isinstance(values, dict):
            flat.update(values)
        else:
            flat[section] = values
    return flat


def get_env_overrides() -> dict:
    """Get configuration overrides from environment variables."""
    overrides = {}

    # Map env vars to config keys
    env_mappings = {
        "RANK": "network_dim",
        "ALPHA": "network_alpha",
        "MAX_STEPS": "max_train_steps",
        "RESOLUTION": "resolution",
        "LEARNING_RATE": "learning_rate",
        "UNET_LR": "learning_rate",
        "WARMUP": "lr_warmup_steps",
        "NOISE_OFFSET": "noise_offset",
        "DROPOUT": "network_dropout",
        "SNR_GAMMA": "min_snr_gamma",
        "GRAD_ACCUM": "gradient_accumulation_steps",
        "BATCH_SIZE": "train_batch_size",
        "SEED": "seed",
        "OPTIMIZER": "optimizer_type",
        "SCHEDULER": "lr_scheduler",
        "MIN_BUCKET": "min_bucket_reso",
        "MAX_BUCKET": "max_bucket_reso",
    }

    for env_var, config_key in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None and value.strip():  # Skip empty strings
            # Try to convert to appropriate type
            try:
                if config_key in ["network_dim", "network_alpha", "max_train_steps",
                                  "lr_warmup_steps", "gradient_accumulation_steps",
                                  "train_batch_size", "seed", "resolution",
                                  "min_bucket_reso", "max_bucket_reso"]:
                    overrides[config_key] = int(value)
                elif config_key in ["learning_rate", "noise_offset", "network_dropout", "min_snr_gamma"]:
                    overrides[config_key] = float(value)
                else:
                    overrides[config_key] = value
            except ValueError:
                pass  # Skip invalid values instead of storing them

    return overrides


def validate_config(config: dict, allow_alpha_mismatch: bool = False) -> list:
    """
    Validate configuration and return list of warnings.

    P0 Fix: Ensures network_alpha matches network_dim unless explicitly allowed.
    """
    warnings = []

    rank = config.get("network_dim", 32)
    alpha = config.get("network_alpha", rank)

    # P0: network_alpha should match rank for FLUX LoRA
    if alpha != rank:
        if allow_alpha_mismatch:
            warnings.append(f"P0 WARNING: network_alpha ({alpha}) != network_dim ({rank}). "
                          f"Proceeding due to --allow-alpha-mismatch flag.")
        else:
            print(f"P0 ERROR: network_alpha ({alpha}) must equal network_dim ({rank}).", file=sys.stderr)
            print("This is a critical misconfiguration that causes incorrect LR scaling.", file=sys.stderr)
            print("If intentional, use --allow-alpha-mismatch flag.", file=sys.stderr)
            sys.exit(1)

    # Warn about potentially problematic values
    try:
        lr = config.get("learning_rate", 1e-4)
        lr = float(lr) if lr else 1e-4
        if lr > 5e-4:
            warnings.append(f"WARNING: Learning rate {lr} may be too high (risk of NaN)")
    except (ValueError, TypeError):
        pass

    try:
        noise = config.get("noise_offset", 0)
        noise = float(noise) if noise else 0
        if noise > 0.15:
            warnings.append(f"WARNING: noise_offset {noise} > 0.15 may cause artifacts")
    except (ValueError, TypeError):
        pass

    return warnings


def build_command(config: dict, paths: dict, run_name: str, resume_from: str = None,
                  fp8_base: bool = False, profile: str = "fast") -> str:
    """
    Build the accelerate launch command for kohya-ss flux_train_network.py.

    This is the canonical command builder - all training scripts should use this.
    """
    train_script = f"{paths['sdscripts']}/flux_train_network.py"

    # Start with accelerate launch
    cmd_parts = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        train_script,
    ]

    # Model paths (FLUX-specific)
    cmd_parts.extend([
        f"--pretrained_model_name_or_path={paths['model']}/flux1-dev.safetensors",
        f"--clip_l={paths['text_encoder']}/clip_l.safetensors",
        f"--t5xxl={paths['text_encoder']}/t5xxl_fp16.safetensors",
        f"--ae={paths['model']}/ae.safetensors",
    ])

    # Dataset
    cmd_parts.append(f"--train_data_dir={paths['data']}")

    # Output
    cmd_parts.extend([
        f"--output_dir={paths['output']}",
        f"--output_name={run_name}",
        "--save_model_as=safetensors",
    ])

    # LoRA network (FLUX-specific module)
    # P0 Fix: network_module is networks.lora_flux (not networks.lora as in TOML)
    cmd_parts.extend([
        "--network_module=networks.lora_flux",
        f"--network_dim={config.get('network_dim', 32)}",
        f"--network_alpha={config.get('network_alpha', config.get('network_dim', 32))}",  # P0: alpha=rank
    ])

    # P0 Fix: network_dropout (final profile)
    dropout = config.get("network_dropout", 0)
    if dropout > 0:
        cmd_parts.append(f"--network_dropout={dropout}")

    # Training parameters
    lr = config.get("learning_rate", 1e-4)
    lr = float(lr) if lr else 1e-4
    cmd_parts.extend([
        f"--learning_rate={lr}",
        f"--optimizer_type={config.get('optimizer_type', 'AdamW8bit')}",
        f"--lr_scheduler={config.get('lr_scheduler', 'constant_with_warmup')}",
        f"--lr_warmup_steps={config.get('lr_warmup_steps', 100)}",
    ])

    # LR scheduler cycles (for cosine)
    if config.get("lr_scheduler") in ["cosine_with_restarts", "cosine"]:
        cycles = config.get("lr_scheduler_num_cycles", 1)
        cmd_parts.append(f"--lr_scheduler_num_cycles={cycles}")

    # P1 Fix: gradient_accumulation_steps
    grad_accum = config.get("gradient_accumulation_steps", 4)
    if grad_accum > 1:
        cmd_parts.append(f"--gradient_accumulation_steps={grad_accum}")

    # Hardware optimization
    cmd_parts.extend([
        "--sdpa",
        f"--mixed_precision={config.get('mixed_precision', 'bf16')}",
        "--gradient_checkpointing",
    ])

    # P1 Fix: fp8_base toggle
    if fp8_base:
        cmd_parts.append("--fp8_base")

    # FLUX-specific settings (required)
    for param, value in FLUX_REQUIRED_PARAMS.items():
        cmd_parts.append(f"--{param}={value}")

    # Memory optimization
    blocks_to_swap = config.get("blocks_to_swap", 18)
    cmd_parts.extend([
        f"--blocks_to_swap={blocks_to_swap}",
        "--cache_text_encoder_outputs",
        "--cache_latents",
        "--cache_latents_to_disk",
    ])

    # Resolution and bucketing
    resolution = config.get("resolution", 512)
    cmd_parts.extend([
        f"--resolution={resolution}",
        "--enable_bucket",
        f"--min_bucket_reso={config.get('min_bucket_reso', 256)}",
        f"--max_bucket_reso={config.get('max_bucket_reso', 1024)}",
        "--bucket_reso_steps=64",
    ])

    # Batch and steps
    cmd_parts.extend([
        f"--train_batch_size={config.get('train_batch_size', 1)}",
        f"--max_train_steps={config.get('max_train_steps', 1500)}",
    ])

    # P0 Fix: noise_offset
    noise_offset = config.get("noise_offset", 0)
    if noise_offset > 0:
        cmd_parts.append(f"--noise_offset={noise_offset}")

    # P0 Fix: min_snr_gamma
    snr_gamma = config.get("min_snr_gamma", 0)
    if snr_gamma > 0:
        cmd_parts.append(f"--min_snr_gamma={snr_gamma}")

    # Caption handling
    # Note: shuffle_caption cannot be used with cache_text_encoder_outputs
    cmd_parts.extend([
        "--caption_extension=.txt",
        f"--keep_tokens={config.get('keep_tokens', 1)}",
    ])

    # Checkpoints
    save_every = config.get("save_every_n_steps", 500)
    cmd_parts.extend([
        f"--save_every_n_steps={save_every}",
        f"--save_precision={config.get('save_precision', 'bf16')}",
    ])

    # Sample generation
    sample_every = config.get("sample_every_n_steps", 250)
    cmd_parts.extend([
        f"--sample_every_n_steps={sample_every}",
        f"--sample_prompts={paths['sample_prompts']}",
        f"--sample_sampler={config.get('sample_sampler', 'euler')}",
    ])

    # Logging
    cmd_parts.extend([
        f"--logging_dir={paths['logs']}",
        f"--seed={config.get('seed', 42)}",
    ])

    # Dataloader workers (P2: performance)
    workers = config.get("max_data_loader_n_workers", 2)
    if workers > 0:
        cmd_parts.append(f"--max_data_loader_n_workers={workers}")

    # P1 Fix: Resume support
    if resume_from:
        if os.path.exists(resume_from):
            cmd_parts.append(f"--network_weights={resume_from}")
            print(f"[RESUME] Resuming from: {resume_from}", file=sys.stderr)
        else:
            print(f"WARNING: Resume checkpoint not found: {resume_from}", file=sys.stderr)

    # Build final command string with proper quoting
    # Quote paths that might have spaces
    quoted_parts = []
    for part in cmd_parts:
        if "=" in part:
            key, value = part.split("=", 1)
            # Quote values with spaces
            if " " in value:
                quoted_parts.append(f'{key}="{value}"')
            else:
                quoted_parts.append(part)
        else:
            quoted_parts.append(part)

    return " ".join(quoted_parts)


def generate_repro_info(config: dict, paths: dict, run_name: str, profile: str) -> dict:
    """Generate reproducibility information for the run."""
    repro = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "profile": profile,
        "hostname": os.uname().nodename,
    }

    # Git info
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=paths.get("workspace", ".")
        ).decode().strip()
        repro["git_commit"] = git_commit

        # Check if dirty
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            cwd=paths.get("workspace", ".")
        ).decode().strip()
        repro["git_dirty"] = len(git_status) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        repro["git_commit"] = "unknown"
        repro["git_dirty"] = None

    # GPU info
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        repro["gpu"] = gpu_info.split("\n")[0] if gpu_info else "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        repro["gpu"] = "unknown"

    # Python version
    repro["python_version"] = sys.version.split()[0]

    # Key hyperparameters
    repro["hyperparameters"] = {
        "network_dim": config.get("network_dim"),
        "network_alpha": config.get("network_alpha"),
        "max_train_steps": config.get("max_train_steps"),
        "resolution": config.get("resolution"),
        "learning_rate": config.get("learning_rate"),
        "noise_offset": config.get("noise_offset"),
        "network_dropout": config.get("network_dropout"),
        "min_snr_gamma": config.get("min_snr_gamma"),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
        "seed": config.get("seed"),
    }

    return repro


def compute_dataset_hash(data_dir: str) -> str:
    """
    Compute a stable hash of the dataset for reproducibility.

    Uses file list + sizes + mtimes for speed (not content hash).
    """
    if not os.path.isdir(data_dir):
        return "dataset_not_found"

    file_info = []
    for root, dirs, files in os.walk(data_dir):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            try:
                stat = os.stat(fpath)
                rel_path = os.path.relpath(fpath, data_dir)
                file_info.append(f"{rel_path}:{stat.st_size}:{int(stat.st_mtime)}")
            except OSError:
                continue

    if not file_info:
        return "empty_dataset"

    return hashlib.md5("\n".join(sorted(file_info)).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Build training command from TOML config (Single Source of Truth)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_train_cmd.py --profile fast
  python scripts/build_train_cmd.py --profile final --dry-run
  RANK=48 MAX_STEPS=2000 python scripts/build_train_cmd.py --profile fast
  RESUME_FROM=/path/to/checkpoint.safetensors python scripts/build_train_cmd.py --profile final
        """
    )

    parser.add_argument("--profile", "-p", choices=["fast", "final"], default="fast",
                       help="Training profile to use (default: fast)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                       help="Print command and config info, don't output for execution")
    parser.add_argument("--output-json", "-j", action="store_true",
                       help="Output parsed config as JSON instead of command")
    parser.add_argument("--allow-alpha-mismatch", action="store_true",
                       help="Allow network_alpha != network_dim (not recommended)")
    parser.add_argument("--workspace", "-w", default=None,
                       help=f"Workspace path (default: {DEFAULT_WORKSPACE})")
    parser.add_argument("--run-name", "-r", default=None,
                       help="Override run name (default: auto-generated)")
    parser.add_argument("--show-repro", action="store_true",
                       help="Output reproducibility info as JSON")

    args = parser.parse_args()

    # Determine workspace
    workspace = args.workspace or os.environ.get("WORKSPACE", DEFAULT_WORKSPACE)

    # Auto-detect workspace if running from repo
    script_dir = Path(__file__).parent.resolve()
    if script_dir.name == "scripts":
        workspace = str(script_dir.parent)

    # Load TOML config
    toml_config = load_toml_config(args.profile, workspace)
    flat_config = flatten_toml_config(toml_config)

    # Start with profile defaults
    config = PROFILE_DEFAULTS[args.profile].copy()

    # Merge TOML config (overrides defaults)
    config.update(flat_config)

    # Apply environment variable overrides
    env_overrides = get_env_overrides()
    config.update(env_overrides)

    # P0 Fix: Auto-sync alpha to match rank when rank is overridden
    # If user sets RANK but not ALPHA, alpha should follow rank
    if "network_dim" in env_overrides and "network_alpha" not in env_overrides:
        config["network_alpha"] = config["network_dim"]
    # If neither is set via env but alpha missing from TOML, use rank
    elif "network_alpha" not in env_overrides and "network_alpha" not in flat_config:
        config["network_alpha"] = config["network_dim"]

    # Auto-adjust bucket resolution when resolution is overridden
    # max_bucket_reso must be >= resolution
    resolution = config.get("resolution", 512)
    max_bucket = config.get("max_bucket_reso", 1024)
    if resolution > max_bucket:
        # Round up to nearest 64 + some margin
        new_max_bucket = ((resolution // 64) + 2) * 64
        config["max_bucket_reso"] = new_max_bucket
        print(f"[AUTO] Adjusted max_bucket_reso: {max_bucket} -> {new_max_bucket} (must be >= resolution {resolution})", file=sys.stderr)

    # Also ensure min_bucket is reasonable (at least 256, at most resolution/2)
    min_bucket = config.get("min_bucket_reso", 256)
    if min_bucket > resolution // 2:
        new_min_bucket = max(256, (resolution // 4 // 64) * 64)
        config["min_bucket_reso"] = new_min_bucket
        print(f"[AUTO] Adjusted min_bucket_reso: {min_bucket} -> {new_min_bucket}", file=sys.stderr)

    # Validate configuration
    warnings = validate_config(config, args.allow_alpha_mismatch)
    for w in warnings:
        print(w, file=sys.stderr)

    # Generate run name
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = os.environ.get("RUN_NAME", f"flux_{args.profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Build paths dict
    paths = {
        "workspace": workspace,
        "sdscripts": os.environ.get("SDSCRIPTS", DEFAULT_SDSCRIPTS),
        "model": os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
        "text_encoder": os.environ.get("TEXT_ENCODER_PATH", DEFAULT_TEXT_ENCODER_PATH),
        "data": os.environ.get("DATA_DIR", DEFAULT_DATA_DIR),
        "output": os.environ.get("OUT_DIR", DEFAULT_OUTPUT_DIR),
        "logs": os.environ.get("LOG_DIR", DEFAULT_LOG_DIR),
        "sample_prompts": os.environ.get("SAMPLE_PROMPTS", DEFAULT_SAMPLE_PROMPTS),
    }

    # Resume support
    resume_from = os.environ.get("RESUME_FROM", "")

    # FP8 base toggle
    fp8_base = os.environ.get("FP8_BASE", "0") == "1"

    # Build the command
    command = build_command(
        config=config,
        paths=paths,
        run_name=run_name,
        resume_from=resume_from if resume_from else None,
        fp8_base=fp8_base,
        profile=args.profile
    )

    # Output mode
    if args.output_json:
        output = {
            "profile": args.profile,
            "run_name": run_name,
            "config": config,
            "paths": paths,
            "command": command,
            "resume_from": resume_from if resume_from else None,
            "fp8_base": fp8_base,
        }
        print(json.dumps(output, indent=2, default=str))
    elif args.show_repro:
        repro = generate_repro_info(config, paths, run_name, args.profile)
        repro["dataset_hash"] = compute_dataset_hash(paths["data"])
        print(json.dumps(repro, indent=2))
    elif args.dry_run:
        print("=" * 80, file=sys.stderr)
        print(f"DRY RUN - Profile: {args.profile}", file=sys.stderr)
        print(f"Run Name: {run_name}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"\nKey Configuration (P0/P1 fixes applied):", file=sys.stderr)
        print(f"  network_dim:                 {config.get('network_dim')}", file=sys.stderr)
        print(f"  network_alpha:               {config.get('network_alpha')} (P0: should match dim)", file=sys.stderr)
        print(f"  noise_offset:                {config.get('noise_offset')} (P0: now wired)", file=sys.stderr)
        print(f"  network_dropout:             {config.get('network_dropout')} (P0: now wired)", file=sys.stderr)
        print(f"  min_snr_gamma:               {config.get('min_snr_gamma')} (P0: now wired)", file=sys.stderr)
        print(f"  gradient_accumulation_steps: {config.get('gradient_accumulation_steps')} (P1: now wired)", file=sys.stderr)
        print(f"  fp8_base:                    {fp8_base} (P1: toggle support)", file=sys.stderr)
        print(f"  resume_from:                 {resume_from if resume_from else 'None'} (P1: resume support)", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("\nGenerated Command:", file=sys.stderr)
        print(command)
    else:
        # Standard output: just the command for shell execution
        print(command)


if __name__ == "__main__":
    main()

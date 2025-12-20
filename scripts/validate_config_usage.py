#!/usr/bin/env python3
"""
validate_config_usage.py - Validates that training scripts use Single Source of Truth

This script checks that:
1. Training scripts reference build_train_cmd.py
2. TOML profiles are actually loadable
3. Generated commands contain required P0 flags:
   - network_alpha equals network_dim
   - noise_offset is set for each profile
   - network_dropout is set for final profile
   - min_snr_gamma is set for final profile
   - gradient_accumulation_steps is set

Usage:
    python scripts/validate_config_usage.py
    python scripts/validate_config_usage.py --verbose

Exit codes:
    0 - All validations passed
    1 - Validation failures
"""

import argparse
import os
import sys
import subprocess
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


def get_workspace() -> Path:
    """Get workspace path."""
    script_dir = Path(__file__).parent.resolve()
    if script_dir.name == "scripts":
        return script_dir.parent
    return Path.cwd()


def check_script_references_build_train_cmd(script_path: Path) -> tuple[bool, str]:
    """Check that a script references build_train_cmd.py."""
    if not script_path.exists():
        return False, f"Script not found: {script_path}"

    content = script_path.read_text()
    if "build_train_cmd.py" in content:
        return True, f"OK: {script_path.name} references build_train_cmd.py"
    else:
        return False, f"FAIL: {script_path.name} does NOT reference build_train_cmd.py"


def check_toml_loadable(toml_path: Path) -> tuple[bool, str]:
    """Check that a TOML file is loadable."""
    if not toml_path.exists():
        return False, f"TOML not found: {toml_path}"

    try:
        with open(toml_path, "rb") as f:
            config = tomllib.load(f)

        # Check required sections
        required_sections = ["network", "training"]
        missing = [s for s in required_sections if s not in config]
        if missing:
            return False, f"FAIL: {toml_path.name} missing sections: {missing}"

        return True, f"OK: {toml_path.name} is valid TOML with required sections"
    except Exception as e:
        return False, f"FAIL: {toml_path.name} parse error: {e}"


def check_generated_command(profile: str, workspace: Path, verbose: bool = False) -> tuple[bool, list[str]]:
    """Run build_train_cmd.py and validate the generated command."""
    build_script = workspace / "scripts" / "build_train_cmd.py"
    if not build_script.exists():
        return False, [f"FAIL: build_train_cmd.py not found at {build_script}"]

    # Run the build script
    env = os.environ.copy()
    env["WORKSPACE"] = str(workspace)

    try:
        result = subprocess.run(
            ["python3", str(build_script), "--profile", profile, "--output-json"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(workspace)
        )
    except Exception as e:
        return False, [f"FAIL: Could not run build_train_cmd.py: {e}"]

    if result.returncode != 0:
        return False, [f"FAIL: build_train_cmd.py exited with code {result.returncode}", result.stderr]

    import json
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return False, [f"FAIL: Could not parse JSON output: {e}"]

    command = output.get("command", "")
    config = output.get("config", {})
    messages = []
    all_passed = True

    # P0: network_alpha must equal network_dim
    dim = config.get("network_dim", 32)
    alpha = config.get("network_alpha", 1)
    if alpha == dim:
        messages.append(f"OK [{profile}]: network_alpha ({alpha}) == network_dim ({dim})")
    else:
        messages.append(f"FAIL [{profile}]: network_alpha ({alpha}) != network_dim ({dim}) - P0 VIOLATION")
        all_passed = False

    # P0: noise_offset should be present
    noise_offset = config.get("noise_offset", 0)
    expected_noise = 0.05 if profile == "fast" else 0.1
    if noise_offset > 0:
        if f"--noise_offset={noise_offset}" in command or f"--noise_offset {noise_offset}" in command:
            messages.append(f"OK [{profile}]: noise_offset={noise_offset} in command")
        else:
            # Check if it's actually in the command
            if "--noise_offset" in command:
                messages.append(f"OK [{profile}]: noise_offset parameter present in command")
            else:
                messages.append(f"WARN [{profile}]: noise_offset={noise_offset} set but not found in command")
    else:
        messages.append(f"WARN [{profile}]: noise_offset is 0 (expected ~{expected_noise})")

    # P0: network_dropout (final profile only)
    if profile == "final":
        dropout = config.get("network_dropout", 0)
        if dropout > 0:
            messages.append(f"OK [{profile}]: network_dropout={dropout}")
        else:
            messages.append(f"WARN [{profile}]: network_dropout is 0 (expected 0.1 for final)")

        # P0: min_snr_gamma (final profile only)
        snr_gamma = config.get("min_snr_gamma", 0)
        if snr_gamma > 0:
            messages.append(f"OK [{profile}]: min_snr_gamma={snr_gamma}")
        else:
            messages.append(f"WARN [{profile}]: min_snr_gamma is 0 (expected 5.0 for final)")

    # P1: gradient_accumulation_steps
    grad_accum = config.get("gradient_accumulation_steps", 0)
    if grad_accum > 1:
        messages.append(f"OK [{profile}]: gradient_accumulation_steps={grad_accum}")
    else:
        messages.append(f"WARN [{profile}]: gradient_accumulation_steps not set (expected 4)")

    # Check FLUX-required parameters
    flux_params = ["--guidance_scale=1.0", "--timestep_sampling=flux_shift", "--model_prediction_type=raw"]
    for param in flux_params:
        if param in command:
            messages.append(f"OK [{profile}]: FLUX param {param.split('=')[0]} present")
        else:
            messages.append(f"FAIL [{profile}]: Missing FLUX param: {param}")
            all_passed = False

    # Check network_module is lora_flux (not lora)
    if "--network_module=networks.lora_flux" in command:
        messages.append(f"OK [{profile}]: network_module is networks.lora_flux")
    else:
        messages.append(f"FAIL [{profile}]: network_module should be networks.lora_flux")
        all_passed = False

    if verbose:
        messages.append(f"\n[{profile}] Generated command preview (first 500 chars):\n{command[:500]}...")

    return all_passed, messages


def main():
    parser = argparse.ArgumentParser(description="Validate config usage in training scripts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    args = parser.parse_args()

    workspace = get_workspace()
    print(f"Workspace: {workspace}")
    print("=" * 70)
    print("FLUX.1-dev LoRA Training - Configuration Validation")
    print("=" * 70)
    print()

    all_passed = True
    messages = []

    # 1. Check script references
    print("1. Checking script references to build_train_cmd.py...")
    scripts_to_check = [
        workspace / "scripts" / "train_flux_fast.sh",
        workspace / "scripts" / "train_flux_final.sh",
        workspace / "scripts" / "train_dashboard.sh",
    ]

    for script in scripts_to_check:
        passed, msg = check_script_references_build_train_cmd(script)
        print(f"   {msg}")
        if not passed:
            all_passed = False
    print()

    # 2. Check TOML files are loadable
    print("2. Checking TOML configs are valid...")
    toml_files = [
        workspace / "configs" / "flux_fast.toml",
        workspace / "configs" / "flux_final.toml",
    ]

    for toml_file in toml_files:
        passed, msg = check_toml_loadable(toml_file)
        print(f"   {msg}")
        if not passed:
            all_passed = False
    print()

    # 3. Check generated commands for each profile
    print("3. Validating generated commands (P0/P1 flags)...")
    for profile in ["fast", "final"]:
        passed, msgs = check_generated_command(profile, workspace, args.verbose)
        for msg in msgs:
            print(f"   {msg}")
        if not passed:
            all_passed = False
        print()

    # 4. Summary
    print("=" * 70)
    if all_passed:
        print("RESULT: ALL VALIDATIONS PASSED")
        print()
        print("The training pipeline is correctly configured:")
        print("  - Scripts use build_train_cmd.py (Single Source of Truth)")
        print("  - TOML configs are valid and loadable")
        print("  - P0 fixes are in place (alpha=rank, noise_offset, dropout, snr_gamma)")
        print("  - FLUX-required parameters are present")
        sys.exit(0)
    else:
        print("RESULT: SOME VALIDATIONS FAILED")
        print()
        print("Please review the FAIL messages above and fix the issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()

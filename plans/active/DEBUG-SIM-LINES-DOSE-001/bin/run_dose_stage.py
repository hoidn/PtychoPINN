"""Legacy dose_experiments runner with compatibility patches.

This is a non-production shim to run the legacy /home/ollie/Documents/PtychoPINN
scripts under the frozen environment. It applies:
  1. A stub tensorflow_addons package (bin/tfa_stub)
  2. A components patch for update_params / setup_configuration
  3. A RawData patch for from_simulation with return_patches
  4. Parameter clamping to avoid KD-tree IndexError and GPU OOM:
     - neighbor_count clamped to min(default, nimages - 1)
     - nimages capped at MAX_SAFE_NIMAGES (512)
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

# GPU-safe limit to avoid OOM on RTX 3090 (24 GB).
# The simulation allocates tensors of shape [nimages, 128, 128, 4] * float64,
# which is ~10 GB at 20000 images.
MAX_SAFE_NIMAGES = 512

# Default neighbor_count if not specified elsewhere
DEFAULT_NEIGHBOR_COUNT = 5


def patch_components():
    import ptycho.workflows.components as components
    from ptycho import params as legacy_params
    from pathlib import Path

    if not hasattr(components, "update_params"):
        def update_params(mapping):
            legacy_params.cfg.update(mapping)

        components.update_params = update_params

    original_setup = components.setup_configuration

    def setup_configuration(args, yaml_path):
        if getattr(args, "train_data_file", None) is None and getattr(args, "train_data_file_path", None):
            args.train_data_file = Path(args.train_data_file_path)
        if getattr(args, "test_data_file", None) is None and getattr(args, "test_data_file_path", None):
            args.test_data_file = Path(args.test_data_file_path)
        return original_setup(args, yaml_path)

    components.setup_configuration = setup_configuration


def patch_raw_data(nimages: int, stage: str):
    """Patch RawData.from_simulation with clamped K for neighbor grouping.

    The KD-tree in group_coords queries for K+1 neighbors. If nimages < K+1,
    the returned indices will be out of bounds when accessing xcoords/ycoords.
    We clamp K to max(1, nimages - 1) to prevent IndexError.

    For simulation stage, we also force gridsize=1 because RawData.from_simulation
    explicitly asserts gridsize=1 (see raw_data.py:108).
    """
    import tensorflow as tf
    from ptycho.raw_data import RawData, calculate_relative_coords as original_calc_relative_coords, get_image_patches
    from ptycho import params as legacy_params

    # For simulation, force gridsize=1 because RawData.from_simulation requires it
    if stage == "simulation":
        original_gridsize = legacy_params.cfg.get("gridsize", 1)
        if original_gridsize != 1:
            print(f"[run_dose_stage] WARNING: Forcing gridsize=1 for simulation stage "
                  f"(was {original_gridsize}). RawData.from_simulation requires gridsize=1.")
            legacy_params.cfg["gridsize"] = 1

    # Clamp neighbor_count so KD-tree never requests more neighbors than exist
    # K+1 neighbors are queried, so K must be <= nimages - 1
    safe_k = max(1, nimages - 1)
    current_neighbor_count = legacy_params.cfg.get("neighbor_count", DEFAULT_NEIGHBOR_COUNT)
    clamped_k = min(current_neighbor_count, safe_k)

    print(f"[run_dose_stage] Clamping neighbor_count: original={current_neighbor_count}, "
          f"nimages={nimages}, safe_k={safe_k}, clamped={clamped_k}")
    legacy_params.cfg["neighbor_count"] = clamped_k

    original_from_sim = RawData.from_simulation

    def patched_calculate_relative_coords(xcoords, ycoords, K=None, C=None, nsamples=10):
        """Wrapper that enforces clamped K."""
        if K is None:
            K = clamped_k
        else:
            K = min(K, clamped_k)
        return original_calc_relative_coords(xcoords, ycoords, K=K, C=C, nsamples=nsamples)

    def from_simulation(xcoords, ycoords, probeGuess, objectGuess, scan_index=None, return_patches=False):
        raw = original_from_sim(xcoords, ycoords, probeGuess, objectGuess, scan_index)
        if not return_patches:
            return raw
        global_offsets, local_offsets, _ = patched_calculate_relative_coords(xcoords, ycoords)
        patches = get_image_patches(objectGuess, global_offsets, local_offsets)
        return raw, tf.convert_to_tensor(patches)

    RawData.from_simulation = staticmethod(from_simulation)


def extract_nimages_from_argv(argv: list[str]) -> int:
    """Parse --nimages from argv without consuming it."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nimages", type=int, default=2000)
    args, _ = parser.parse_known_args(argv)
    return args.nimages


def clamp_nimages_in_argv(argv: list[str], max_nimages: int, force_gridsize_1: bool = False,
                          force_N: int | None = None) -> tuple[list[str], int]:
    """Clamp --nimages in argv to max_nimages and optionally force gridsize=1.

    Parameters:
        argv: Command line arguments
        max_nimages: Maximum allowed number of images
        force_gridsize_1: If True, force --gridsize=1
        force_N: If set, force --N to this value (e.g., 64 to match probe size)

    Returns (new_argv, effective_nimages).
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nimages", type=int, default=2000)
    parser.add_argument("--gridsize", type=int, default=1)
    parser.add_argument("--N", type=int, default=128)
    args, remaining = parser.parse_known_args(argv)

    original_nimages = args.nimages
    effective_nimages = min(original_nimages, max_nimages)

    if original_nimages > max_nimages:
        print(f"[run_dose_stage] Clamping --nimages: original={original_nimages}, "
              f"max_safe={max_nimages}, effective={effective_nimages}")

    # Rebuild argv with clamped --nimages (and forced gridsize=1 for simulation)
    new_argv = remaining + ["--nimages", str(effective_nimages)]

    if force_gridsize_1:
        if args.gridsize != 1:
            print(f"[run_dose_stage] WARNING: Forcing --gridsize=1 for simulation "
                  f"(was {args.gridsize}). RawData.from_simulation requires gridsize=1.")
        new_argv += ["--gridsize", "1"]
    else:
        new_argv += ["--gridsize", str(args.gridsize)]

    # Force N if specified (to match probe size from NPZ)
    if force_N is not None:
        if args.N != force_N:
            print(f"[run_dose_stage] WARNING: Forcing --N={force_N} for simulation "
                  f"(was {args.N}). N must match probe size from NPZ file.")
        new_argv += ["--N", str(force_N)]
    else:
        new_argv += ["--N", str(args.N)]

    return new_argv, effective_nimages


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_dose_stage.py <simulation|training|inference> [args...]")

    stage = sys.argv[1]
    repo_root = Path(os.environ.get("DOSE_REAL_REPO", "/home/ollie/Documents/PtychoPINN"))

    script_map = {
        "simulation": repo_root / "scripts" / "simulation" / "simulation.py",
        "training": repo_root / "scripts" / "training" / "train.py",
        "inference": repo_root / "scripts" / "inference" / "inference.py",
    }

    if stage not in script_map:
        raise SystemExit(f"Unknown stage '{stage}'")

    stub_root = Path(__file__).resolve().parent / "tfa_stub"
    sys.path.insert(0, str(stub_root))
    sys.path.insert(0, str(repo_root))

    # Extract script arguments (everything after stage)
    script_args = sys.argv[2:]

    # For simulation stage, clamp nimages, force gridsize=1, and force N=64 (probe size)
    # The legacy NPZ files have 64x64 probes, so N must match
    if stage == "simulation":
        script_args, effective_nimages = clamp_nimages_in_argv(
            script_args, MAX_SAFE_NIMAGES, force_gridsize_1=True, force_N=64
        )
    else:
        # For training/inference, extract nimages if present for neighbor_count clamping
        effective_nimages = extract_nimages_from_argv(script_args)
        effective_nimages = min(effective_nimages, MAX_SAFE_NIMAGES)

    patch_components()
    patch_raw_data(effective_nimages, stage)

    script_path = script_map[stage]
    sys.argv = [str(script_path), *script_args]
    print(f"[run_dose_stage] Running: {script_path}")
    print(f"[run_dose_stage] Args: {script_args}")
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()

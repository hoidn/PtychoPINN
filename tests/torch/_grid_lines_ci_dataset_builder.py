"""Build the synthetic dataset for the main-native CI training gate.

TensorFlow imports and initializes CUDA during grid-lines simulation. Keeping
that runtime in this short-lived CPU-only process prevents it from retaining
GPU memory while the parent pytest process launches PyTorch training.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must be set before importing any workflow module that can reach TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PTYCHO_MEMOIZE_KEY_MODE", "dataset")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptycho.workflows.grid_lines_workflow import (  # noqa: E402
    GridLinesConfig,
    apply_probe_mask,
    configure_legacy_params,
    load_probe_guess,
    save_split_npz,
    scale_probe,
    simulate_grid_data,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-npz", type=Path, required=True)
    args = parser.parse_args()

    cfg = GridLinesConfig(
        N=128,
        gridsize=1,
        output_dir=args.output_dir,
        probe_npz=args.probe_npz,
        nimgs_train=1,
        nimgs_test=1,
        nphotons=1e9,
        probe_source="custom",
        probe_smoothing_sigma=0.5,
        probe_scale_mode="pad_extrapolate",
        set_phi=True,
    )

    probe = load_probe_guess(cfg.probe_npz)
    probe = scale_probe(probe, cfg.N, cfg.probe_smoothing_sigma, cfg.probe_scale_mode)
    probe = apply_probe_mask(probe, cfg.probe_mask_diameter)

    simulation = simulate_grid_data(cfg, probe)
    config = configure_legacy_params(cfg, probe)
    for split in ("train", "test"):
        simulation[split]["probeGuess"] = probe
        if simulation[split].get("probe_simulated") is None:
            raise RuntimeError(f"{split} split lacks probe_simulated")

    save_split_npz(cfg, "train", simulation["train"], config)
    save_split_npz(cfg, "test", simulation["test"], config)


if __name__ == "__main__":
    main()

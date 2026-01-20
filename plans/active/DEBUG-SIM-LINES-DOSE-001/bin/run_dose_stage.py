"""Legacy dose_experiments runner with compatibility patches."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


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


def patch_raw_data():
    import tensorflow as tf
    from ptycho.raw_data import RawData, calculate_relative_coords, get_image_patches

    original_from_sim = RawData.from_simulation

    def from_simulation(xcoords, ycoords, probeGuess, objectGuess, scan_index=None, return_patches=False):
        raw = original_from_sim(xcoords, ycoords, probeGuess, objectGuess, scan_index)
        if not return_patches:
            return raw
        global_offsets, local_offsets, _ = calculate_relative_coords(xcoords, ycoords)
        patches = get_image_patches(objectGuess, global_offsets, local_offsets)
        return raw, tf.convert_to_tensor(patches)

    RawData.from_simulation = staticmethod(from_simulation)


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

    patch_components()
    patch_raw_data()

    script_path = script_map[stage]
    sys.argv = [str(script_path), *sys.argv[2:]]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()

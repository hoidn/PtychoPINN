#!/usr/bin/env python
"""TF reference-CNN train+score driver for the N=128 torch<->TF parity campaign.

Runs the TensorFlow reference implementation (`ptycho.workflows.components`
+ `ptycho.workflows.backend_selector.run_cdi_example_with_backend`) end to
end -- train, then score on the T2 objframe metric basis shared with every
torch arm in `scripts/studies/varpro_probe_ablation_runner.py` -- so its
output is directly comparable to the torch winning-configuration runs.

Promoted from a session-scratch driver per Task 4 Step 1 of
docs/plans/2026-07-08-cnn-n128-tf-parity.md, which uses it (n=10 independent
draws) as the TF side of the behavioral-equivalence gate against the torch
reference-parity preset.

Recipe defaults mirror the E4 driver (`.superpowers/sdd/ext/etiology-e4-report.md`):
`--N 128 --gridsize 1 --nphotons 1768920 --batch_size 8 --n_groups 512`.
`nphotons=1768920` is the measured photons/img of the frozen `lines_N128`
dataset at 108 counts/px -- the count-Poisson recipe this campaign compares
against.
"""
import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts" / "studies") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "studies"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tf_reference_cnn_runner")

from ptycho.workflows.components import setup_configuration, load_data
from ptycho.workflows.backend_selector import run_cdi_example_with_backend
from ptycho.config.config import update_legacy_dict
from ptycho import params

from varpro_probe_ablation_runner import compute_objframe_metrics, place_patches_objframe
from ablation_diagnostics import canvas_rail_diagnostics


def parse_args():
    ap = argparse.ArgumentParser(description="TF reference-CNN train+score driver (parity)")
    ap.add_argument("--train_data_file", required=True)
    ap.add_argument("--test_data_file", required=True)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--gridsize", type=int, default=1)
    ap.add_argument("--nepochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_groups", type=int, default=512)
    ap.add_argument("--nphotons", type=float, default=1768920.0)
    ap.add_argument("--intensity_scale_trainable", type=int, choices=[0, 1], default=1)
    ap.add_argument("--output_dir", required=True)
    return ap.parse_args()


def main():
    cli = parse_args()
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    args = argparse.Namespace(
        train_data_file=cli.train_data_file,
        test_data_file=cli.test_data_file,
        N=cli.N,
        gridsize=cli.gridsize,
        nepochs=cli.nepochs,
        batch_size=cli.batch_size,
        n_groups=cli.n_groups,
        nphotons=cli.nphotons,
        output_dir=str(output_dir),
        backend="tensorflow",
    )
    config = setup_configuration(args, None)
    want_trainable = bool(cli.intensity_scale_trainable)
    # Override on the modern config object if the field lives there...
    if hasattr(config, "intensity_scale_trainable"):
        config = dataclasses.replace(config, intensity_scale_trainable=want_trainable)
    update_legacy_dict(params.cfg, config)
    # ...and pin the legacy key ptycho/model.py::_get_log_scale actually reads.
    params.cfg["intensity_scale.trainable"] = want_trainable
    effective = params.params()["intensity_scale.trainable"]
    logger.info(f"intensity_scale.trainable effective = {effective}")
    assert effective == want_trainable, "trainable override did not take"

    (output_dir / "invocation.json").write_text(json.dumps({
        "argv": sys.argv,
        "cli_args": vars(cli),
    }, indent=2, default=str))

    n_groups = config.n_groups
    n_subsample = config.n_groups

    train_data = load_data(
        str(config.train_data_file),
        n_images=n_groups,
        n_subsample=n_subsample,
        subsample_seed=config.subsample_seed,
    )
    test_data = load_data(str(config.test_data_file))

    logger.info(f"Starting TF training: N={config.model.N} gridsize={config.model.gridsize} "
                f"nepochs={config.nepochs} batch_size={config.batch_size} n_groups={n_groups} "
                f"nphotons={config.nphotons} trainable={want_trainable}")

    recon_amp, recon_phase, results = run_cdi_example_with_backend(
        train_data, test_data, config, do_stitching=False,
    )

    train_wall_s = time.time() - t0
    logger.info(f"Training complete in {train_wall_s:.1f}s")

    # log_scale init/final (the tf.Variable ptycho/model.py trains)
    log_scale_init = float(np.log(params.cfg["intensity_scale"]))
    log_scale_final = None
    try:
        from ptycho.model import _lazy_cache
        ls_var = _lazy_cache.get("log_scale")
        if ls_var is not None:
            log_scale_final = float(ls_var.numpy())
    except Exception as exc:  # surface, don't crash scoring
        logger.warning(f"log_scale extraction failed: {exc}")
    logger.info(f"log_scale init={log_scale_init:.4f} final={log_scale_final}")

    reconstructed_obj = np.asarray(results["reconstructed_obj"])
    test_container = results["test_container"]
    global_offsets = np.asarray(test_container.global_offsets)

    patches = reconstructed_obj[..., 0]
    coords_global = global_offsets.reshape(-1, 2)
    truth = np.load(str(config.test_data_file))["objectGuess"]

    metrics = compute_objframe_metrics(patches, coords_global, truth, patch_size=config.model.N)
    canvas, coverage_mask, n_used, n_total = place_patches_objframe(
        patches, coords_global, truth.shape, config.model.N,
    )
    canvas_diag = canvas_rail_diagnostics(canvas)
    metrics.update({
        "canvas_amp_std": canvas_diag["canvas_amp_std"],
        "canvas_phase_std": canvas_diag["canvas_phase_std"],
        "intensity_scale_trainable": want_trainable,
        "log_scale_init": log_scale_init,
        "log_scale_final": log_scale_final,
        "train_wall_s": train_wall_s,
        "nepochs": config.nepochs,
    })
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    np.savez(output_dir / "canvas.npz", canvas=canvas.astype(np.complex64), coverage_mask=coverage_mask)
    logger.info("Run complete.")


if __name__ == "__main__":
    main()

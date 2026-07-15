"""Aligned-regime hybrid_resnet ablation gate.

Executable regression coverage for docs/findings.md HYBRES-ALIGN-001: the
hybrid_resnet "training-regime artifact" resolution. The earlier ablation
looked degenerate because it trained under a misaligned regime
(torch_loss_mode='poisson', lr 1e-3, no scheduler, N=64 fly001 data) instead
of the trusted integration gate's config (unsupervised PINN, MAE amplitude
objective, lr 2e-4 + ReduceLROnPlateau, N=128 grid-lines/Run1084 data, 5
epochs). This module reruns the ablation on the integration config verbatim
(only the ablation knob under test varies) and encodes two of the aligned
rerun's decisive findings as pass/fail gates:

1. ``rectangular_scaled`` is the toxic training knob under the MAE/PINN
   objective (20x amp MAE, SSIM collapse toward 0).
2. Training-time ``training_patch_weighting`` is structurally inert at
   gridsize 1: the dispatch is gated by ``if self.object_big:``
   (``ptycho_torch/model.py:1584``) and the grid-lines runner hard-codes
   ``object_big=False`` (``scripts/studies/grid_lines_torch_runner.py:1159``),
   so the probe-weighting branch is unreachable for this pipeline.

Driver script for the full 5-arm aligned rerun (this module only exercises
the 2 arms needed for the two assertions above):
``scripts/studies/aligned_hybres_ablation_driver.sh``.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

try:
    from tests.torch.test_grid_lines_hybrid_resnet_integration import (
        _ensure_dataset,
        METRICS_BASELINE_PATH,
    )
except ImportError:
    # Fallback replicating tests/torch/test_grid_lines_hybrid_resnet_integration.py's
    # _ensure_dataset minimally, in case this repo's pytest layout cannot import
    # across tests/torch as a package.
    from ptycho.workflows.grid_lines_workflow import (
        GridLinesConfig,
        apply_probe_mask,
        configure_legacy_params,
        dataset_out_dir,
        load_probe_guess,
        save_split_npz,
        scale_probe,
        simulate_grid_data,
    )

    METRICS_BASELINE_PATH = Path("tests/fixtures/grid_lines_hybrid_resnet_metrics.json")

    def _resolve_probe_npz() -> Path:
        candidates = [
            Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"),
            Path("tmp/Run1084_recon3_postPC_shrunk_3_torch.npz"),
            Path(".artifacts/pytorch_integration_workflow/canonical/Run1084_recon3_postPC_shrunk_3_canonical.npz"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _ensure_dataset(grid_lines_scratch_root: Path) -> tuple[Path, Path]:
        cfg = GridLinesConfig(
            N=128,
            gridsize=1,
            output_dir=grid_lines_scratch_root,
            probe_npz=_resolve_probe_npz(),
            nimgs_train=1,
            nimgs_test=1,
            nphotons=1e9,
            probe_source="custom",
            probe_smoothing_sigma=0.5,
            probe_scale_mode="pad_extrapolate",
            set_phi=True,
        )
        dataset_dir = dataset_out_dir(cfg)
        train_npz = dataset_dir / "train.npz"
        test_npz = dataset_dir / "test.npz"
        if train_npz.exists() and test_npz.exists():
            return train_npz, test_npz

        probe = load_probe_guess(cfg.probe_npz)
        probe = scale_probe(probe, cfg.N, cfg.probe_smoothing_sigma, cfg.probe_scale_mode)
        probe = apply_probe_mask(probe, cfg.probe_mask_diameter)

        sim = simulate_grid_data(cfg, probe)
        config = configure_legacy_params(cfg, probe)
        sim["train"]["probeGuess"] = probe
        sim["test"]["probeGuess"] = probe

        train_npz = save_split_npz(cfg, "train", sim["train"], config)
        test_npz = save_split_npz(cfg, "test", sim["test"], config)
        return train_npz, test_npz


SCRATCH_ROOT = Path(".artifacts/integration/grid_lines_hybres_aligned_ablation")

pytestmark = [pytest.mark.grid_lines_hybrid_resnet_aligned_ablation, pytest.mark.slow]

# Flags copied verbatim from
# tests/torch/test_grid_lines_hybrid_resnet_integration.py::test_grid_lines_hybrid_resnet_metrics
# (the alignment contract this module exists to preserve).
_ALIGNED_BASE_ARGS = [
    "--architecture", "hybrid_resnet",
    "--N", "128",
    "--gridsize", "1",
    "--epochs", "5",
    "--batch-size", "16",
    "--infer-batch-size", "16",
    "--learning-rate", "2e-4",
    "--scheduler", "ReduceLROnPlateau",
    "--plateau-factor", "0.5",
    "--plateau-patience", "2",
    "--plateau-min-lr", "1e-4",
    "--plateau-threshold", "0.0",
    "--seed", "3",
    "--optimizer", "adam",
    "--weight-decay", "0.0",
    "--beta1", "0.9",
    "--beta2", "0.999",
    "--torch-loss-mode", "mae",
    "--probe-source", "custom",
    "--fno-modes", "12",
    "--fno-width", "32",
    "--fno-blocks", "4",
    "--fno-cnn-blocks", "2",
    "--torch-logger", "mlflow",
]


@pytest.fixture(scope="session")
def grid_lines_aligned_scratch_root():
    if SCRATCH_ROOT.exists():
        shutil.rmtree(SCRATCH_ROOT)
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    return SCRATCH_ROOT


def _run_arm(train_npz: Path, test_npz: Path, output_dir: Path, extra_args: list[str]) -> Path:
    cmd = [
        "python",
        "scripts/studies/grid_lines_torch_runner.py",
        "--output-dir", str(output_dir),
        "--train-npz", str(train_npz),
        "--test-npz", str(test_npz),
        *_ALIGNED_BASE_ARGS,
        *extra_args,
    ]
    subprocess.run(cmd, check=True)
    return output_dir / "runs/pinn_hybrid_resnet/metrics.json"


def test_rectangular_scaled_collapses_under_mae_pinn(grid_lines_aligned_scratch_root):
    """rectangular_scaled is the toxic training knob under the MAE/PINN objective.

    Findings: docs/findings.md HYBRES-ALIGN-001. Aligned rerun evidence showed
    rect_only amp MAE (1.5608911514282227) is ~20x the control amp MAE
    (0.0780567154288292) and SSIM collapses toward 0 (0.0034537767703777824).
    This asserts the incompatibility as executable knowledge without pinning
    the exact collapse value, so minor training-noise drift does not flip the
    gate.
    """
    train_npz, test_npz = _ensure_dataset(grid_lines_aligned_scratch_root)
    output_dir = grid_lines_aligned_scratch_root / "rect_only"

    metrics_path = _run_arm(
        train_npz,
        test_npz,
        output_dir,
        [
            "--output-mode", "real_imag",
            "--training-patch-weighting", "central_mask",
            "--physics-forward-mode", "rectangular_scaled",
        ],
    )
    assert metrics_path.exists()

    baseline = json.loads(METRICS_BASELINE_PATH.read_text())
    baseline_amp_mae = float(baseline["metrics"]["mae"][0])
    tol_amp = float(baseline["tolerances"]["mae_amp"])

    current = json.loads(metrics_path.read_text())
    rect_amp_mae = float(current["mae"][0])

    assert rect_amp_mae > 0.5
    assert rect_amp_mae > 5 * (baseline_amp_mae + tol_amp)


def test_training_patch_weighting_inert_at_gs1(grid_lines_aligned_scratch_root):
    """training_patch_weighting is a structural no-op at gridsize 1.

    Findings: docs/findings.md HYBRES-ALIGN-001. The reassembly-weighting
    dispatch in ForwardModel.forward sits under ``if self.object_big:``
    (``ptycho_torch/model.py:1584``), and
    ``scripts/studies/grid_lines_torch_runner.py:1159`` hard-codes
    ``object_big=False`` for the grid-lines pipeline, so the 'probe' branch
    of ``training_patch_weighting`` is unreachable here. Proven bit-for-bit in
    the aligned rerun (weight_only == neither, both == rect_only across all
    four metrics at full precision). This asserts exact equality (no
    tolerance) between a 'central_mask' control run and a 'probe'-weighted
    run at the same seed: if this assertion ever fails, the object_big guard
    at model.py:1584 has changed and docs/findings.md HYBRES-ALIGN-001 plus
    this module's docstrings must be revisited.
    """
    train_npz, test_npz = _ensure_dataset(grid_lines_aligned_scratch_root)

    control_metrics_path = _run_arm(
        train_npz,
        test_npz,
        grid_lines_aligned_scratch_root / "neither",
        [
            "--output-mode", "real_imag",
            "--training-patch-weighting", "central_mask",
            "--physics-forward-mode", "amplitude",
        ],
    )
    weighted_metrics_path = _run_arm(
        train_npz,
        test_npz,
        grid_lines_aligned_scratch_root / "weight_only",
        [
            "--output-mode", "real_imag",
            "--training-patch-weighting", "probe",
            "--physics-forward-mode", "amplitude",
        ],
    )

    control = json.loads(control_metrics_path.read_text())
    weighted = json.loads(weighted_metrics_path.read_text())

    assert control["mae"] == weighted["mae"]
    assert control["ssim"] == weighted["ssim"]

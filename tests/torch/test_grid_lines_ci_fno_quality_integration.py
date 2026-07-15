"""Slow visual-quality gate for main's corrected CI probe and scale chain."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_ROOT = REPO_ROOT / ".artifacts/integration/grid_lines_fno_ci_nll"
BASELINE_PATH = REPO_ROOT / "tests/fixtures/grid_lines_fno_ci_nll_5ep_metrics.json"
DATASET_BUILDER_PATH = REPO_ROOT / "tests/torch/_grid_lines_ci_dataset_builder.py"
TORCH_RUNNER_PATH = REPO_ROOT / "scripts/studies/grid_lines_torch_runner.py"
TRAIN_GROUPS = 1
TRAIN_PATTERNS = 4_489

pytestmark = [pytest.mark.slow]


def _run_repo_python(script: Path, *args: object) -> None:
    subprocess.run(
        [sys.executable, str(script), *(str(arg) for arg in args)],
        check=True,
        cwd=REPO_ROOT,
    )


def _resolve_probe_npz() -> Path:
    candidates = [
        REPO_ROOT / "datasets/Run1084_recon3_postPC_shrunk_3.npz",
        REPO_ROOT / "tmp/Run1084_recon3_postPC_shrunk_3_torch.npz",
        REPO_ROOT
        / ".artifacts/pytorch_integration_workflow/canonical/Run1084_recon3_postPC_shrunk_3_canonical.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@pytest.fixture(scope="session")
def grid_lines_fno_ci_scratch_root() -> Path:
    if SCRATCH_ROOT.exists():
        shutil.rmtree(SCRATCH_ROOT)
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    return SCRATCH_ROOT


def _ensure_dataset(scratch_root: Path) -> tuple[Path, Path]:
    train_npz = scratch_root / "datasets/N128/gs1/train.npz"
    test_npz = scratch_root / "datasets/N128/gs1/test.npz"
    if not (train_npz.exists() and test_npz.exists()):
        _run_repo_python(
            DATASET_BUILDER_PATH,
            "--output-dir",
            scratch_root,
            "--probe-npz",
            _resolve_probe_npz(),
        )

    assert train_npz.is_file()
    assert test_npz.is_file()
    with np.load(train_npz, allow_pickle=True) as train_data:
        metadata = json.loads(train_data["_metadata"].item())
        assert metadata["additional_parameters"]["nimgs_train"] == TRAIN_GROUPS
        assert train_data["coords_nominal"].shape[0] == TRAIN_PATTERNS
        assert train_data["YY_full"].shape[0] == TRAIN_GROUPS
    return train_npz, test_npz


def _load_baseline() -> dict:
    assert BASELINE_PATH.is_file(), (
        "A fresh five-epoch main/FNO CI run must establish "
        f"{BASELINE_PATH}; do not reuse the Hybrid ResNet calibration."
    )
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["contract"] == {
        "N": 128,
        "gridsize": 1,
        "epochs": 5,
        "seed": 3,
        "architecture": "fno",
        "scale_contract_version": "ci_intensity_v2",
        "measurement_domain": "count_intensity",
        "physics_forward_mode": "rectangular_scaled",
        "torch_loss_mode": "poisson",
        "ci_probe_provenance": "simulated",
        "nimgs_train": TRAIN_GROUPS,
        "n_train_patterns": TRAIN_PATTERNS,
    }
    return baseline


def _plane_aligned_phase(value: np.ndarray, border: int = 2) -> np.ndarray:
    from ptycho.evaluation import fit_and_remove_plane

    phase = np.angle(np.asarray(value))
    if border:
        phase = phase[border:-border, border:-border]
    return fit_and_remove_plane(phase)


def _render_truth_recon_error_grid(
    *,
    output_path: Path,
    ground_truth: np.ndarray,
    reconstruction: np.ndarray,
    metrics: dict,
    amplitude_ratio: float,
    provenance: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    border = 2
    gt = np.asarray(ground_truth)[border:-border, border:-border]
    pred = np.asarray(reconstruction)[border:-border, border:-border]
    gt_amp = np.abs(gt)
    pred_amp = np.abs(pred)
    amp_error = np.abs(pred_amp - gt_amp)
    gt_phase = _plane_aligned_phase(ground_truth, border=border)
    pred_phase = _plane_aligned_phase(reconstruction, border=border)
    phase_error = np.abs(np.angle(np.exp(1j * (pred_phase - gt_phase))))

    amp_error_vmax = float(np.percentile(amp_error, 99.0))
    if amp_error_vmax <= 0:
        amp_error_vmax = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), squeeze=False)
    panels = (
        (
            axes[0, 0],
            gt_amp,
            "Ground-truth amplitude",
            "viridis",
            float(np.min(gt_amp)),
            float(np.max(gt_amp)),
        ),
        (
            axes[0, 1],
            pred_amp,
            "Scaled reconstruction amplitude",
            "viridis",
            float(np.min(gt_amp)),
            float(np.max(gt_amp)),
        ),
        (axes[0, 2], amp_error, "Absolute amplitude error", "magma", 0.0, amp_error_vmax),
        (
            axes[1, 0],
            gt_phase,
            "Plane-aligned ground-truth phase",
            "twilight",
            -np.pi,
            np.pi,
        ),
        (
            axes[1, 1],
            pred_phase,
            "Plane-aligned reconstruction phase",
            "twilight",
            -np.pi,
            np.pi,
        ),
        (axes[1, 2], phase_error, "Wrapped absolute phase error", "magma", 0.0, np.pi),
    )
    for axis, image, title, cmap, vmin, vmax in panels:
        mappable = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(mappable, ax=axis, shrink=0.78)

    fig.suptitle(
        "Main FNO CI/NLL, N=128, grid=1, seed=3, epochs=5\n"
        "registered evaluation metrics: "
        f"SSIM amp/phase={metrics['ssim'][0]:.4f}/{metrics['ssim'][1]:.4f} | "
        f"MAE amp/phase={metrics['mae'][0]:.4f}/{metrics['mae'][1]:.4f}\n"
        f"raw amp mean ratio={amplitude_ratio:.4f} | probe={provenance}",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    rendered = plt.imread(output_path)
    assert rendered.ndim == 3
    assert rendered.shape[0] >= 900
    assert rendered.shape[1] >= 1400
    assert float(np.std(rendered)) > 0.05


@pytest.mark.integration
@pytest.mark.torch
@pytest.mark.deterministic
def test_grid_lines_fno_ci_nll_five_epoch_quality_and_visual(
    grid_lines_fno_ci_scratch_root: Path,
) -> None:
    """Protect the corrected CI probe gauge with a learned main-native FNO run."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the N=128 five-epoch FNO CI integration gate requires CUDA")
    if not _resolve_probe_npz().is_file():
        pytest.skip("Run1084 probe fixture is unavailable")

    train_npz, test_npz = _ensure_dataset(grid_lines_fno_ci_scratch_root)
    for split_path in (train_npz, test_npz):
        with np.load(split_path, allow_pickle=True) as split:
            assert "probe_simulated" in split.files
            assert split["probe_simulated"].shape == (128, 128)
            assert split["probe_simulated"].dtype == np.complex64

    baseline = _load_baseline()
    output_root = grid_lines_fno_ci_scratch_root / "ci_nll_5ep"
    cmd = [
        "--output-dir", str(output_root),
        "--architecture", "fno",
        "--train-npz", str(train_npz),
        "--test-npz", str(test_npz),
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
        "--grad-clip", "1.0",
        "--gradient-clip-algorithm", "norm",
        "--scale-contract-version", "ci_intensity_v2",
        "--measurement-domain", "count_intensity",
        "--physics-forward-mode", "rectangular_scaled",
        "--torch-loss-mode", "poisson",
        "--count-scale-mode", "off",
        "--rect-s1s2-refit", "dataset",
        "--output-mode", "real_imag",
        "--input-conditioning-mode", "diffraction_only",
        "--training-patch-weighting", "central_mask",
        "--probe-source", "custom",
        "--no-probe-mask",
        "--fno-modes", "12",
        "--fno-width", "32",
        "--fno-blocks", "4",
        "--fno-cnn-blocks", "2",
        "--torch-logger", "csv",
    ]
    _run_repo_python(TORCH_RUNNER_PATH, *cmd)

    run_dir = output_root / "runs/pinn_fno"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    resolved = config["torch_runner_config"]
    assert resolved["N"] == 128
    assert resolved["gridsize"] == 1
    assert resolved["epochs"] == 5
    assert resolved["seed"] == 3
    assert resolved["architecture"] == "fno"
    assert resolved["scale_contract_version"] == "ci_intensity_v2"
    assert resolved["measurement_domain"] == "count_intensity"
    assert resolved["physics_forward_mode"] == "rectangular_scaled"
    assert resolved["torch_loss_mode"] == "poisson"
    assert resolved["count_scale_mode"] == "off"
    assert resolved["gradient_clip_val"] == 1.0
    assert config["ci_probe_provenance"] == "simulated"
    assert config["rect_s1s2_refit"]["mode"] == "dataset"
    assert config["scaled_recon"]["source"] == "refit"

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    train_loss = np.asarray(history["train_loss"], dtype=np.float64)
    val_loss = np.asarray(history["val_loss"], dtype=np.float64)
    assert train_loss.shape == (5,)
    assert val_loss.shape in {(5,), (6,)}
    assert np.isfinite(train_loss).all()
    assert np.isfinite(val_loss).all()
    assert train_loss[-1] <= baseline["loss"]["train"] + baseline["tolerances"]["train_loss"]
    assert val_loss[-1] <= (
        baseline["loss"]["validation"] + baseline["tolerances"]["validation_loss"]
    )

    metrics_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "metrics_scaled" in metrics_payload
    metrics = metrics_payload["metrics_scaled"]
    for key in ("mae", "ssim"):
        current_amp = float(metrics[key][0])
        current_phase = float(metrics[key][1])
        assert np.isfinite([current_amp, current_phase]).all()
        baseline_amp = float(baseline["metrics"][key][0])
        baseline_phase = float(baseline["metrics"][key][1])
        if key == "mae":
            assert current_amp <= baseline_amp + baseline["tolerances"]["mae_amp"]
            assert current_phase <= baseline_phase + baseline["tolerances"]["mae_phase"]
        else:
            assert current_amp >= baseline_amp - baseline["tolerances"]["ssim_amp"]
            assert current_phase >= baseline_phase - baseline["tolerances"]["ssim_phase"]

    recon_path = output_root / "recons/pinn_fno/recon.npz"
    gt_path = output_root / "recons/gt/recon.npz"
    with np.load(recon_path) as reconstruction_payload:
        assert "YY_pred_scaled" in reconstruction_payload.files
        reconstruction = np.asarray(reconstruction_payload["YY_pred_scaled"])
    with np.load(gt_path) as ground_truth_payload:
        ground_truth = np.asarray(ground_truth_payload["YY_pred"])
    assert reconstruction.shape == ground_truth.shape
    assert np.isfinite(reconstruction.real).all()
    assert np.isfinite(reconstruction.imag).all()

    border = 2
    pred_amp = np.abs(reconstruction[border:-border, border:-border])
    gt_amp = np.abs(ground_truth[border:-border, border:-border])
    amplitude_ratio = float(pred_amp.mean() / gt_amp.mean())
    with np.load(test_npz) as test_data:
        probe_ratio = test_data["probe_simulated"] / test_data["probeGuess"]
    probe_gain = float(np.median(probe_ratio.real))
    np.testing.assert_allclose(probe_ratio.imag, 0.0, atol=2e-5)
    np.testing.assert_allclose(probe_ratio.real, probe_gain, rtol=2e-5, atol=2e-5)
    assert amplitude_ratio > 0.0
    assert abs(np.log(amplitude_ratio)) < 0.5 * abs(np.log(probe_gain))
    assert amplitude_ratio == pytest.approx(
        baseline["raw_amplitude_mean_ratio"],
        abs=baseline["tolerances"]["raw_amplitude_mean_ratio"],
    )

    comparison_path = output_root / "visuals/ci_nll_full_comparison.png"
    _render_truth_recon_error_grid(
        output_path=comparison_path,
        ground_truth=ground_truth,
        reconstruction=reconstruction,
        metrics=metrics,
        amplitude_ratio=amplitude_ratio,
        provenance=config["ci_probe_provenance"],
    )
    assert comparison_path.is_file()
    assert comparison_path.stat().st_size > 50_000

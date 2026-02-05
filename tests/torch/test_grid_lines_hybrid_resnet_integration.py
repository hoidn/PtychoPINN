import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ptycho.workflows.grid_lines_workflow import (
    GridLinesConfig,
    apply_probe_mask,
    configure_legacy_params,
    load_ideal_disk_probe,
    save_split_npz,
    scale_probe,
    simulate_grid_data,
)


SCRATCH_ROOT = Path(".artifacts/integration/grid_lines_hybrid_resnet")
DATASET_STATS_PATH = Path("tests/fixtures/grid_lines_hybrid_resnet_dataset_stats.json")
METRICS_BASELINE_PATH = Path("tests/fixtures/grid_lines_hybrid_resnet_metrics.json")

pytestmark = pytest.mark.grid_lines_hybrid_resnet_integration


@pytest.fixture(scope="session")
def grid_lines_scratch_root():
    if SCRATCH_ROOT.exists():
        shutil.rmtree(SCRATCH_ROOT)
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    return SCRATCH_ROOT


def _ensure_dataset(grid_lines_scratch_root: Path) -> tuple[Path, Path]:
    train_npz = grid_lines_scratch_root / "datasets/N64/gs1/train.npz"
    test_npz = grid_lines_scratch_root / "datasets/N64/gs1/test.npz"
    if train_npz.exists() and test_npz.exists():
        return train_npz, test_npz

    cfg = GridLinesConfig(
        N=64,
        gridsize=1,
        output_dir=grid_lines_scratch_root,
        probe_npz=grid_lines_scratch_root / "probe_unused.npz",
        nimgs_train=2,
        nimgs_test=2,
        nphotons=1e9,
        probe_source="ideal_disk",
        probe_smoothing_sigma=0.5,
        probe_scale_mode="pad_extrapolate",
        set_phi=True,
    )

    probe = load_ideal_disk_probe(cfg.N)
    probe = scale_probe(probe, cfg.N, cfg.probe_smoothing_sigma, cfg.probe_scale_mode)
    probe = apply_probe_mask(probe, cfg.probe_mask_diameter)

    sim = simulate_grid_data(cfg, probe)
    config = configure_legacy_params(cfg, probe)
    sim["train"]["probeGuess"] = probe
    sim["test"]["probeGuess"] = probe

    train_npz = save_split_npz(cfg, "train", sim["train"], config)
    test_npz = save_split_npz(cfg, "test", sim["test"], config)
    return train_npz, test_npz


@pytest.mark.integration
def test_grid_lines_scratch_root_created(grid_lines_scratch_root):
    assert grid_lines_scratch_root.exists()


@pytest.mark.integration
def test_grid_lines_dataset_stats(grid_lines_scratch_root):
    train_npz, _ = _ensure_dataset(grid_lines_scratch_root)

    with np.load(train_npz, allow_pickle=True) as data:
        diff = data["diffraction"].astype(np.float32)
        y_i = data["Y_I"].astype(np.float32)
        y_phi = data["Y_phi"].astype(np.float32)

    stats = {
        "diff_mean": float(diff.mean()),
        "diff_std": float(diff.std()),
        "yi_mean": float(y_i.mean()),
        "yi_std": float(y_i.std()),
        "yphi_mean": float(y_phi.mean()),
        "yphi_std": float(y_phi.std()),
    }

    baseline = json.loads(DATASET_STATS_PATH.read_text())
    tol = baseline["tolerances"]
    for key, value in stats.items():
        assert abs(value - baseline["stats"][key]) <= tol[key]


@pytest.mark.integration
def test_grid_lines_hybrid_resnet_metrics(grid_lines_scratch_root):
    train_npz, test_npz = _ensure_dataset(grid_lines_scratch_root)

    cmd = [
        "python",
        "scripts/studies/grid_lines_torch_runner.py",
        "--output-dir", str(grid_lines_scratch_root),
        "--architecture", "hybrid_resnet",
        "--train-npz", str(train_npz),
        "--test-npz", str(test_npz),
        "--N", "64",
        "--gridsize", "1",
        "--epochs", "10",
        "--batch-size", "16",
        "--infer-batch-size", "16",
        "--learning-rate", "2e-4",
        "--scheduler", "ReduceLROnPlateau",
        "--plateau-factor", "0.5",
        "--plateau-patience", "2",
        "--plateau-min-lr", "1e-4",
        "--plateau-threshold", "0.0",
        "--seed", "2147483645",
        "--optimizer", "adam",
        "--weight-decay", "0.0",
        "--beta1", "0.9",
        "--beta2", "0.999",
        "--torch-loss-mode", "mae",
        "--output-mode", "real_imag",
        "--probe-source", "custom",
        "--fno-modes", "12",
        "--fno-width", "32",
        "--fno-blocks", "4",
        "--fno-cnn-blocks", "2",
        "--torch-logger", "none",
    ]

    subprocess.run(cmd, check=True)

    metrics_path = grid_lines_scratch_root / "runs/pinn_hybrid_resnet/metrics.json"
    visuals_dir = grid_lines_scratch_root / "visuals"
    assert metrics_path.exists()
    assert (visuals_dir / "amp_phase_pinn_hybrid_resnet.png").exists()

    baseline = json.loads(METRICS_BASELINE_PATH.read_text())
    current = json.loads(metrics_path.read_text())

    for key in ("mae", "ssim"):
        cur_amp = float(current[key][0])
        cur_phase = float(current[key][1])
        base_amp = float(baseline["metrics"][key][0])
        base_phase = float(baseline["metrics"][key][1])
        tol_amp = baseline["tolerances"][f"{key}_amp"]
        tol_phase = baseline["tolerances"][f"{key}_phase"]
        if key == "mae":
            assert cur_amp <= base_amp + tol_amp
            assert cur_phase <= base_phase + tol_phase
        else:
            assert cur_amp >= base_amp - tol_amp
            assert cur_phase >= base_phase - tol_phase

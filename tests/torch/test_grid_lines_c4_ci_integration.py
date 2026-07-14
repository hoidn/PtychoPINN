"""Slow project-level gates for the corrected C4 CI lines path.

These tests intentionally reuse the checked Task 30 manifest and materialized
dataset. Lower-level tests own individual loader, multimode, and decoder shape
contracts; this module only protects the integrated training and inference
behavior that exposed the C4 collapse.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_SPEC = REPO_ROOT / "scripts/studies/specs/grid_lines_ci_convergence.toml"
DATASET_ROOT = REPO_ROOT / ".artifacts/ci_compatibility/datasets_v3"
TRAIN_NPZ = DATASET_ROOT / "lines_ci_3p5m_train.npz"
TEST_NPZ = DATASET_ROOT / "lines_ci_3p5m_test.npz"
DRIVER = REPO_ROOT / "scripts/studies/torch_ablation_driver.py"

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.torch]


def _require_gpu_and_dataset() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("C4 CI integration gates require CUDA")
    missing = [path for path in (TRAIN_NPZ, TEST_NPZ) if not path.is_file()]
    if missing:
        pytest.skip(
            "materialized C4 CI lines fixture is absent: "
            + ", ".join(str(path) for path in missing)
        )


def _run_arm(output_root: Path, *, architecture: str, epochs: int) -> Path:
    _require_gpu_and_dataset()
    subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--spec",
            str(STUDY_SPEC),
            "--only",
            f"architecture={architecture},physics_profile=ci_nll",
            "--epochs",
            str(epochs),
            "--output-root",
            str(output_root),
            "--fail-fast",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    metrics_paths = list((output_root / "runs").glob("*/attempt-*/metrics.json"))
    assert len(metrics_paths) == 1
    return metrics_paths[0].parent


def _metric_records(run_dir: Path) -> dict[str, float]:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        record["path"]: float(record["value"])
        for record in payload["records"]
        if record.get("value") is not None
    }


def _assert_healthy_c4_ci_run(run_dir: Path, *, architecture: str) -> dict[str, float]:
    resolved = json.loads(
        (run_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    assert resolved["data"]["C"] == 4
    assert resolved["data"]["grid_size"] == [2, 2]
    assert resolved["model"]["architecture"] == architecture
    assert resolved["model"]["object_big"] is True
    assert resolved["model"]["probe_big"] is True
    assert resolved["model"]["physics_forward_mode"] == "rectangular_scaled"
    assert resolved["model"]["training_patch_weighting"] == "probe"
    assert resolved["training"]["torch_loss_mode"] == "poisson"
    assert resolved["inference"]["patch_weighting"] == "probe"
    assert resolved["inference"]["varpro_scaling"] is True
    if architecture == "cnn":
        assert resolved["model"]["decoder_last_amp_channels"] == 4

    records = _metric_records(run_dir)
    assert records["stability.reload_allclose"] == 1.0
    assert records["stability.finite"] == 1.0
    assert records["stability.loss_all_finite"] == 1.0
    assert records["stability.patches_accepted"] > 0
    assert records["stability.amp_variance"] > 1e-6
    assert records["stability.phase_variance"] > 1e-6
    assert records["stability.real_head_saturation_fraction"] < 0.10
    assert records["stability.imag_head_saturation_fraction"] < 0.10
    assert np.isfinite(records["measurement_consistency.relative_l2_intensity_error"])
    assert np.isfinite(records["measurement_consistency.varpro.s1"])
    assert np.isfinite(records["measurement_consistency.varpro.s2"])
    return records


def test_cnn_c4_ci_lines_quality_does_not_regress(tmp_path: Path) -> None:
    run_dir = _run_arm(
        tmp_path / "cnn_c4_ci_quality",
        architecture="cnn",
        epochs=20,
    )

    records = _assert_healthy_c4_ci_run(run_dir, architecture="cnn")
    assert records["truth_quality.post_varpro.amp_ssim"] >= 0.60
    assert records["truth_quality.post_varpro.phase_ssim"] >= 0.94
    assert records["measurement_consistency.relative_l2_intensity_error"] < 0.50

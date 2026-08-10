"""Tests for the study-level comparison table renderer (stage-14 contract).

The renderer must build the table from per-arm reconstruction.npz data,
enforce the six-column structure, share color limits across rows, and fail
loudly when phase panels carry no structure (the black-phase regression).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "studies" / "render_study_comparison.py"


def _write_arm(
    arm_dir: Path,
    *,
    phase_structure: float = 0.6,
    amplitude_ssim: float = 0.91,
    phase_ssim: float = 0.87,
) -> None:
    """Create one minimal arm: 40x40 truth object, 20x20 reconstruction."""
    seed = sum(arm_dir.name.encode()) % (2**32)
    rng = np.random.default_rng(seed)
    structure = rng.standard_normal((40, 40))
    amplitude = 1.0 + 0.2 * structure
    y, x = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
    phase = 0.01 * x + 0.02 * y + phase_structure * np.tanh(structure)
    truth = (amplitude * np.exp(1j * phase)).astype(np.complex64)

    prediction = truth[10:30, 10:30] * 2.0  # scale exercises mean-matching
    noise = 0.01 if phase_structure else 0.0  # keep degenerate case degenerate
    prediction = (prediction + noise * rng.standard_normal((20, 20))).astype(
        np.complex64
    )

    datasets = arm_dir / "datasets"
    reconstruction = arm_dir / "reconstruction"
    datasets.mkdir(parents=True)
    reconstruction.mkdir(parents=True)
    np.savez(datasets / "test.npz", objectGuess=truth)
    np.savez(reconstruction / "reconstruction.npz", complex_canvas=prediction)
    (reconstruction / "metrics.json").write_text(
        json.dumps(
            {
                "amplitude_ssim": amplitude_ssim,
                "phase_ssim": phase_ssim,
                "alignment": {
                    "method": "truth_origin_slice_v1",
                    "truth_origin": [10, 10],
                    "aligned_shape": [20, 20],
                    "metric_crop_border": 2,
                },
            }
        )
    )


def _run(*args: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True
    )


@pytest.mark.study
def test_renders_six_column_table_with_shared_limits(tmp_path):
    _write_arm(tmp_path / "lines__cnn__legacy_mae")
    _write_arm(tmp_path / "deadleaves__hybrid_resnet__ci_nll")
    out = tmp_path / "table.png"
    proc = _run(str(tmp_path), "--output", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.is_file() and out.stat().st_size > 0
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["rows"] == 2
    assert report["columns"] == 6
    # lines family sorts before deadleaves in the canonical matrix order
    assert report["arms"][0] == "lines__cnn__legacy_mae"


@pytest.mark.study
def test_legacy_matrix_results_layout_is_discovered(tmp_path):
    _write_arm(tmp_path / "matrix_results" / "lines__hybrid_resnet__legacy_nll")
    out = tmp_path / "table.png"
    proc = _run(str(tmp_path), "--output", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["rows"] == 1


@pytest.mark.study
def test_structureless_phase_fails_variance_floor(tmp_path):
    _write_arm(tmp_path / "lines__cnn__legacy_mae", phase_structure=0.0)
    proc = _run(str(tmp_path), "--output", str(tmp_path / "table.png"))
    assert proc.returncode != 0
    assert "phase variance" in proc.stderr

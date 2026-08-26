"""Public one-epoch C4/count-intensity workflow smokes."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from ptycho.workflows.synthetic_pipeline import (
    DIAGNOSTICS_SCHEMA,
    METRIC_CONTRACT_VERSION,
    STAGE_MANIFEST_SCHEMA,
    STAGE_ORDER,
)
from ptycho_torch.reassembly_diagnostics import FittedCountMetrics
from ptycho_torch.rect_s1s2_initialization import (
    RECT_S1S2_DOSE_CLOSURE_PATTERNS,
    RectS1S2InitializationRecord,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = REPO_ROOT / "datasets/Run1084_recon3_postPC_shrunk_3.npz"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.torch,
    pytest.mark.deterministic,
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _nonempty(path: Path) -> Path:
    assert path.is_file() and path.stat().st_size > 0, path
    return path


def _require_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("public C4/CI integration requires CUDA")


@pytest.mark.parametrize("architecture", ["cnn"])
def test_public_synthetic_c4_ci_one_epoch_smoke(
    tmp_path: Path,
    architecture: str,
) -> None:
    _require_cuda()
    root = tmp_path / f"{architecture}-c4-ci"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.simulation.synthetic_pipeline",
            "--profile",
            "cnn-lines-ci",
            "--output-root",
            str(root),
            "--architecture",
            architecture,
            "--gridsize",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "16",
            "--seed",
            "3",
            "--probe-source",
            "custom",
            "--probe-path",
            str(PROBE_PATH),
            "--probe-transform",
            "pad_extrapolate:128|smooth:0.5",
            "--train-patterns",
            "256",
            "--test-patterns",
            "64",
            "--train-raw-selection",
            "256",
            "--training-groups",
            "256",
            "--validation-groups",
            "64",
            "--neighbor-count",
            "4",
            "--neighbor-pool-size",
            "4",
            "--groups-per-center",
            "1",
            "--accelerator",
            "cuda",
            "--devices",
            "1",
            "--precision",
            "32-true",
            "--workers",
            "0",
            "--logger",
            "csv",
            "--deterministic",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    resolved = _load_json(_nonempty(root / "resolved_workflow.json"))
    assert resolved["profile"] == "cnn-lines-ci"
    assert resolved["data"]["gridsize"] == 2
    assert resolved["data"]["scale_contract_version"] == "ci_intensity_v2"
    assert resolved["data"]["measurement_domain"] == "count_intensity"
    assert resolved["model"]["architecture"] == architecture
    assert resolved["model"]["physics_forward_mode"] == "rectangular_scaled"
    assert resolved["model"]["cnn_output_mode"] == "real_imag"
    assert resolved["model"]["rect_s1s2_init"] == "dose_closure"
    assert resolved["training"]["epochs"] == 1
    assert resolved["training"]["torch_loss_mode"] == "poisson"

    stage_manifest = _load_json(_nonempty(root / "stage_manifest.json"))
    assert stage_manifest["schema_version"] == STAGE_MANIFEST_SCHEMA
    assert stage_manifest["metric_contract_version"] == METRIC_CONTRACT_VERSION
    assert list(stage_manifest["stages"]) == list(STAGE_ORDER)
    assert all(
        stage["status"] == "complete"
        for stage in stage_manifest["stages"].values()
    )

    for path in (
        root / "datasets/manifest.json",
        root / "training/wts.h5.zip",
        root / "training/training_summary.json",
        root / "reconstruction/reconstruction.npz",
        root / "reconstruction/diagnostics.json",
        root / "reconstruction/metrics.json",
        root / "reconstruction/comparison.png",
    ):
        _nonempty(path)

    initialization = RectS1S2InitializationRecord.from_mapping(
        _load_json(root / "training/training_summary.json")
    )
    assert initialization.mode == "dose_closure"
    assert initialization.sampled_patterns == RECT_S1S2_DOSE_CLOSURE_PATTERNS

    diagnostics = _load_json(root / "reconstruction/diagnostics.json")
    assert diagnostics["schema_version"] == DIAGNOSTICS_SCHEMA
    validity = diagnostics["metric_validity"]
    assert validity["valid"] is True
    assert validity["channel_groups"] == {
        "group_count": 64,
        "channel_count": 4,
        "all_groups_distinct": True,
    }
    count_metrics = FittedCountMetrics(**validity["count_diagnostics"])
    assert count_metrics.n_samples == 256

    metrics = _load_json(root / "reconstruction/metrics.json")
    assert metrics["metric_contract_version"] == METRIC_CONTRACT_VERSION
    for name in (
        "amplitude_ssim",
        "phase_ssim",
        "absolute_amp_mae",
        "phase_wrapped_mae",
    ):
        assert math.isfinite(float(metrics[name]))

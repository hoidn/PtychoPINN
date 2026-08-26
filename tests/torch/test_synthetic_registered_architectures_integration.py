"""Public one-epoch GS1 smokes for migrated generator paths."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from ptycho.workflows.synthetic_pipeline import (
    METRIC_CONTRACT_VERSION,
    STAGE_MANIFEST_SCHEMA,
    STAGE_ORDER,
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


@pytest.mark.parametrize(
    ("architecture", "profile", "measurement_domain", "loss_mode"),
    [
        pytest.param(
            "ffno",
            "hybrid-resnet-lines-ci",
            "count_intensity",
            "poisson",
            id="ffno-ci",
        ),
    ],
)
def test_public_synthetic_registered_architecture_gs1_smoke(
    tmp_path: Path,
    architecture: str,
    profile: str,
    measurement_domain: str,
    loss_mode: str,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("public generator integration requires CUDA")

    root = tmp_path / architecture
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.simulation.synthetic_pipeline",
            "--profile",
            profile,
            "--output-root",
            str(root),
            "--architecture",
            architecture,
            "--gridsize",
            "1",
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
            "1",
            "--neighbor-pool-size",
            "1",
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
    assert resolved["model"]["architecture"] == architecture
    assert resolved["data"]["gridsize"] == 1
    assert resolved["data"]["measurement_domain"] == measurement_domain
    assert resolved["training"]["epochs"] == 1
    assert resolved["training"]["torch_loss_mode"] == loss_mode

    stage_manifest = _load_json(_nonempty(root / "stage_manifest.json"))
    assert stage_manifest["schema_version"] == STAGE_MANIFEST_SCHEMA
    assert stage_manifest["metric_contract_version"] == METRIC_CONTRACT_VERSION
    assert list(stage_manifest["stages"]) == list(STAGE_ORDER)
    assert all(
        stage["status"] == "complete"
        for stage in stage_manifest["stages"].values()
    )

    for path in (
        root / "training/wts.h5.zip",
        root / "training/training_summary.json",
        root / "training/effective_runtime.json",
        root / "reconstruction/reconstruction.npz",
        root / "reconstruction/diagnostics.json",
        root / "reconstruction/metrics.json",
        root / "reconstruction/comparison.png",
    ):
        _nonempty(path)

    diagnostics = _load_json(root / "reconstruction/diagnostics.json")
    assert diagnostics["metric_validity"]["valid"] is True
    assert diagnostics["metric_validity"]["channel_groups"] == {
        "group_count": 64,
        "channel_count": 1,
        "all_groups_distinct": True,
    }

    metrics = _load_json(root / "reconstruction/metrics.json")
    assert metrics["metric_contract_version"] == METRIC_CONTRACT_VERSION
    for name in (
        "amplitude_ssim",
        "phase_ssim",
        "absolute_amp_mae",
        "phase_wrapped_mae",
    ):
        assert math.isfinite(float(metrics[name]))

"""Five-epoch FFNO/GS1/count-intensity quality gate on the public synthetic pipeline.

Migrated from the retired ``grid_lines_torch_runner`` quality test
(``tests/torch/test_grid_lines_ci_ffno_quality_integration.py``, deleted at
``ce7d60c44``). Same protections — pinned dataset identity, loss ceilings,
frozen metric baseline, visual artifact — on the modern
``scripts/simulation/synthetic_pipeline.py`` path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from ptycho.workflows.synthetic_pipeline import (
    METRIC_CONTRACT_VERSION,
    STAGE_MANIFEST_SCHEMA,
    STAGE_ORDER,
)
from ptycho_torch.rect_s1s2_initialization import (
    RECT_S1S2_DOSE_CLOSURE_PATTERNS,
    RectS1S2InitializationRecord,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = REPO_ROOT / "datasets/Run1084_recon3_postPC_shrunk_3.npz"
BASELINE_PATH = REPO_ROOT / "tests/fixtures/synthetic_ffno_gs1_ci_5ep_metrics.json"
TRAIN_PATTERNS = 4_489  # 67**2; fixed_pitch_raster requires a perfect square
TEST_PATTERNS = 729  # 27**2

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


def _load_baseline() -> dict:
    assert BASELINE_PATH.is_file(), (
        "A fresh five-epoch synthetic-pipeline FFNO run must establish "
        f"{BASELINE_PATH}; do not reuse the retired grid_lines_torch_runner "
        "baseline (tests/fixtures/grid_lines_ffno_ci_nll_5ep_metrics.json)."
    )
    baseline = _load_json(BASELINE_PATH)
    contract = baseline["contract"]
    assert contract["N"] == 128
    assert contract["gridsize"] == 1
    assert contract["epochs"] == 5
    assert contract["seed"] == 3
    assert contract["architecture"] == "ffno"
    assert contract["scale_contract_version"] == "ci_intensity_v2"
    assert contract["measurement_domain"] == "count_intensity"
    assert contract["physics_forward_mode"] == "rectangular_scaled"
    assert contract["torch_loss_mode"] == "poisson"
    assert contract["rect_s1s2_init"] == "dose_closure"
    assert contract["train_patterns"] == TRAIN_PATTERNS
    assert contract["test_patterns"] == TEST_PATTERNS
    return baseline


def _serving_checkpoint_losses(
    training_root: Path,
    *,
    epochs: int,
) -> tuple[float, float]:
    """Read final train health and the validation score of bundled weights."""
    from ptycho_torch.training_history import read_metrics_series

    csv_candidates = sorted(training_root.glob("lightning_logs/version_*/metrics.csv"))
    assert len(csv_candidates) == 1, csv_candidates
    series = read_metrics_series(csv_candidates[0])
    values = {}
    for name in ("poisson_train_loss_epoch", "poisson_val_loss"):
        metric = series[name]
        assert metric["epoch"] == list(range(epochs)), metric["epoch"]
        assert all(
            math.isfinite(value) and value >= 0.0
            for value in metric["value"]
        )
        values[name] = metric["value"]

    bundle_path = _nonempty(training_root / "wts.h5.zip")
    with zipfile.ZipFile(bundle_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    selection = manifest.get("checkpoint_selection")
    assert isinstance(selection, dict), "bundle manifest checkpoint_selection is missing"

    root_record = _load_json(
        _nonempty(training_root / "checkpoint_selection.json")
    )
    root_record.pop("selection_token", None)
    assert root_record == selection
    assert selection["schema_version"] == "serving-checkpoint-selection-v1"
    assert selection["policy"] == "best"
    assert selection["weights_source"] == "checkpoint"
    assert selection["monitor"] == "poisson_val_loss"
    assert selection["mode"] == "min"

    selected_epoch = selection["selected_epoch"]
    assert (
        not isinstance(selected_epoch, bool)
        and isinstance(selected_epoch, int)
        and 0 <= selected_epoch < epochs
    )
    selected_score = selection["selected_score"]
    assert (
        not isinstance(selected_score, bool)
        and isinstance(selected_score, (int, float))
        and math.isfinite(selected_score)
        and selected_score >= 0.0
    )
    validation_values = values["poisson_val_loss"]
    assert selected_score == validation_values[selected_epoch]
    assert selected_score == min(validation_values)

    selected_path = selection["selected_path"]
    selected_sha256 = selection["selected_sha256"]
    assert isinstance(selected_path, str)
    assert isinstance(selected_sha256, str)
    checkpoint_path = (training_root / selected_path).resolve()
    assert (training_root / "checkpoints").resolve() in checkpoint_path.parents
    assert checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0
    with checkpoint_path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == selected_sha256

    return values["poisson_train_loss_epoch"][-1], float(selected_score)


def test_public_synthetic_ffno_gs1_ci_five_epoch_quality(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the five-epoch FFNO CI quality gate requires CUDA")

    baseline = _load_baseline()

    root = tmp_path / "ffno-gs1-ci-5ep"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.simulation.synthetic_pipeline",
            "--profile",
            "synthetic-lines",
            "--output-root",
            str(root),
            "--architecture",
            "ffno",
            "--N",
            "128",
            "--gridsize",
            "1",
            "--epochs",
            "5",
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
            "--scale-contract-version",
            "ci_intensity_v2",
            "--measurement-domain",
            "count_intensity",
            "--physics-forward-mode",
            "rectangular_scaled",
            "--generator-output-mode",
            "real_imag",
            "--torch-loss-mode",
            "poisson",
            "--rect-s1s2-init",
            "dose_closure",
            "--gradient-clip-val",
            "1.0",
            "--train-patterns",
            str(TRAIN_PATTERNS),
            "--test-patterns",
            str(TEST_PATTERNS),
            "--train-raw-selection",
            str(TRAIN_PATTERNS),
            "--training-groups",
            str(TRAIN_PATTERNS),
            "--validation-groups",
            str(TEST_PATTERNS),
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
    assert resolved["profile"] == "synthetic-lines"
    assert resolved["model"]["architecture"] == "ffno"
    assert resolved["data"]["N"] == 128
    assert resolved["data"]["gridsize"] == 1
    assert resolved["data"]["scale_contract_version"] == "ci_intensity_v2"
    assert resolved["data"]["measurement_domain"] == "count_intensity"
    assert resolved["model"]["physics_forward_mode"] == "rectangular_scaled"
    assert resolved["model"]["generator_output_mode"] == "real_imag"
    assert resolved["model"]["rect_s1s2_init"] == "dose_closure"
    assert resolved["training"]["epochs"] == 5
    assert resolved["training"]["torch_loss_mode"] == "poisson"

    stage_manifest = _load_json(_nonempty(root / "stage_manifest.json"))
    assert stage_manifest["schema_version"] == STAGE_MANIFEST_SCHEMA
    assert stage_manifest["metric_contract_version"] == METRIC_CONTRACT_VERSION
    assert list(stage_manifest["stages"]) == list(STAGE_ORDER)
    assert all(
        stage["status"] == "complete"
        for stage in stage_manifest["stages"].values()
    )

    # Dataset identity: the simulate stage must reproduce the pinned splits.
    from ptycho.simulation.identity import array_sha256

    for split, expected_key in (
        ("train", "train_diffraction_sha256"),
        ("test", "test_diffraction_sha256"),
    ):
        with np.load(_nonempty(root / f"datasets/{split}.npz")) as data:
            digest = array_sha256(data["diff3d"].astype(np.float32))
        assert digest == baseline["contract"][expected_key], split

    initialization = RectS1S2InitializationRecord.from_mapping(
        _load_json(_nonempty(root / "training/training_summary.json"))
    )
    assert initialization.mode == "dose_closure"
    assert initialization.sampled_patterns == RECT_S1S2_DOSE_CLOSURE_PATTERNS

    diagnostics = _load_json(_nonempty(root / "reconstruction/diagnostics.json"))
    assert diagnostics["metric_validity"]["valid"] is True
    assert diagnostics["metric_validity"]["channel_groups"] == {
        "group_count": TEST_PATTERNS,
        "channel_count": 1,
        "all_groups_distinct": True,
    }

    # Loss ceilings against the frozen baseline.
    tolerances = baseline["tolerances"]
    train_loss, val_loss = _serving_checkpoint_losses(
        root / "training",
        epochs=baseline["contract"]["epochs"],
    )
    assert train_loss <= baseline["loss"]["train"] + tolerances["train_loss"]
    assert val_loss <= baseline["loss"]["validation"] + tolerances["validation_loss"]

    # Quality metrics: SSIMs are floors, MAEs are ceilings.
    metrics = _load_json(_nonempty(root / "reconstruction/metrics.json"))
    assert metrics["metric_contract_version"] == METRIC_CONTRACT_VERSION
    for name in (
        "amplitude_ssim",
        "phase_ssim",
        "absolute_amp_mae",
        "phase_wrapped_mae",
    ):
        assert math.isfinite(float(metrics[name])), name
    frozen = baseline["metrics"]
    assert metrics["amplitude_ssim"] >= frozen["amplitude_ssim"] - tolerances["ssim_amp"]
    assert metrics["phase_ssim"] >= frozen["phase_ssim"] - tolerances["ssim_phase"]
    assert metrics["absolute_amp_mae"] <= frozen["absolute_amp_mae"] + tolerances["mae_amp"]
    assert (
        metrics["phase_wrapped_mae"]
        <= frozen["phase_wrapped_mae"] + tolerances["mae_phase"]
    )

    # Visual artifact: the evaluate stage's comparison figure must be real.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparison_path = _nonempty(root / "reconstruction/comparison.png")
    assert comparison_path.stat().st_size > 50_000
    rendered = plt.imread(comparison_path)
    assert float(np.std(rendered)) > 0.05

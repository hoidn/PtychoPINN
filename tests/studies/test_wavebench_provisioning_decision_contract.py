import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DECISION_PATH = (
    REPO_ROOT
    / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/provisioning_decision.json"
)
DATASET_MANIFEST_PATH = (
    REPO_ROOT
    / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/dataset_manifest.json"
)
NATIVE_BASELINE_PATH = (
    REPO_ROOT
    / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/native_baseline_provenance.json"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_wavebench_provisioning_decision_validator_passes():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/studies/validate_wavebench_provisioning_decision.py"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_wavebench_provisioning_decision_records_expected_dataset_target():
    decision = load_json(DECISION_PATH)
    dataset_manifest = load_json(DATASET_MANIFEST_PATH)

    assert decision["selected_variant"] == "time_varying/is/thick_lines_gaussian_lens"
    assert decision["stable_dataset_target"]["repo_relative"] == "wavebench_dataset/time_varying/is/"
    assert dataset_manifest["selected_member"] == "wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton"


def test_wavebench_native_baseline_decisions_are_explicit():
    native_baselines = load_json(NATIVE_BASELINE_PATH)

    for baseline_name in ("fno", "unet"):
        baseline = native_baselines[baseline_name]
        assert baseline["status"] in {"checkpoint_reusable", "retrain_required", "not_available"}
        if baseline["status"] == "checkpoint_reusable":
            assert baseline["checkpoints"]
        else:
            assert baseline["retrain_route"]

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = (
    REPO_ROOT
    / "docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight_summary.md"
)
METADATA_PATH = (
    REPO_ROOT
    / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/preflight_metadata.json"
)


def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def test_wavebench_preflight_contract_validator_passes():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/studies/validate_wavebench_preflight_contract.py"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_wavebench_preflight_metadata_records_inverse_source_variant_inventory():
    metadata = load_metadata()

    inventory = metadata["inverse_source_variant_inventory"]
    assert len(inventory) == 6

    variant_ids = {entry["variant_id"] for entry in inventory}
    assert variant_ids == {
        "time_varying/is/thick_lines_gaussian_lens",
        "time_varying/is/thick_lines_grf_isotropic",
        "time_varying/is/thick_lines_grf_anisotropic",
        "time_varying/is/mnist_gaussian_lens",
        "time_varying/is/mnist_grf_isotropic",
        "time_varying/is/mnist_grf_anisotropic",
    }

    thick_lines_entry = next(
        entry
        for entry in inventory
        if entry["variant_id"] == "time_varying/is/thick_lines_gaussian_lens"
    )
    assert thick_lines_entry["split_semantics"] == (
        "seed-42 random permutation into 9000 train / 500 val / 500 test via "
        "`get_dataloaders_is_thick_lines`"
    )

    mnist_entry = next(
        entry
        for entry in inventory
        if entry["variant_id"] == "time_varying/is/mnist_gaussian_lens"
    )
    assert mnist_entry["role"] == "ood_only_probe"
    assert mnist_entry["split_semantics"] == (
        "sequential evaluation-only loader via `get_dataloaders_is_mnist`; no "
        "train/val/test split is defined in the upstream loader"
    )


def test_wavebench_preflight_staging_contract_uses_stable_followup_target():
    metadata = load_metadata()
    summary = SUMMARY_PATH.read_text(encoding="utf-8")

    stable_target = "<wavebench repo>/wavebench_dataset/time_varying/is/"
    staging_path = metadata["staging_path"]

    assert staging_path["followup_target_description"] == stable_target
    assert staging_path["followup_target_repo_relative"] == "wavebench_dataset/time_varying/is/"
    assert staging_path["observed_local_path"] is None
    assert staging_path["retained_checkout_path"] is None
    assert stable_target in summary

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = REPO_ROOT / "docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight_summary.md"
METADATA_PATH = REPO_ROOT / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/preflight_metadata.json"
REPORT_PATH = REPO_ROOT / ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/execution_report.md"
INSPECTION_PATH = REPO_ROOT / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/verification/dataset_slice_inspection.json"
INDEX_PATH = REPO_ROOT / "docs/index.md"

ALLOWED_FINAL_STATUSES = {
    "ready_for_supervised_plan",
    "ready_for_supervised_and_physics_plan",
    "needs_dataset_or_checkpoint_decision",
    "not_suitable_for_current_manuscript",
}
ALLOWED_BASELINE_STATUSES = {"checkpoint_reusable", "retrain_required", "not_available"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load_json(path: Path) -> dict:
    require(path.exists(), f"missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    require(SUMMARY_PATH.exists(), f"missing summary: {SUMMARY_PATH}")
    require(REPORT_PATH.exists(), f"missing execution report: {REPORT_PATH}")
    require(INDEX_PATH.exists(), f"missing docs index: {INDEX_PATH}")

    summary = SUMMARY_PATH.read_text(encoding="utf-8")
    report = REPORT_PATH.read_text(encoding="utf-8")
    metadata = load_json(METADATA_PATH)
    inspection = load_json(INSPECTION_PATH)
    index = INDEX_PATH.read_text(encoding="utf-8")

    match = re.search(r"^- Final status: `([^`]+)`$", summary, flags=re.MULTILINE)
    require(bool(match), "summary is missing an explicit `Final status:` line")
    summary_status = match.group(1)
    require(
        summary_status in ALLOWED_FINAL_STATUSES,
        f"summary final status is invalid: {summary_status}",
    )
    require(
        metadata.get("final_status") == summary_status,
        "summary final status does not match metadata final_status",
    )

    for heading in (
        "## Completed In This Pass",
        "## Completed Current-Scope Work",
        "## Follow-Up Work",
        "## Residual Risks",
    ):
        require(heading in report, f"execution report missing required heading: {heading}")

    for field in (
        "repo_url",
        "repo_revision",
        "dataset_source",
        "dataset_files",
        "dataset_access",
        "dataset_provenance",
        "staging_path",
        "selected_variant",
        "selected_split",
        "sample_counts",
        "tensor_contracts",
        "native_baselines",
        "supervised_readiness",
        "forward_model",
        "follow_up_routes",
        "follow_up_directions",
        "final_status",
    ):
        require(field in metadata, f"metadata missing required top-level field: {field}")

    require(
        metadata["dataset_files"]["selected_variant_expected_files"]
        == ["wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton"],
        "metadata must record the distributed selected-variant `.beton` member",
    )

    observed_member = metadata["dataset_files"].get("selected_variant_observed_member", {})
    require(
        observed_member.get("inspection_artifact")
        == ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/verification/dataset_slice_inspection.json",
        "metadata must point at the dataset slice inspection artifact",
    )
    require(
        observed_member.get("zip_member") == inspection.get("remote_member"),
        "metadata observed-member zip path must match inspection artifact",
    )

    for key in ("train", "validation", "test"):
        require(key in metadata["sample_counts"], f"sample_counts missing {key}")

    tensor_contracts = metadata["tensor_contracts"]
    for key in ("observed_y", "target_q0", "wavespeed_cx"):
        require(key in tensor_contracts, f"tensor_contracts missing {key}")

    observed_y = tensor_contracts["observed_y"]
    target_q0 = tensor_contracts["target_q0"]
    require(observed_y.get("raw_shape_per_sample") == [334, 128], "observed_y raw shape must remain explicit")
    require(observed_y.get("archive_shape_per_sample") == [1, 128, 128], "observed_y archive shape missing")
    require(target_q0.get("archive_shape_per_sample") == [1, 128, 128], "target_q0 archive shape missing")
    for key in ("train_sample", "validation_sample", "test_sample"):
        require(
            key in observed_y.get("observed_loader_value_ranges", {}),
            f"observed_y missing value range for {key}",
        )
        require(
            key in target_q0.get("observed_loader_value_ranges", {}),
            f"target_q0 missing value range for {key}",
        )

    wavespeed = tensor_contracts["wavespeed_cx"]
    require(wavespeed.get("value_range_m_per_s") == [1400.0, 4000.0], "wavespeed value range must be stable")

    native_baselines = metadata["native_baselines"]
    for name in ("fno", "unet"):
        require(name in native_baselines, f"native_baselines missing {name}")
        entry = native_baselines[name]
        status = entry.get("status")
        require(status in ALLOWED_BASELINE_STATUSES, f"invalid native baseline status for {name}: {status}")
        if status == "checkpoint_reusable":
            require(
                not entry.get("blocker_reason"),
                f"{name} cannot be checkpoint_reusable while recording a blocker_reason",
            )
            require(
                bool(entry.get("checkpoint_identifier") or entry.get("local_checkpoint_path")),
                f"{name} checkpoint_reusable entries must name an exact checkpoint identifier or local path",
            )
        else:
            require(bool(entry.get("blocker_reason")), f"{name} non-reusable status must carry blocker_reason")

    supervised = metadata["supervised_readiness"]
    require(
        supervised.get("measurement_image_contract") == "stable_2d_loader_image",
        "supervised readiness must classify the WaveBench input as a stable 2D loader image",
    )
    require(
        supervised.get("recommended_channel_widths") == ["C=32", "C=64"],
        "supervised readiness must carry the approved C=32/C=64 follow-up widths",
    )

    forward_model = metadata["forward_model"]
    for key in ("availability", "sample_count", "normalization", "metrics", "accepted_thresholds", "physics_readiness"):
        require(key in forward_model, f"forward_model missing {key}")
    thresholds = forward_model["accepted_thresholds"]
    for key in (
        "median_normalized_residual_max",
        "fraction_examples_above_residual_ceiling_max",
        "per_example_normalized_residual_ceiling",
    ):
        require(thresholds.get(key) is not None, f"forward_model.accepted_thresholds missing numeric {key}")

    follow_up_routes = metadata["follow_up_routes"]
    for key in (
        "supervised_shared_encoder_benchmark",
        "native_baseline_reproduction",
        "forward_model_validation_or_physics_loss",
        "paper_bundle_assembly",
    ):
        require(key in follow_up_routes, f"follow_up_routes missing {key}")

    summary_terms = (
        "Selected split",
        "train/validation/test counts",
        "value ranges",
        "checkpoint",
        "Waveform MAE",
        "RMSE",
        "relative L2",
        "normalized residual",
        "C=32",
        "C=64",
    )
    for term in summary_terms:
        require(term in summary, f"summary missing required term: {term}")

    require(
        "wavebench_inverse_source_preflight_summary.md" in index,
        "docs/index.md must reference the WaveBench preflight summary",
    )

    print("wavebench preflight contract validated")


if __name__ == "__main__":
    main()

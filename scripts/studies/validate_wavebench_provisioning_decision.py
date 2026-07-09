from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = REPO_ROOT / "docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_provisioning_decision_summary.md"
DECISION_PATH = REPO_ROOT / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/provisioning_decision.json"
DATASET_MANIFEST_PATH = REPO_ROOT / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/dataset_manifest.json"
ENVIRONMENT_PROBE_PATH = REPO_ROOT / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/environment_probe.json"
NATIVE_BASELINE_PATH = REPO_ROOT / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/native_baseline_provenance.json"
CHECKPOINT_PROBE_PATH = REPO_ROOT / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/checkpoint_probe.json"
PREFLIGHT_SUMMARY_PATH = REPO_ROOT / "docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight_summary.md"
INDEX_PATH = REPO_ROOT / "docs/index.md"
REPORT_PATH = REPO_ROOT / "artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/execution_report.md"

EXPECTED_VARIANT = "time_varying/is/thick_lines_gaussian_lens"
EXPECTED_MEMBER = "wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton"
EXPECTED_TARGET = "wavebench_dataset/time_varying/is/"
EXPECTED_TARGET_DESCRIPTION = "<wavebench repo>/wavebench_dataset/time_varying/is/"
EXPECTED_ROUTE_ITEMS = {
    "2026-04-29-wavebench-native-baseline-reproduction",
    "2026-04-29-wavebench-shared-encoder-supervised-benchmark",
    "2026-04-29-wavebench-forward-model-physics-validation",
    "2026-04-29-wavebench-hybrid-physics-rows",
    "2026-04-29-wavebench-paper-table-figure-bundle",
}
ALLOWED_ROUTE_STATUSES = {"unblocked", "still_blocked", "needs_narrowing"}
ALLOWED_BASELINE_STATUSES = {"checkpoint_reusable", "retrain_required", "not_available"}
REQUIRED_IMPORTS = {"ffcv", "jax", "jwave", "ml_collections"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load_json(path: Path) -> dict:
    require(path.exists(), f"missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    require(SUMMARY_PATH.exists(), f"missing summary: {SUMMARY_PATH}")
    require(REPORT_PATH.exists(), f"missing execution report: {REPORT_PATH}")
    require(PREFLIGHT_SUMMARY_PATH.exists(), f"missing preflight summary: {PREFLIGHT_SUMMARY_PATH}")
    require(INDEX_PATH.exists(), f"missing docs index: {INDEX_PATH}")

    summary = SUMMARY_PATH.read_text(encoding="utf-8")
    report = REPORT_PATH.read_text(encoding="utf-8")
    preflight_summary = PREFLIGHT_SUMMARY_PATH.read_text(encoding="utf-8")
    index = INDEX_PATH.read_text(encoding="utf-8")

    decision = load_json(DECISION_PATH)
    dataset_manifest = load_json(DATASET_MANIFEST_PATH)
    environment_probe = load_json(ENVIRONMENT_PROBE_PATH)
    native_baselines = load_json(NATIVE_BASELINE_PATH)

    if CHECKPOINT_PROBE_PATH.exists():
        checkpoint_probe = load_json(CHECKPOINT_PROBE_PATH)
    else:
        checkpoint_probe = None

    for heading in (
        "## Completed In This Pass",
        "## Completed Plan Tasks",
        "## Remaining Required Plan Tasks",
        "## Verification",
        "## Residual Risks",
    ):
        require(heading in report, f"execution report missing required heading: {heading}")

    match = re.search(r"^- Selected variant: `([^`]+)`$", summary, flags=re.MULTILINE)
    require(bool(match), "summary is missing an explicit selected-variant line")
    require(match.group(1) == EXPECTED_VARIANT, "summary selected variant is incorrect")
    require(decision.get("selected_variant") == EXPECTED_VARIANT, "decision selected variant is incorrect")

    stable_target = decision.get("stable_dataset_target", {})
    require(
        stable_target.get("repo_relative") == EXPECTED_TARGET,
        "decision stable dataset target repo-relative path is incorrect",
    )
    require(
        stable_target.get("description") == EXPECTED_TARGET_DESCRIPTION,
        "decision stable dataset target description is incorrect",
    )
    require(EXPECTED_TARGET in summary, "summary must name the stable WaveBench dataset target")

    require(
        dataset_manifest.get("selected_member") == EXPECTED_MEMBER,
        "dataset manifest selected_member is incorrect",
    )
    require(
        dataset_manifest.get("stable_target_repo_relative") == EXPECTED_TARGET,
        "dataset manifest stable target is incorrect",
    )
    actual_dataset_path = Path(dataset_manifest.get("actual_path", ""))
    require(actual_dataset_path.exists(), f"staged dataset path does not exist: {actual_dataset_path}")
    require(
        dataset_manifest.get("transfer_mode") in {"already_present", "downloaded_remote_zip_member", "copied", "symlinked"},
        "dataset manifest transfer_mode is invalid",
    )
    require(
        bool(dataset_manifest.get("sha256") or dataset_manifest.get("mtime_epoch_s")),
        "dataset manifest must include checksum or size/mtime provenance",
    )
    require("wavebench_datasets" in summary and "wavebench_dataset" in summary, "summary must record path normalization")

    recommended_env = environment_probe.get("recommended_environment", {})
    require(
        recommended_env.get("name"),
        "environment probe must record a recommended environment name",
    )
    require(
        REQUIRED_IMPORTS.issubset(environment_probe.get("active_path_python", {}).get("imports", {})),
        "active PATH python probe is missing one or more required imports",
    )
    for name, probe in environment_probe.get("active_path_python", {}).get("imports", {}).items():
        require("ok" in probe, f"active import probe is missing ok status for {name}")
    if "probe_results" in recommended_env:
        require(
            REQUIRED_IMPORTS.issubset(recommended_env["probe_results"].get("imports", {})),
            "recommended environment probe is missing one or more required imports",
        )

    require("native_baselines" in decision, "decision missing native_baselines")
    for baseline_name in ("fno", "unet"):
        baseline = native_baselines.get(baseline_name)
        require(baseline is not None, f"native baseline provenance missing {baseline_name}")
        status = baseline.get("status")
        require(status in ALLOWED_BASELINE_STATUSES, f"invalid native baseline status for {baseline_name}: {status}")
        decision_status = decision["native_baselines"].get(baseline_name, {}).get("status")
        require(status == decision_status, f"decision and native baseline provenance disagree for {baseline_name}")
        if status == "checkpoint_reusable":
            checkpoints = baseline.get("checkpoints", [])
            require(checkpoints, f"{baseline_name} must name exact checkpoint artifacts when reusable")
            if checkpoint_probe is not None:
                require(
                    baseline_name in checkpoint_probe.get("baseline_load_smokes", {}),
                    f"checkpoint probe missing load smoke for {baseline_name}",
                )
        else:
            require(
                baseline.get("retrain_route"),
                f"{baseline_name} must include a retraining route when not reusable",
            )

    route_matrix = decision.get("downstream_route_matrix", {})
    require(set(route_matrix) == EXPECTED_ROUTE_ITEMS, "decision route matrix items are incomplete")
    for item_id, route in route_matrix.items():
        require(
            route.get("status") in ALLOWED_ROUTE_STATUSES,
            f"route matrix item {item_id} has invalid status {route.get('status')}",
        )
        require(route.get("reason"), f"route matrix item {item_id} is missing a reason")
        require(item_id in summary, f"summary must mention route matrix item {item_id}")
        require(route["status"] in summary, f"summary must mention status {route['status']}")

    require(
        "wavebench_provisioning_decision_summary.md" in preflight_summary,
        "preflight summary must link forward to the provisioning decision summary",
    )
    require(
        "wavebench_provisioning_decision_summary.md" in index,
        "docs index must reference the provisioning decision summary",
    )

    print("wavebench provisioning decision validated")


if __name__ == "__main__":
    main()

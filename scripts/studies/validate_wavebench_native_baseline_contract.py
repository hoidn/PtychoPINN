from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


OUTPUT_ROOT_RELATIVE = (
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-04-29-wavebench-native-baseline-reproduction"
)
SUMMARY_RELATIVE = (
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_native_baseline_summary.md"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.exists(), f"missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the WaveBench native-baseline contract bundle.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_RELATIVE)
    return parser.parse_args()


def row_by_id(rows: list[dict[str, Any]], row_id: str) -> dict[str, Any]:
    for row in rows:
        if row["row_id"] == row_id:
            return row
    raise SystemExit(f"table_ready_metrics.json missing row_id={row_id}")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if Path(args.output_root).is_absolute()
        else (repo_root / args.output_root).resolve()
    )

    summary_path = repo_root / SUMMARY_RELATIVE
    index_path = repo_root / "docs/index.md"
    evidence_matrix_path = repo_root / "docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md"
    manifest_path = output_root / "native_baseline_execution_manifest.json"
    table_metrics_path = output_root / "table_ready_metrics.json"
    unet_path = output_root / "native_unet_eval.json"
    fno_path = output_root / "native_fno_result.json"

    require(summary_path.exists(), f"missing summary: {summary_path}")
    require(index_path.exists(), f"missing docs index: {index_path}")
    require(evidence_matrix_path.exists(), f"missing evidence matrix: {evidence_matrix_path}")

    summary = summary_path.read_text(encoding="utf-8")
    index = index_path.read_text(encoding="utf-8")
    evidence_matrix = evidence_matrix_path.read_text(encoding="utf-8")
    manifest = load_json(manifest_path)
    table_metrics = load_json(table_metrics_path)
    unet = load_json(unet_path)
    fno = load_json(fno_path)

    match = re.search(r"^- Selected variant: `([^`]+)`$", summary, flags=re.MULTILINE)
    require(bool(match), "summary is missing an explicit selected-variant line")
    selected_variant = match.group(1)
    require(
        selected_variant == "time_varying/is/thick_lines_gaussian_lens",
        "summary selected variant is incorrect",
    )
    require(
        manifest["selected_variant"] == selected_variant == table_metrics["selected_variant"],
        "selected variant is inconsistent across summary/manifest/table metrics",
    )

    require(
        manifest["stable_dataset_target"]["repo_relative"] == "wavebench_dataset/time_varying/is/",
        "manifest stable dataset target is incorrect",
    )
    require(
        "<wavebench repo>/wavebench_dataset/time_varying/is/" in summary,
        "summary must name the stable dataset target",
    )

    require(
        "wavebench_native_baseline_summary.md" in index,
        "docs/index.md must reference the WaveBench native-baseline summary",
    )
    require(
        "wavebench_native_baseline_summary.md" in evidence_matrix,
        "evidence_matrix.md must reference the WaveBench native-baseline summary",
    )

    rows = table_metrics.get("rows", [])
    require(len(rows) == 2, "table_ready_metrics.json must contain exactly two native rows")
    unet_row = row_by_id(rows, manifest["native_rows"]["unet"]["row_id"])
    fno_row = row_by_id(rows, manifest["native_rows"]["fno"]["row_id"])

    require(
        unet_row["status"] == manifest["native_rows"]["unet"]["status"] == unet["status"],
        "U-Net row status is inconsistent",
    )
    require(
        fno_row["status"] == manifest["native_rows"]["fno"]["status"] == fno["status"],
        "FNO row status is inconsistent",
    )

    for row in (unet_row, fno_row):
        require(
            row["selected_variant"] == selected_variant,
            f"row {row['row_id']} selected_variant is inconsistent",
        )
        require(
            row["split"]["test_samples"] == 500,
            f"row {row['row_id']} must report the full 500-sample test split",
        )

    for metric in ("MAE", "RMSE", "RelL2", "SSIM"):
        require(metric in unet_row["metrics"], f"U-Net row missing metric {metric}")

    if fno_row["status"] == "completed":
        for metric in ("MAE", "RMSE", "RelL2", "SSIM"):
            require(metric in fno_row["metrics"], f"FNO row missing metric {metric}")
    else:
        require(
            bool(fno_row.get("blocker_reason") or fno.get("blocker_reason")),
            "blocked FNO rows must carry blocker_reason",
        )

    require(
        "candidate-lane" in summary and "external references" in summary,
        "summary must retain the WaveBench candidate-lane claim boundary",
    )

    print("wavebench native-baseline contract validated")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Runbook for the CDI hybrid-spectral to FFNO parameter-space study."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.studies.cdi_hybrid_spectral_ffno_parameter_space import (
    FRESH_ROWS,
    REUSED_ROWS,
    _sha256,
    build_preflight_artifacts,
)
from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare


DEFAULT_PROBE_NPZ = Path("datasets/Run1084_recon3_postPC_shrunk_3.npz")
DEFAULT_AUTHORITATIVE_ROOT = (
    Path(".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog")
    / "2026-04-29-cdi-lines128-paper-benchmark-execution/runs"
    / "complete_table_20260430T150757Z_repair_tmux"
)


def _fixed_compare_kwargs(output_root: Path, probe_npz: Path) -> Dict[str, Any]:
    return {
        "N": 128,
        "gridsize": 1,
        "output_dir": Path(output_root),
        "probe_npz": Path(probe_npz),
        "architectures": (),
        "seed": 3,
        "set_phi": True,
        "probe_source": "custom",
        "probe_scale_mode": "pad_extrapolate",
        "probe_smoothing_sigma": 0.5,
        "torch_epochs": 40,
        "torch_learning_rate": 2e-4,
        "torch_scheduler": "ReduceLROnPlateau",
        "torch_plateau_factor": 0.5,
        "torch_plateau_patience": 2,
        "torch_plateau_min_lr": 1e-4,
        "torch_plateau_threshold": 0.0,
        "torch_loss_mode": "mae",
        "torch_output_mode": "real_imag",
        "nimgs_train": 2,
        "nimgs_test": 2,
        "nphotons": 1e9,
        "fno_modes": 12,
        "fno_width": 32,
        "fno_blocks": 4,
        "fno_cnn_blocks": 2,
    }


def _fresh_row_specs() -> List[Dict[str, Any]]:
    return [dict(row) for row in FRESH_ROWS]


def _all_row_ids() -> Tuple[str, ...]:
    return tuple([row["model_id"] for row in REUSED_ROWS] + [row["model_id"] for row in FRESH_ROWS])


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _copy_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        raise RuntimeError(f"refusing to reuse symlinked path for authoritative materialization: {dst}")
    if dst.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _materialize_reused_rows(*, authoritative_root: Path, output_root: Path) -> None:
    _copy_path(authoritative_root / "recons" / "gt", output_root / "recons" / "gt")
    for row in REUSED_ROWS:
        model_id = str(row["model_id"])
        _copy_path(authoritative_root / "runs" / model_id, output_root / "runs" / model_id)
        _copy_path(authoritative_root / "recons" / model_id, output_root / "recons" / model_id)


def _ensure_row_not_active(output_root: Path, model_id: str) -> None:
    invocation_path = Path(output_root) / "runs" / model_id / "invocation.json"
    if not invocation_path.exists():
        return
    payload = _load_json(invocation_path)
    if str(payload.get("status", "")).lower() not in {"", "completed", "failed"}:
        raise RuntimeError(f"row output root already appears active for {model_id}: {invocation_path}")


def _fresh_row_required_paths(output_root: Path, model_id: str) -> Dict[str, Path]:
    run_dir = Path(output_root) / "runs" / model_id
    recon_dir = Path(output_root) / "recons" / model_id
    return {
        "invocation_json": run_dir / "invocation.json",
        "config_json": run_dir / "config.json",
        "history_json": run_dir / "history.json",
        "metrics_json": run_dir / "metrics.json",
        "exit_code_proof_json": run_dir / "exit_code_proof.json",
        "recon_npz": recon_dir / "recon.npz",
    }


def _fresh_row_has_completion_proof(output_root: Path, model_id: str) -> bool:
    required = _fresh_row_required_paths(output_root, model_id)
    if not all(path.exists() for path in required.values()):
        return False
    payload = _load_json(required["invocation_json"])
    return payload.get("status") == "completed" and payload.get("exit_code") == 0


def _scrub_row_outputs(output_root: Path, model_id: str) -> None:
    for root_name in ("runs", "recons"):
        row_path = Path(output_root) / root_name / model_id
        if row_path.is_symlink() or row_path.is_file():
            row_path.unlink()
        elif row_path.exists():
            shutil.rmtree(row_path)


def _prepare_fresh_row_for_launch(output_root: Path, model_id: str) -> bool:
    _ensure_row_not_active(output_root, model_id)
    if _fresh_row_has_completion_proof(output_root, model_id):
        return True
    required = _fresh_row_required_paths(output_root, model_id)
    if any(path.exists() for path in required.values()):
        _scrub_row_outputs(output_root, model_id)
    return False


def _collect_reused_root_drift(
    *,
    output_root: Path,
    reference_runs_path: Path,
    authoritative_root: str | None = None,
) -> Dict[str, List[str]]:
    reference_runs = _load_json(Path(reference_runs_path))
    reused_root_drift: Dict[str, List[str]] = {}
    if authoritative_root is not None and authoritative_root != reference_runs.get("authoritative_root"):
        reused_root_drift["__authoritative_root__"] = [
            f"materialized_authoritative_root={authoritative_root!r}",
            f"reference_runs.authoritative_root={reference_runs.get('authoritative_root')!r}",
        ]
    for row in reference_runs.get("reused_rows", []):
        model_id = str(row["model_id"])
        problems: List[str] = []
        local_run_dir = Path(output_root) / "runs" / model_id
        local_recon_dir = Path(output_root) / "recons" / model_id
        if local_run_dir.is_symlink():
            problems.append(f"run dir is still a symlink: {local_run_dir}")
        if local_recon_dir.is_symlink():
            problems.append(f"recon dir is still a symlink: {local_recon_dir}")
        for key, expected_sha in row.get("validation", {}).get("source_sha256", {}).items():
            local_path = {
                "invocation_json": local_run_dir / "invocation.json",
                "config_json": local_run_dir / "config.json",
                "history_json": local_run_dir / "history.json",
                "metrics_json": local_run_dir / "metrics.json",
                "recon_npz": local_recon_dir / "recon.npz",
            }[key]
            if local_path.is_symlink():
                problems.append(f"artifact is symlinked: {local_path}")
                continue
            if not local_path.exists():
                problems.append(f"artifact missing: {local_path}")
                continue
            actual_sha = _sha256(local_path)
            if actual_sha != expected_sha:
                problems.append(f"{key} sha256 drifted: expected {expected_sha}, found {actual_sha}")
        if problems:
            reused_root_drift[model_id] = problems
    return reused_root_drift


def _format_validation_failures(validation: Dict[str, Any]) -> str:
    parts: List[str] = []
    if validation.get("missing_rows"):
        parts.append("missing_rows=" + ", ".join(validation["missing_rows"]))
    for model_id, problems in validation.get("reused_root_drift", {}).items():
        parts.append(f"{model_id}: " + "; ".join(problems))
    for model_id, problems in validation.get("fresh_row_completion_failures", {}).items():
        parts.append(f"{model_id}: " + "; ".join(problems))
    if validation.get("missing_artifacts"):
        parts.append("missing_artifacts=" + json.dumps(validation["missing_artifacts"], sort_keys=True))
    if validation.get("missing_merged_outputs"):
        parts.append("missing_merged_outputs=" + ", ".join(validation["missing_merged_outputs"]))
    for output_name, problems in validation.get("merged_output_failures", {}).items():
        parts.append(f"{output_name}: " + "; ".join(problems))
    if validation.get("shared_contract_failures"):
        parts.append("shared_contract_failures=" + "; ".join(validation["shared_contract_failures"]))
    return " | ".join(parts) if parts else json.dumps(validation, sort_keys=True)


def _collect_contract_projection_mismatches(
    *,
    model_id: str,
    scope: str,
    actual: Dict[str, Any],
    expected: Dict[str, Any],
) -> List[str]:
    problems: List[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            problems.append(
                f"{scope} contract mismatch for {key}: expected {expected_value!r}, found {actual_value!r}"
            )
    return problems


def _collect_shared_contract_failures(
    *,
    output_root: Path,
    shared_contract: Dict[str, Any],
) -> List[str]:
    failures: List[str] = []
    dataset_manifest_path = Path(output_root) / "dataset_identity_manifest.json"
    split_manifest_path = Path(output_root) / "split_manifest.json"
    preflight_validation_path = Path(output_root) / "preflight" / "preflight_validation.json"

    if not dataset_manifest_path.exists():
        failures.append(f"missing dataset_identity_manifest.json: {dataset_manifest_path}")
    else:
        dataset_manifest = _load_json(dataset_manifest_path)
        for key in ("dataset_source", "probe_source", "probe_scale_mode"):
            actual = dataset_manifest.get(key)
            expected = shared_contract.get(key)
            if actual != expected:
                failures.append(
                    f"dataset_identity_manifest.json mismatch for {key}: expected {expected!r}, found {actual!r}"
                )
        for key in ("train_npz", "test_npz", "probe_npz"):
            record = dataset_manifest.get(key)
            if not isinstance(record, dict):
                failures.append(f"dataset_identity_manifest.json missing mapping for {key}")
                continue
            actual_path = record.get("path")
            expected_path = shared_contract.get(key)
            expected_path_obj = Path(str(expected_path))
            if not expected_path_obj.is_absolute():
                candidate_paths = [expected_path_obj, Path(output_root) / expected_path_obj]
                expected_path_obj = next((path for path in candidate_paths if path.exists()), candidate_paths[-1])
            if actual_path is None or not Path(actual_path).exists():
                failures.append(f"dataset_identity_manifest.json path missing on disk for {key}: {actual_path!r}")
                continue
            actual_path_obj = Path(actual_path)
            if key != "probe_npz" and actual_path_obj.resolve() != expected_path_obj.resolve():
                failures.append(
                    f"dataset_identity_manifest.json mismatch for {key}.path: "
                    f"expected {expected_path!r}, found {actual_path!r}"
                )
            actual_sha = _sha256(actual_path_obj)
            recorded_sha = record.get("sha256")
            if key == "probe_npz":
                if not expected_path_obj.exists():
                    failures.append(
                        f"dataset_identity_manifest.json expected probe path missing on disk for {key}: {expected_path!r}"
                    )
                    continue
                expected_sha = _sha256(expected_path_obj)
                if actual_sha != expected_sha:
                    failures.append(
                        f"dataset_identity_manifest.json mismatch for {key}.sha256: "
                        f"expected canonical probe digest {expected_sha!r}, found {actual_sha!r}"
                    )
                if recorded_sha != expected_sha:
                    failures.append(
                        f"dataset_identity_manifest.json mismatch for {key}.recorded_sha256: "
                        f"expected canonical probe digest {expected_sha!r}, found {recorded_sha!r}"
                    )
            elif recorded_sha != actual_sha:
                failures.append(
                    f"dataset_identity_manifest.json mismatch for {key}.sha256: "
                    f"expected {actual_sha!r}, found {recorded_sha!r}"
                )

    if not split_manifest_path.exists():
        failures.append(f"missing split_manifest.json: {split_manifest_path}")
    else:
        split_manifest = _load_json(split_manifest_path)
        for key in ("seed", "nimgs_train", "nimgs_test", "gridsize", "set_phi"):
            actual = split_manifest.get(key)
            expected = shared_contract.get(key)
            if actual != expected:
                failures.append(f"split_manifest.json mismatch for {key}: expected {expected!r}, found {actual!r}")
        for key in ("train_npz", "test_npz"):
            actual_path = split_manifest.get(key)
            expected_path = shared_contract.get(key)
            expected_path_obj = Path(str(expected_path))
            if actual_path is None or Path(actual_path).resolve() != expected_path_obj.resolve():
                failures.append(
                    f"split_manifest.json mismatch for {key}: expected {expected_path!r}, found {actual_path!r}"
                )

    if not preflight_validation_path.exists():
        failures.append(f"missing preflight/preflight_validation.json: {preflight_validation_path}")
    else:
        preflight_validation = _load_json(preflight_validation_path)
        preflight_contract = preflight_validation.get("contract")
        if not isinstance(preflight_contract, dict):
            failures.append("preflight/preflight_validation.json missing contract mapping")
        else:
            for key, expected in shared_contract.items():
                if key in {"train_npz", "test_npz"}:
                    continue
                actual = preflight_contract.get(key)
                if actual != expected:
                    failures.append(
                        f"preflight/preflight_validation.json mismatch for {key}: "
                        f"expected {expected!r}, found {actual!r}"
                    )

    return failures


def _validate_fresh_row_completion(
    *,
    output_root: Path,
    matrix_rows: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    failures: Dict[str, List[str]] = {}
    for row in matrix_rows:
        if row.get("row_kind") != "fresh_bridge":
            continue
        model_id = str(row["model_id"])
        required = _fresh_row_required_paths(output_root, model_id)
        if not all(path.exists() for path in required.values()):
            continue
        problems: List[str] = []
        try:
            invocation_payload = _load_json(required["invocation_json"])
        except json.JSONDecodeError as exc:
            problems.append(f"invocation.json parse failed: {exc}")
            invocation_payload = {}
        try:
            config_payload = _load_json(required["config_json"])
        except json.JSONDecodeError as exc:
            problems.append(f"config.json parse failed: {exc}")
            config_payload = {}
        try:
            exit_code_payload = _load_json(required["exit_code_proof_json"])
        except json.JSONDecodeError as exc:
            problems.append(f"exit_code_proof.json parse failed: {exc}")
            exit_code_payload = {}
        status = invocation_payload.get("status")
        exit_code = invocation_payload.get("exit_code")
        proof_exit_code = exit_code_payload.get("exit_code")
        if status != "completed":
            problems.append(f"expected invocation status='completed', found status={status!r}")
        if exit_code != 0:
            problems.append(f"expected invocation exit_code=0, found exit_code={exit_code!r}")
        if proof_exit_code != 0:
            problems.append(f"expected exit_code_proof exit_code=0, found exit_code={proof_exit_code!r}")
        contract_projection = row.get("contract_projection", {})
        if contract_projection:
            parsed_args = invocation_payload.get("parsed_args", {})
            problems.extend(
                _collect_contract_projection_mismatches(
                    model_id=model_id,
                    scope="invocation.parsed_args",
                    actual=parsed_args,
                    expected=contract_projection,
                )
            )
            torch_runner_config = config_payload.get("torch_runner_config")
            if isinstance(torch_runner_config, dict):
                problems.extend(
                    _collect_contract_projection_mismatches(
                        model_id=model_id,
                        scope="config.torch_runner_config",
                        actual=torch_runner_config,
                        expected=contract_projection,
                    )
                )
            else:
                problems.append("config.json missing torch_runner_config mapping for contract validation")
        if problems:
            failures[model_id] = problems
    return failures


def _validate_merged_outputs(
    *,
    output_root: Path,
    matrix_rows: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, List[str]]]:
    required_outputs = {
        "metrics_by_model.json": Path(output_root) / "metrics_by_model.json",
        "metrics.json": Path(output_root) / "metrics.json",
        "model_manifest.json": Path(output_root) / "model_manifest.json",
        "metrics_table.csv": Path(output_root) / "metrics_table.csv",
        "metrics_table.tex": Path(output_root) / "metrics_table.tex",
    }
    missing_outputs = [name for name, path in required_outputs.items() if not path.exists()]
    failures: Dict[str, List[str]] = {}
    expected_row_ids = tuple(str(row["model_id"]) for row in matrix_rows)
    expected_ids = set(expected_row_ids)

    metrics_by_model_path = required_outputs["metrics_by_model.json"]
    if metrics_by_model_path.exists():
        try:
            metrics_by_model = _load_json(metrics_by_model_path)
        except json.JSONDecodeError as exc:
            failures["metrics_by_model.json"] = [f"parse failed: {exc}"]
        else:
            found_ids = set(metrics_by_model.keys())
            missing_ids = sorted(expected_ids - found_ids)
            if missing_ids:
                failures["metrics_by_model.json"] = [
                    "missing expected row IDs: " + ", ".join(missing_ids)
                ]

    metrics_path = required_outputs["metrics.json"]
    if metrics_path.exists():
        try:
            merged_metrics = _load_json(metrics_path)
        except json.JSONDecodeError as exc:
            failures["metrics.json"] = [f"parse failed: {exc}"]
        else:
            found_ids = set(merged_metrics.keys())
            missing_ids = sorted(expected_ids - found_ids)
            if missing_ids:
                failures["metrics.json"] = [
                    "missing expected row IDs: " + ", ".join(missing_ids)
                ]

    model_manifest_path = required_outputs["model_manifest.json"]
    if model_manifest_path.exists():
        try:
            model_manifest = _load_json(model_manifest_path)
        except json.JSONDecodeError as exc:
            failures["model_manifest.json"] = [f"parse failed: {exc}"]
        else:
            rows = model_manifest.get("rows")
            if not isinstance(rows, list):
                failures["model_manifest.json"] = ["missing rows list"]
            else:
                found_ids = {
                    str(row.get("model_id", "")).strip()
                    for row in rows
                    if isinstance(row, dict)
                }
                missing_ids = sorted(expected_ids - found_ids)
                if missing_ids:
                    failures["model_manifest.json"] = [
                        "missing expected row IDs: " + ", ".join(missing_ids)
                    ]

    table_csv_path = required_outputs["metrics_table.csv"]
    if table_csv_path.exists():
        try:
            with table_csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except (csv.Error, UnicodeDecodeError) as exc:
            failures["metrics_table.csv"] = [f"parse failed: {exc}"]
        else:
            if not rows:
                failures["metrics_table.csv"] = ["table is empty"]
            else:
                if "model_id" not in rows[0]:
                    failures["metrics_table.csv"] = ["missing model_id column"]
                else:
                    expected_labels = {
                        str(row["model_id"]): str(row.get("display_label") or row.get("model_label") or row["model_id"])
                        for row in matrix_rows
                    }
                    found_ids = {str(row.get("model_id", "")).strip() for row in rows}
                    missing_ids = sorted(expected_ids - found_ids)
                    csv_problems: List[str] = []
                    if missing_ids:
                        csv_problems.append("missing expected row IDs: " + ", ".join(missing_ids))
                    for row in rows:
                        model_id = str(row.get("model_id", "")).strip()
                        if model_id not in expected_labels:
                            continue
                        actual_label = str(row.get("model_label", "")).strip()
                        expected_label = expected_labels[model_id]
                        if actual_label != expected_label:
                            csv_problems.append(
                                f"model_label mismatch for {model_id}: expected {expected_label!r}, found {actual_label!r}"
                            )
                    if csv_problems:
                        failures["metrics_table.csv"] = csv_problems

    table_tex_path = required_outputs["metrics_table.tex"]
    if table_tex_path.exists():
        try:
            table_tex = table_tex_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            failures["metrics_table.tex"] = [f"read failed: {exc}"]
        else:
            if not table_tex.strip():
                failures["metrics_table.tex"] = ["table is empty"]
            else:
                allowed_tex_names: Dict[str, str] = {}
                for row in matrix_rows:
                    model_id = str(row["model_id"])
                    label = str(row.get("display_label") or row.get("model_label") or model_id)
                    for candidate in {label, label.replace("_", r"\_")}:
                        allowed_tex_names[candidate] = model_id

                body_lines = []
                in_body = False
                for raw_line in table_tex.splitlines():
                    line = raw_line.strip()
                    if line == r"\midrule":
                        in_body = True
                        continue
                    if line == r"\bottomrule":
                        break
                    if in_body and "&" in line:
                        body_lines.append(line)

                tex_problems: List[str] = []
                if len(body_lines) != len(expected_row_ids):
                    tex_problems.append(
                        f"expected {len(expected_row_ids)} table rows between \\midrule and \\bottomrule, "
                        f"found {len(body_lines)}"
                    )

                found_ids = set()
                for line in body_lines:
                    columns = [column.strip() for column in line.split("&")]
                    if len(columns) < 2:
                        tex_problems.append(f"malformed table row: {line}")
                        continue
                    model_cell = columns[1]
                    resolved_model_id = allowed_tex_names.get(model_cell)
                    if resolved_model_id is None:
                        tex_problems.append(f"unexpected model cell {model_cell!r}")
                        continue
                    found_ids.add(resolved_model_id)

                missing_ids = sorted(expected_ids - found_ids)
                if missing_ids:
                    tex_problems.append("missing expected row IDs: " + ", ".join(missing_ids))
                if tex_problems:
                    failures["metrics_table.tex"] = tex_problems

    return missing_outputs, failures


def _validate_reused_anchor_materialization(
    *,
    output_root: Path,
    reference_runs_path: Path,
    authoritative_root: str,
) -> Dict[str, Any]:
    reused_root_drift = _collect_reused_root_drift(
        output_root=Path(output_root),
        reference_runs_path=Path(reference_runs_path),
        authoritative_root=authoritative_root,
    )
    return {
        "ok": not reused_root_drift,
        "reused_root_drift": reused_root_drift,
    }


def validate_cdi_parameter_space_bundle(
    *,
    output_root: Path,
    study_matrix_path: Path,
    reference_runs_path: Path | None = None,
) -> Dict[str, Any]:
    matrix = _load_json(Path(study_matrix_path))
    missing_rows: List[str] = []
    row_reports: Dict[str, List[str]] = {}
    matrix_rows = list(matrix.get("rows", []))
    shared_contract = dict(matrix.get("shared_contract", {}))
    for row in matrix_rows:
        model_id = str(row["model_id"])
        required = [
            Path(output_root) / "runs" / model_id / "invocation.json",
            Path(output_root) / "runs" / model_id / "config.json",
            Path(output_root) / "runs" / model_id / "history.json",
            Path(output_root) / "runs" / model_id / "metrics.json",
            Path(output_root) / "runs" / model_id / "exit_code_proof.json",
            Path(output_root) / "recons" / model_id / "recon.npz",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            missing_rows.append(model_id)
            row_reports[model_id] = missing
    reused_root_drift: Dict[str, List[str]] = {}
    if reference_runs_path is not None:
        reused_root_drift = _collect_reused_root_drift(
            output_root=Path(output_root),
            reference_runs_path=Path(reference_runs_path),
            authoritative_root=matrix.get("authoritative_anchor_root"),
        )
    fresh_row_completion_failures = _validate_fresh_row_completion(
        output_root=Path(output_root),
        matrix_rows=matrix_rows,
    )
    missing_merged_outputs, merged_output_failures = _validate_merged_outputs(
        output_root=Path(output_root),
        matrix_rows=matrix_rows,
    )
    shared_contract_failures = _collect_shared_contract_failures(
        output_root=Path(output_root),
        shared_contract=shared_contract,
    ) if shared_contract else []
    return {
        "ok": (
            not missing_rows
            and not reused_root_drift
            and not fresh_row_completion_failures
            and not missing_merged_outputs
            and not merged_output_failures
            and not shared_contract_failures
        ),
        "missing_rows": missing_rows,
        "missing_artifacts": row_reports,
        "reused_root_drift": reused_root_drift,
        "fresh_row_completion_failures": fresh_row_completion_failures,
        "missing_merged_outputs": missing_merged_outputs,
        "merged_output_failures": merged_output_failures,
        "shared_contract_failures": shared_contract_failures,
    }


def run_cdi_parameter_space_study(
    *,
    authoritative_root: Path,
    output_root: Path,
    preflight_root: Path,
    note_path: Path,
    probe_npz: Path = DEFAULT_PROBE_NPZ,
    preflight_only: bool = False,
) -> Dict[str, Any]:
    matrix_path = Path(preflight_root) / "study_matrix.json"
    reference_runs_path = Path(preflight_root) / "reference_runs.json"
    preflight_paths = build_preflight_artifacts(
        authoritative_root=Path(authoritative_root),
        artifact_root=Path(output_root),
        note_path=Path(note_path),
        matrix_path=matrix_path,
        reference_runs_path=reference_runs_path,
    )

    fresh_specs = _fresh_row_specs()
    all_specs = [*REUSED_ROWS, *fresh_specs]
    fresh_models = tuple(str(spec["model_id"]) for spec in fresh_specs)
    result: Dict[str, Any] = dict(preflight_paths)
    if preflight_only:
        result["preflight_validation"] = run_grid_lines_compare(
            **_fixed_compare_kwargs(Path(output_root), Path(probe_npz)),
            models=fresh_models,
            row_specs=tuple(fresh_specs),
            model_n={model_id: 128 for model_id in fresh_models},
            preflight_only=True,
        )
        _write_json(Path(preflight_root) / "preflight_validation.json", result["preflight_validation"])
        return result

    _materialize_reused_rows(
        authoritative_root=Path(authoritative_root),
        output_root=Path(output_root),
    )
    reused_validation = _validate_reused_anchor_materialization(
        output_root=Path(output_root),
        reference_runs_path=Path(preflight_paths["reference_runs_path"]),
        authoritative_root=str(authoritative_root),
    )
    result["reused_anchor_validation"] = reused_validation
    if not reused_validation["ok"]:
        raise RuntimeError(
            "reused-anchor validation failed before fresh launches: "
            + _format_validation_failures(reused_validation)
        )

    for spec in fresh_specs:
        model_id = str(spec["model_id"])
        if _prepare_fresh_row_for_launch(Path(output_root), model_id):
            continue
        run_grid_lines_compare(
            **_fixed_compare_kwargs(Path(output_root), Path(probe_npz)),
            models=(model_id,),
            row_specs=(spec,),
            model_n={model_id: 128},
            reuse_existing_recons=False,
        )

    all_models = _all_row_ids()
    result["collated_bundle"] = run_grid_lines_compare(
        **_fixed_compare_kwargs(Path(output_root), Path(probe_npz)),
        models=all_models,
        row_specs=tuple(all_specs),
        model_n={model_id: 128 for model_id in all_models},
        reuse_existing_recons=True,
    )
    result["bundle_validation"] = validate_cdi_parameter_space_bundle(
        output_root=Path(output_root),
        study_matrix_path=Path(preflight_paths["study_matrix_path"]),
        reference_runs_path=Path(preflight_paths["reference_runs_path"]),
    )
    _write_json(Path(output_root) / "analysis" / "bundle_validation.json", result["bundle_validation"])
    if not result["bundle_validation"]["ok"]:
        raise RuntimeError(
            "bundle validation failed after collation: "
            + _format_validation_failures(result["bundle_validation"])
        )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative-root", type=Path, default=DEFAULT_AUTHORITATIVE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--note-path", type=Path, required=True)
    parser.add_argument("--probe-npz", type=Path, default=DEFAULT_PROBE_NPZ)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    run_cdi_parameter_space_study(
        authoritative_root=args.authoritative_root,
        output_root=args.output_root,
        preflight_root=args.preflight_root,
        note_path=args.note_path,
        probe_npz=args.probe_npz,
        preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    main()

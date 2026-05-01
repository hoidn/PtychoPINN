#!/usr/bin/env python3
"""Runbook for the CDI hybrid-spectral to FFNO parameter-space study."""

from __future__ import annotations

import argparse
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
    if validation.get("missing_artifacts"):
        parts.append("missing_artifacts=" + json.dumps(validation["missing_artifacts"], sort_keys=True))
    return " | ".join(parts) if parts else json.dumps(validation, sort_keys=True)


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
    for row in matrix.get("rows", []):
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
    return {
        "ok": not missing_rows and not reused_root_drift,
        "missing_rows": missing_rows,
        "missing_artifacts": row_reports,
        "reused_root_drift": reused_root_drift,
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
        row_specs=tuple(fresh_specs),
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

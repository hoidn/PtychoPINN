#!/usr/bin/env python
"""Study-local HIO/ER baseline for the non-ML single-shot CDI benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_RELATIVE = Path("scripts/reconstruction/hio_cdi_benchmark.py")
TARGET_N = 64
DEFAULT_CONDITION = "gs1"
DEFAULT_TABLE2_SIZE = 392
DEFAULT_TABLE2_OFFSET = 4
DEFAULT_OUTER_OFFSET_TRAIN = 8
DEFAULT_OUTER_OFFSET_TEST = 20
DEFAULT_NIMGS_TRAIN = 2
DEFAULT_NIMGS_TEST = 2
DEFAULT_NPHOTONS = 1e9
DEFAULT_EPSILON_RATIO = 1e-6
SELECTED_SOLVER = "study_local_hio_er"


@dataclass(frozen=True)
class RestartResult:
    seed: int
    psi: np.ndarray
    final_residual: float
    residual_curve: list[float]


@dataclass(frozen=True)
class RestartRun:
    restarts: list[RestartResult]
    selected: RestartResult


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (float, np.floating)):
        item = float(value)
        if not math.isfinite(item):
            return None
        return item
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def _repo_relative(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256_file(path: Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    record: dict[str, Any] = {
        "path": _repo_relative(path),
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        stat = path.stat()
        record.update(
            {
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": _sha256_file(path),
            }
        )
    return record


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _validate_2d_complex(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(np.complex128, copy=False)


def _validate_target_magnitude(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target = np.asarray(value, dtype=np.float64)
    if target.shape != shape:
        raise ValueError(f"target magnitude shape {target.shape} does not match psi shape {shape}")
    if not np.all(np.isfinite(target)):
        raise ValueError("target magnitude must contain only finite values")
    if np.any(target < 0):
        raise ValueError("target magnitude must be nonnegative")
    return target


def make_probe_support(
    probe: np.ndarray,
    threshold: float,
    threshold_grid: Sequence[float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct the pre-registered support from abs(P) >= threshold * max(abs(P))."""
    probe_array = _validate_2d_complex("probe", probe)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("support threshold must be finite and nonnegative")
    amplitude = np.abs(probe_array)
    max_amp = float(amplitude.max(initial=0.0))
    if max_amp <= 0:
        raise ValueError("zero-amplitude probe cannot define a support")

    cutoff = float(threshold) * max_amp
    support = amplitude >= cutoff
    pixel_count = int(np.count_nonzero(support))
    if pixel_count == 0:
        raise ValueError("empty support mask")
    if pixel_count == support.size:
        raise ValueError("full-frame support mask")

    record = {
        "support_source": "known_probe_amplitude",
        "support_rule": "abs(P) >= support_threshold * max(abs(P))",
        "support_threshold": float(threshold),
        "threshold_grid": [float(item) for item in (threshold_grid or [threshold])],
        "selection_policy": "pre_registered_primary_not_metric_selected",
        "support_pixel_count": pixel_count,
        "support_fraction": pixel_count / int(support.size),
        "probe_max_amplitude": max_amp,
        "probe_division_epsilon": DEFAULT_EPSILON_RATIO * max_amp,
        "oracle_known_probe_prior": True,
    }
    return support.astype(np.bool_), record


def forward_amplitude(psi: np.ndarray) -> np.ndarray:
    psi_array = _validate_2d_complex("psi", psi)
    norm = math.sqrt(float(psi_array.size))
    return np.abs(np.fft.fftshift(np.fft.fft2(psi_array)) / norm)


def project_fourier_magnitude(psi: np.ndarray, target_magnitude: np.ndarray) -> np.ndarray:
    psi_array = _validate_2d_complex("psi", psi)
    target = _validate_target_magnitude(target_magnitude, psi_array.shape)
    norm = math.sqrt(float(psi_array.size))
    current_shifted = np.fft.fftshift(np.fft.fft2(psi_array)) / norm
    projected_shifted = target * np.exp(1j * np.angle(current_shifted))
    return np.fft.ifft2(np.fft.ifftshift(projected_shifted * norm))


def hio_update(
    previous: np.ndarray,
    target_magnitude: np.ndarray,
    support: np.ndarray,
    beta: float = 0.9,
) -> np.ndarray:
    previous_array = _validate_2d_complex("previous", previous)
    support_mask = np.asarray(support, dtype=bool)
    if support_mask.shape != previous_array.shape:
        raise ValueError("support shape must match previous")
    projected = project_fourier_magnitude(previous_array, target_magnitude)
    updated = np.empty_like(projected)
    updated[support_mask] = projected[support_mask]
    updated[~support_mask] = previous_array[~support_mask] - float(beta) * projected[~support_mask]
    return updated


def er_cleanup(previous: np.ndarray, target_magnitude: np.ndarray, support: np.ndarray) -> np.ndarray:
    previous_array = _validate_2d_complex("previous", previous)
    support_mask = np.asarray(support, dtype=bool)
    if support_mask.shape != previous_array.shape:
        raise ValueError("support shape must match previous")
    projected = project_fourier_magnitude(previous_array, target_magnitude)
    cleaned = np.zeros_like(projected)
    cleaned[support_mask] = projected[support_mask]
    return cleaned


def fourier_residual(psi: np.ndarray, target_magnitude: np.ndarray) -> float:
    psi_array = _validate_2d_complex("psi", psi)
    target = _validate_target_magnitude(target_magnitude, psi_array.shape)
    denominator = float(np.linalg.norm(target))
    if denominator == 0:
        denominator = 1.0
    residual = np.linalg.norm(forward_amplitude(psi_array) - target) / denominator
    return float(residual)


def select_restart_by_residual(results: Sequence[RestartResult]) -> RestartResult:
    if not results:
        raise ValueError("at least one restart result is required")
    return min(results, key=lambda item: (float(item.final_residual), int(item.seed)))


def _initial_psi(target_magnitude: np.ndarray, seed: int) -> np.ndarray:
    target = np.asarray(target_magnitude, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    phase = rng.uniform(-np.pi, np.pi, size=target.shape)
    norm = math.sqrt(float(target.size))
    shifted = target * np.exp(1j * phase)
    return np.fft.ifft2(np.fft.ifftshift(shifted * norm))


def run_restarts(
    target_magnitude: np.ndarray,
    support: np.ndarray,
    seeds: Sequence[int],
    beta: float,
    hio_iters: int,
    er_iters: int,
    residual_period: int = 10,
) -> RestartRun:
    target = np.asarray(target_magnitude, dtype=np.float64)
    if target.ndim != 2:
        raise ValueError("target magnitude must be a 2D array")
    target = _validate_target_magnitude(target, target.shape)
    support_mask = np.asarray(support, dtype=bool)
    if support_mask.shape != target.shape:
        raise ValueError("support shape must match target magnitude")
    if not seeds:
        raise ValueError("at least one restart seed is required")
    if hio_iters < 0 or er_iters < 0:
        raise ValueError("iteration counts must be nonnegative")
    residual_period = max(1, int(residual_period))

    results: list[RestartResult] = []
    for seed in seeds:
        psi = _initial_psi(target, int(seed))
        curve = [fourier_residual(psi, target)]
        for iteration in range(1, int(hio_iters) + 1):
            psi = hio_update(psi, target, support_mask, beta=beta)
            if iteration % residual_period == 0:
                curve.append(fourier_residual(psi, target))
        for iteration in range(1, int(er_iters) + 1):
            psi = er_cleanup(psi, target, support_mask)
            if iteration % residual_period == 0:
                curve.append(fourier_residual(psi, target))
        final_residual = fourier_residual(psi, target)
        if not np.isclose(curve[-1], final_residual, rtol=0.0, atol=0.0):
            curve.append(final_residual)
        if not np.isfinite(final_residual):
            raise FloatingPointError(f"non-finite Fourier residual for restart seed {seed}")
        results.append(
            RestartResult(
                seed=int(seed),
                psi=np.asarray(psi),
                final_residual=float(final_residual),
                residual_curve=[float(item) for item in curve],
            )
        )

    selected = select_restart_by_residual(results)
    return RestartRun(restarts=results, selected=selected)


def recover_object_patch(
    psi: np.ndarray,
    probe: np.ndarray,
    support: np.ndarray,
    epsilon_ratio: float = DEFAULT_EPSILON_RATIO,
) -> np.ndarray:
    psi_array = _validate_2d_complex("psi", psi)
    probe_array = _validate_2d_complex("probe", probe)
    support_mask = np.asarray(support, dtype=bool)
    if probe_array.shape != psi_array.shape or support_mask.shape != psi_array.shape:
        raise ValueError("psi, probe, and support shapes must match")
    max_amp = float(np.abs(probe_array).max(initial=0.0))
    if max_amp <= 0:
        raise ValueError("zero-amplitude probe cannot be used for object recovery")
    epsilon = float(epsilon_ratio) * max_amp
    safe_probe = probe_array.copy()
    small = np.abs(safe_probe) < epsilon
    safe_probe[small] = epsilon + 0j
    recovered = np.zeros_like(psi_array)
    recovered[support_mask] = psi_array[support_mask] / safe_probe[support_mask]
    return recovered.astype(np.complex64)


def build_ambiguity_policy(
    oracle_diagnostic: bool = False,
    output_label: str | None = None,
) -> dict[str, Any]:
    if oracle_diagnostic:
        if output_label in {None, "", "primary", "main"}:
            raise ValueError("oracle diagnostics require a separate output label")
        row_type = "oracle_diagnostic"
    else:
        row_type = "main"
    return {
        "row_type": row_type,
        "output_label": output_label or row_type,
        "ground_truth_shift_alignment": bool(oracle_diagnostic),
        "twin_selection_by_metric": False,
        "phase_sign_selection_by_metric": False,
        "orientation_selection_by_metric": False,
        "restart_selection_metric": "final_fourier_amplitude_residual",
        "restart_selection_uses_ground_truth": False,
        "support_threshold_selection_uses_ground_truth": False,
    }


def refuse_duplicate_output_root(output_root: Path, force: bool = False) -> None:
    if force:
        return
    output_root = Path(output_root)
    if not output_root.exists():
        return
    marker_names = {
        "manifest.json",
        "solver_manifest.json",
        "data_identity_manifest.json",
        "metric_contract_manifest.json",
        "runtime_provenance.json",
        "invocation.json",
    }
    if any((output_root / name).exists() for name in marker_names):
        raise FileExistsError(f"{output_root} already contains benchmark artifacts; pass --force to overwrite")
    if list(output_root.glob("metrics*.json")) or list(output_root.glob("residuals*.json")):
        raise FileExistsError(f"{output_root} already contains benchmark artifacts; pass --force to overwrite")


def _solver_candidate(distribution: str, import_name: str) -> dict[str, Any]:
    version = _package_version(distribution)
    return {
        "distribution": distribution,
        "import_name": import_name,
        "installed_version": version,
        "installed": version is not None,
    }


def write_solver_manifest(
    output_root: Path,
    run_id: str,
    selected_solver: str = SELECTED_SOLVER,
) -> Path:
    output_root = Path(output_root)
    candidates = [
        {
            **_solver_candidate("pynx", "pynx"),
            "source_url": "https://pynx.esrf.fr/en/latest/modules/cdi/index.html",
            "decision": "rejected_for_this_pass",
            "reason": "external CDI-capable package; not adopted because the approved plan requires bounded study-local HIO/ER unless install/runtime provenance is frozen first",
        },
        {
            **_solver_candidate("cdiutils", "cdiutils"),
            "source_url": "https://pypi.org/project/cdiutils/",
            "decision": "rejected_for_this_pass",
            "reason": "external package not installed in the current environment and not selected by the approved solver-discovery gate",
        },
        {
            "name": "tike",
            "source": "local_or_optional",
            "decision": "rejected_for_this_pass",
            "reason": "ptychographic multi-frame orientation, not a bounded single-frame support-constrained CDI HIO/ER baseline",
        },
        {
            "name": "PtyChi",
            "source": "local_or_optional",
            "decision": "rejected_for_this_pass",
            "reason": "ptychographic multi-frame orientation, not a bounded single-frame support-constrained CDI HIO/ER baseline",
        },
        {
            "name": SELECTED_SOLVER,
            "source": _repo_relative(SCRIPT_RELATIVE),
            "decision": "selected",
            "reason": "study-local support-constrained HIO/ER implementation with explicit restart and support policy",
        },
    ]
    payload = {
        "run_id": run_id,
        "selected_solver": selected_solver,
        "selection_date_utc": datetime.now(timezone.utc).isoformat(),
        "search_scope": ["repo", "environment", "PyPI", "GitHub", "web"],
        "external_solver_adopted": False,
        "candidates": candidates,
    }
    return _write_json(output_root / "solver_manifest.json", payload)


def _default_table2_artifact_paths() -> list[Path]:
    base = REPO_ROOT / ".artifacts" / "sim_lines_4x_metrics_2026-01-27"
    return [
        base / "gs1_custom" / "metrics.json",
        base / "gs1_ideal" / "metrics.json",
        base / "gs2_custom" / "metrics.json",
        base / "gs2_ideal" / "metrics.json",
        base / "gs2_ideal_nll" / "metrics.json",
        REPO_ROOT.parent / "ptychopinnpaper2" / "data" / "sim_lines_4x_metrics.json",
        REPO_ROOT.parent / "ptychopinnpaper2" / "tables" / "scripts" / "generate_sim_lines_4x_metrics.py",
    ]


def write_data_identity_manifest(
    output_root: Path,
    branch: str,
    artifact_paths: Sequence[Path | str] | None = None,
    run_id: str | None = None,
    data_generation_control: str = "loader-compatible",
) -> Path:
    if branch not in {"frozen-artifact", "same-split-rerun"}:
        raise ValueError(f"unknown data identity branch: {branch}")
    paths = [Path(item) for item in (_default_table2_artifact_paths() if artifact_paths is None else artifact_paths)]
    records = [_file_record(path) for path in paths]
    missing = [record["path"] for record in records if not record["exists"]]
    exact_inputs_available = branch == "same-split-rerun" and not missing
    payload = {
        "run_id": run_id,
        "branch": branch,
        "data_generation_control": data_generation_control,
        "records": records,
        "missing_records": missing,
        "same_data_comparator_allowed": bool(exact_inputs_available),
        "decision": (
            "same_split_rerun_required_for_table2_claim"
            if branch == "same-split-rerun"
            else "frozen_artifact_insufficient_exact_inputs"
        ),
        "provenance_mismatch_annotations": [
            {
                "id": "frozen_artifact_missing_exact_arrays",
                "status": "open" if branch == "frozen-artifact" else "superseded_by_rerun",
                "note": "Historical Table 2 metric artifacts do not prove the exact diffraction/object split needed for a same-data HIO/ER comparator.",
            },
            {
                "id": "gs2_ideal_nll_name_mismatch",
                "status": "open",
                "note": "Paper-side README references gs2_ideal_nll, but the inspected artifact tree only contained gs2_ideal metrics.",
            },
        ],
    }
    return _write_json(Path(output_root) / "data_identity_manifest.json", payload)


def write_metric_contract_manifest(output_root: Path, mode: str, run_id: str | None = None) -> Path:
    if mode not in {"direct-stitch", "align-for-evaluation", "unresolved"}:
        raise ValueError(f"unknown metric contract mode: {mode}")
    payload = {
        "run_id": run_id,
        "mode": mode,
        "table2_compatibility": "unresolved" if mode == "unresolved" else "mode_explicitly_selected",
        "main_row_policy": "direct_support_anchored_no_ground_truth_shift_twin_or_orientation_selection",
        "evaluation_arguments": {
            "phase_align_method": "plane",
            "frc_sigma": 0,
            "ms_ssim_sigma": 1.0,
            "single_image_frc": False,
        },
        "compatibility_annotations": [
            {
                "id": "metric_contract_current_workflow",
                "status": "selected" if mode == "direct-stitch" else "informational",
                "note": "Current grid-lines workflow uses direct stitching followed by eval_reconstruction.",
            },
            {
                "id": "metric_contract_paper_json_alignment_note",
                "status": "unresolved" if mode == "unresolved" else "reviewed",
                "note": "Paper-side historical JSON records align_for_evaluation provenance; do not silently compare if this remains unresolved.",
            },
            {
                "id": "table_script_subsample_policy",
                "status": "unresolved" if mode == "unresolved" else "reviewed",
                "note": "Historical table script uses a fixed subsample policy for reported rows; fresh CDI rows must label any deviation.",
            },
        ],
    }
    if mode == "unresolved":
        payload["promotion_policy"] = "exploratory_only_until_metric_contract_is_resolved"
    return _write_json(Path(output_root) / "metric_contract_manifest.json", payload)


def write_benchmark_manifest(
    output_root: Path,
    run_id: str,
    solver_manifest: Path,
    data_identity_manifest: Path,
    metric_contract_manifest: Path,
    preflight_only: bool,
    support_manifest: Path | None = None,
    runtime_provenance: Path | None = None,
    metrics_paths: Sequence[Path] | None = None,
    residual_paths: Sequence[Path] | None = None,
    recon_paths: Sequence[Path] | None = None,
    smoke: bool = False,
) -> Path:
    payload = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(SCRIPT_RELATIVE),
        "git_commit": _git_commit(),
        "solver_manifest": str(solver_manifest),
        "data_identity_manifest": str(data_identity_manifest),
        "metric_contract_manifest": str(metric_contract_manifest),
        "support_manifest": str(support_manifest) if support_manifest else None,
        "runtime_provenance": str(runtime_provenance) if runtime_provenance else None,
        "preflight_only": bool(preflight_only),
        "smoke": bool(smoke),
        "metrics_paths": [str(path) for path in metrics_paths or []],
        "residual_paths": [str(path) for path in residual_paths or []],
        "recon_paths": [str(path) for path in recon_paths or []],
    }
    return _write_json(Path(output_root) / "manifest.json", payload)


def _capture_runtime_provenance(output_root: Path, args: argparse.Namespace) -> Path:
    from scripts.studies.invocation_logging import capture_runtime_provenance

    payload = capture_runtime_provenance()
    payload.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "pid": os.getpid(),
            "package_versions": {
                "numpy": np.__version__,
                "scipy": _package_version("scipy"),
                "tensorflow": _package_version("tensorflow"),
                "torch": _package_version("torch"),
                "scikit-image": _package_version("scikit-image"),
            },
            "determinism_environment": {
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
                "PTYCHO_DISABLE_MEMOIZE": os.environ.get("PTYCHO_DISABLE_MEMOIZE"),
            },
            "run_id": args.run_id,
        }
    )
    return _write_json(Path(output_root) / "runtime_provenance.json", payload)


def _write_invocation(output_root: Path, argv: Sequence[str], args: argparse.Namespace) -> tuple[Path, Path]:
    from scripts.studies.invocation_logging import write_invocation_artifacts

    return write_invocation_artifacts(
        output_dir=Path(output_root),
        script_path=str(SCRIPT_RELATIVE),
        argv=argv,
        parsed_args=vars(args),
    )


def _load_probe_for_args(args: argparse.Namespace) -> tuple[np.ndarray, str, list[dict[str, Any]], dict[str, Any]]:
    from ptycho.workflows.grid_lines_workflow import (
        apply_probe_mask,
        apply_probe_transform_pipeline,
        load_ideal_disk_probe,
        load_probe_guess,
        normalize_probe_transform_pipeline,
    )

    if args.probe_source == "custom":
        raw_probe = load_probe_guess(Path(args.probe_npz))
        source_path = _file_record(Path(args.probe_npz))
    else:
        raw_probe = load_ideal_disk_probe(TARGET_N)
        source_path = {"path": "ptycho.probe.get_default_probe", "exists": True}

    pipeline, steps = normalize_probe_transform_pipeline(
        target_N=TARGET_N,
        probe_shape=tuple(np.asarray(raw_probe).shape),
        probe_scale_mode=args.probe_scale_mode,
        probe_smoothing_sigma=args.probe_smoothing_sigma,
        probe_transform_pipeline=args.probe_transform_pipeline,
    )
    probe = apply_probe_transform_pipeline(raw_probe, steps)
    probe = apply_probe_mask(probe, None)
    source_path.update(
        {
            "probe_source": args.probe_source,
            "raw_shape": list(np.asarray(raw_probe).shape),
            "transformed_shape": list(np.asarray(probe).shape),
            "transformed_sha256": _array_sha256(np.asarray(probe)),
        }
    )
    return probe.astype(np.complex64), pipeline, steps, source_path


def _write_support_manifest(
    output_root: Path,
    args: argparse.Namespace,
    probe: np.ndarray,
    probe_transform_pipeline: str,
    probe_transform_steps: Sequence[dict[str, Any]],
    probe_record: dict[str, Any],
) -> tuple[Path, np.ndarray, dict[str, Any]]:
    threshold_grid = [float(item) for item in args.support_thresholds]
    support_records: list[dict[str, Any]] = []
    primary_support: np.ndarray | None = None
    primary_record: dict[str, Any] | None = None
    for threshold in threshold_grid:
        try:
            support, record = make_probe_support(probe, threshold=threshold, threshold_grid=threshold_grid)
        except ValueError as exc:
            support_records.append(
                {
                    "support_threshold": float(threshold),
                    "status": "invalid",
                    "reason": str(exc),
                }
            )
            continue
        record["status"] = "selected_primary" if threshold == float(args.primary_support_threshold) else "valid_sensitivity"
        support_records.append(record)
        if threshold == float(args.primary_support_threshold):
            primary_support = support
            primary_record = record
    if primary_support is None or primary_record is None:
        raise ValueError("primary support threshold did not produce a valid support mask")

    payload = {
        "run_id": args.run_id,
        "primary_support_threshold": float(args.primary_support_threshold),
        "threshold_grid": threshold_grid,
        "probe_record": probe_record,
        "probe_transform_pipeline": probe_transform_pipeline,
        "probe_transform_steps": list(probe_transform_steps),
        "support_records": support_records,
        "primary_support_record": primary_record,
    }
    return _write_json(Path(output_root) / "support_manifest.json", payload), primary_support, primary_record


def _condition_label(args: argparse.Namespace) -> str:
    probe_label = "custom" if args.probe_source == "custom" else "ideal"
    return f"{DEFAULT_CONDITION}_{probe_label}"


def _threshold_token(threshold: float) -> str:
    token = f"{float(threshold):.6g}".replace("-", "m").replace(".", "p")
    return token


def _frame_amplitude(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"expected 2D diffraction amplitude frame, got shape {array.shape}")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("diffraction amplitude frame contains non-finite values")
    return np.maximum(array, 0.0)


def _patch_object(y_i: np.ndarray, y_phi: np.ndarray) -> np.ndarray:
    amp = np.asarray(y_i)
    phase = np.asarray(y_phi)
    if amp.ndim == 3:
        amp = amp[..., 0]
    if phase.ndim == 3:
        phase = phase[..., 0]
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def _bounded_frame_count(total: int, requested: int | None, nimgs: int) -> int:
    if total <= 0:
        raise ValueError("no test frames available")
    limit = total if requested is None or requested <= 0 else min(total, int(requested))
    candidates = [nimgs * (segments**2) for segments in range(1, 256) if nimgs * (segments**2) <= limit]
    if candidates:
        return max(candidates)
    return min(total, nimgs)


def _generate_table2_smoke_data(args: argparse.Namespace, probe: np.ndarray) -> tuple[Any, dict[str, Any]]:
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig, simulate_grid_data

    cfg = GridLinesConfig(
        N=TARGET_N,
        gridsize=1,
        output_dir=Path(args.output_root),
        probe_npz=Path(args.probe_npz),
        size=DEFAULT_TABLE2_SIZE,
        offset=DEFAULT_TABLE2_OFFSET,
        outer_offset_train=DEFAULT_OUTER_OFFSET_TRAIN,
        outer_offset_test=DEFAULT_OUTER_OFFSET_TEST,
        nimgs_train=DEFAULT_NIMGS_TRAIN,
        nimgs_test=DEFAULT_NIMGS_TEST,
        nphotons=DEFAULT_NPHOTONS,
        probe_smoothing_sigma=float(args.probe_smoothing_sigma),
        probe_source=args.probe_source,
        probe_scale_mode=args.probe_scale_mode,
        probe_transform_pipeline=args.probe_transform_pipeline,
    )
    return cfg, simulate_grid_data(cfg, probe)


def _nonfinite_label(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if value > 0:
        return "inf"
    return "-inf"


def _metric_value_jsonable(
    value: Any,
    path: str,
    annotations: list[dict[str, Any]],
) -> Any:
    if isinstance(value, np.ndarray):
        return [
            _metric_value_jsonable(item, f"{path}.{index}", annotations)
            for index, item in enumerate(value.tolist())
        ]
    if isinstance(value, (list, tuple)):
        return [
            _metric_value_jsonable(item, f"{path}.{index}", annotations)
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.generic):
        return _metric_value_jsonable(value.item(), path, annotations)
    if isinstance(value, (float, int)):
        item = float(value)
        if not math.isfinite(item):
            annotations.append(
                {
                    "metric": path,
                    "value": _nonfinite_label(item),
                    "stored_as": None,
                }
            )
            return None
        return item
    return _jsonable(value)


def _metrics_jsonable(metrics: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload: dict[str, Any] = {}
    annotations: list[dict[str, Any]] = []
    for key, value in metrics.items():
        if key == "frc":
            payload[key] = "omitted_curve_arrays_from_smoke_json"
        else:
            payload[key] = _metric_value_jsonable(value, key, annotations)
    return payload, annotations


def run_smoke_benchmark(
    output_root: Path,
    args: argparse.Namespace,
    probe: np.ndarray,
    support: np.ndarray,
    support_record: dict[str, Any],
) -> tuple[list[Path], list[Path], list[Path]]:
    from ptycho.evaluation import eval_reconstruction
    from ptycho.workflows.grid_lines_workflow import save_recon_artifact, stitch_predictions

    start = time.perf_counter()
    cfg, data = _generate_table2_smoke_data(args, probe)
    x_test = np.asarray(data["test"]["X"])
    y_i_test = np.asarray(data["test"]["Y_I"])
    y_phi_test = np.asarray(data["test"]["Y_phi"])
    count = _bounded_frame_count(x_test.shape[0], args.max_test_frames, cfg.nimgs_test)

    reconstructed_patches: list[np.ndarray] = []
    ground_truth_patches: list[np.ndarray] = []
    self_consistency_checks: list[dict[str, Any]] = []
    residual_payload: dict[str, Any] = {
        "run_id": args.run_id,
        "condition": _condition_label(args),
        "support_threshold": float(args.primary_support_threshold),
        "restart_seeds": [int(seed) for seed in args.restart_seeds],
        "patches": [],
    }

    for index in range(count):
        target = _frame_amplitude(x_test[index])
        restart_run = run_restarts(
            target,
            support,
            seeds=args.restart_seeds,
            beta=float(args.beta),
            hio_iters=int(args.hio_iters),
            er_iters=int(args.er_iters),
            residual_period=max(1, int(args.residual_period)),
        )
        reconstructed = recover_object_patch(
            restart_run.selected.psi,
            probe,
            support,
            epsilon_ratio=DEFAULT_EPSILON_RATIO,
        )
        recomputed_residual = fourier_residual(restart_run.selected.psi, target)
        self_consistency_checks.append(
            {
                "patch_index": index,
                "selected_seed": restart_run.selected.seed,
                "recorded_final_residual": restart_run.selected.final_residual,
                "recomputed_final_residual": recomputed_residual,
                "consistent": math.isclose(
                    restart_run.selected.final_residual,
                    recomputed_residual,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                and math.isfinite(recomputed_residual),
            }
        )
        reconstructed_patches.append(reconstructed[..., None])
        ground_truth_patches.append(_patch_object(y_i_test[index], y_phi_test[index])[..., None])
        residual_payload["patches"].append(
            {
                "patch_index": index,
                "selected_seed": restart_run.selected.seed,
                "selected_final_residual": restart_run.selected.final_residual,
                "restart_final_residuals": [
                    {"seed": item.seed, "final_residual": item.final_residual}
                    for item in restart_run.restarts
                ],
                "selected_residual_curve": restart_run.selected.residual_curve,
            }
        )

    recon_array = np.asarray(reconstructed_patches, dtype=np.complex64)
    gt_patch_array = np.asarray(ground_truth_patches, dtype=np.complex64)
    norm_y_i = float(data["test"]["norm_Y_I"])
    stitched = stitch_predictions(recon_array, norm_y_i, part="complex")
    gt_stitched = stitch_predictions(gt_patch_array, norm_y_i, part="complex")

    condition_label = _condition_label(args)
    row_label = f"{condition_label}_support_{_threshold_token(args.primary_support_threshold)}"
    recon_path = save_recon_artifact(Path(output_root), row_label, stitched)
    residual_path = _write_json(Path(output_root) / f"residuals_{row_label}.json", residual_payload)

    metrics_payload: dict[str, Any] = {
        "run_id": args.run_id,
        "condition": condition_label,
        "row_label": row_label,
        "smoke": bool(args.smoke),
        "n_test_frames": int(count),
        "hio_hyperparameters": {
            "beta": float(args.beta),
            "hio_iters": int(args.hio_iters),
            "er_iters": int(args.er_iters),
            "restart_seeds": [int(seed) for seed in args.restart_seeds],
            "restart_selection": "final_fourier_amplitude_residual",
        },
        "support_policy": support_record,
        "ambiguity_policy": build_ambiguity_policy(False),
        "forward_amplitude_self_consistency": {
            "status": "ok" if all(item["consistent"] for item in self_consistency_checks) else "failed",
            "checks": self_consistency_checks,
        },
        "timing_seconds": time.perf_counter() - start,
        "recon_npz": str(recon_path),
        "residual_json": str(residual_path),
    }
    try:
        metrics = eval_reconstruction(
            stitched[:1],
            gt_stitched[0],
            label=row_label,
            phase_align_method="plane",
            frc_sigma=0,
            debug_save_images=False,
            ms_ssim_sigma=1.0,
            single_image_frc=False,
        )
    except Exception as exc:
        metrics_payload["eval_status"] = "failed"
        metrics_payload["eval_error"] = repr(exc)
    else:
        metrics_payload["eval_status"] = "ok"
        metrics_json, nonfinite_annotations = _metrics_jsonable(metrics)
        metrics_payload["metrics"] = metrics_json
        if nonfinite_annotations:
            metrics_payload["metric_nonfinite_annotations"] = nonfinite_annotations

    metrics_path = _write_json(Path(output_root) / f"metrics_{row_label}.json", metrics_payload)
    return [metrics_path], [residual_path], [recon_path]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--probe-npz", required=True, type=Path)
    parser.add_argument("--probe-source", required=True, choices=["custom", "ideal_disk"])
    parser.add_argument(
        "--probe-scale-mode",
        required=True,
        choices=["pad_preserve", "pad_extrapolate", "interpolate", "pipeline"],
    )
    parser.add_argument("--probe-transform-pipeline", default=None)
    parser.add_argument("--probe-smoothing-sigma", required=True, type=float)
    parser.add_argument("--support-thresholds", required=True, nargs="+", type=float)
    parser.add_argument("--primary-support-threshold", required=True, type=float)
    parser.add_argument("--restart-seeds", required=True, nargs="+", type=int)
    parser.add_argument("--beta", default=0.9, type=float)
    parser.add_argument("--hio-iters", default=1000, type=int)
    parser.add_argument("--er-iters", default=200, type=int)
    parser.add_argument("--residual-period", default=50, type=int)
    parser.add_argument("--max-test-frames", default=0, type=int)
    parser.add_argument(
        "--data-identity-branch",
        required=True,
        choices=["frozen-artifact", "same-split-rerun"],
    )
    parser.add_argument(
        "--data-generation-control",
        default="loader-compatible",
        choices=["loader-compatible", "study-local-seeded"],
    )
    parser.add_argument(
        "--metric-contract-mode",
        required=True,
        choices=["direct-stitch", "align-for-evaluation", "unresolved"],
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    output_root = Path(args.output_root)
    refuse_duplicate_output_root(output_root, force=bool(args.force))
    output_root.mkdir(parents=True, exist_ok=True)

    runtime_path = _capture_runtime_provenance(output_root, args)
    _write_invocation(output_root, raw_argv, args)
    solver_manifest = write_solver_manifest(output_root, run_id=args.run_id, selected_solver=SELECTED_SOLVER)
    data_identity_manifest = write_data_identity_manifest(
        output_root,
        branch=args.data_identity_branch,
        run_id=args.run_id,
        data_generation_control=args.data_generation_control,
    )
    metric_contract_manifest = write_metric_contract_manifest(
        output_root,
        mode=args.metric_contract_mode,
        run_id=args.run_id,
    )

    probe, probe_pipeline, probe_steps, probe_record = _load_probe_for_args(args)
    support_manifest, support, support_record = _write_support_manifest(
        output_root,
        args,
        probe,
        probe_pipeline,
        probe_steps,
        probe_record,
    )

    metrics_paths: list[Path] = []
    residual_paths: list[Path] = []
    recon_paths: list[Path] = []
    if args.preflight_only:
        write_benchmark_manifest(
            output_root,
            run_id=args.run_id,
            solver_manifest=solver_manifest,
            data_identity_manifest=data_identity_manifest,
            metric_contract_manifest=metric_contract_manifest,
            support_manifest=support_manifest,
            runtime_provenance=runtime_path,
            preflight_only=True,
            smoke=bool(args.smoke),
        )
        print(f"preflight complete: {output_root}")
        return 0

    metrics_paths, residual_paths, recon_paths = run_smoke_benchmark(
        output_root,
        args,
        probe,
        support,
        support_record,
    )
    write_benchmark_manifest(
        output_root,
        run_id=args.run_id,
        solver_manifest=solver_manifest,
        data_identity_manifest=data_identity_manifest,
        metric_contract_manifest=metric_contract_manifest,
        support_manifest=support_manifest,
        runtime_provenance=runtime_path,
        preflight_only=False,
        smoke=bool(args.smoke),
        metrics_paths=metrics_paths,
        residual_paths=residual_paths,
        recon_paths=recon_paths,
    )
    print(f"benchmark complete: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

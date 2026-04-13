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
    base_seed: int | None = None
    restart_index: int | None = None


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
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _array_record(
    canonical_key: str,
    value: Any,
    *,
    split: str | None = None,
    npz_key: str | None = None,
) -> dict[str, Any]:
    if value is None:
        return {
            "split": split,
            "canonical_key": canonical_key,
            "npz_key": npz_key or canonical_key,
            "exists": False,
            "shape": None,
            "dtype": None,
            "sha256": None,
        }
    array = np.asarray(value)
    return {
        "split": split,
        "canonical_key": canonical_key,
        "npz_key": npz_key or canonical_key,
        "exists": True,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _array_sha256(array),
    }


def _stable_int_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


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
    denominator = max(float(np.linalg.norm(target)), 1e-12)
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
    condition_id: str | None = None,
    patch_index: int | None = None,
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
    for restart_index, seed in enumerate(seeds):
        base_seed = int(seed)
        actual_seed = (
            _stable_int_seed(condition_id, int(patch_index), restart_index, base_seed)
            if condition_id is not None and patch_index is not None
            else base_seed
        )
        psi = _initial_psi(target, actual_seed)
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
            raise FloatingPointError(f"non-finite Fourier residual for restart seed {actual_seed}")
        results.append(
            RestartResult(
                seed=int(actual_seed),
                psi=np.asarray(psi),
                final_residual=float(final_residual),
                residual_curve=[float(item) for item in curve],
                base_seed=base_seed,
                restart_index=int(restart_index),
            )
        )

    selected = select_restart_by_residual(results)
    return RestartRun(restarts=results, selected=selected)


def _restart_records(restart_run: RestartRun) -> list[dict[str, Any]]:
    selected_seed = int(restart_run.selected.seed)
    return [
        {
            "restart_index": item.restart_index,
            "base_seed": item.base_seed,
            "seed": int(item.seed),
            "selected": int(item.seed) == selected_seed,
            "final_residual": float(item.final_residual),
            "residual_curve": [float(value) for value in item.residual_curve],
            "metrics": {
                "final_fourier_amplitude_residual": float(item.final_residual),
                "n_residual_samples": len(item.residual_curve),
            },
        }
        for item in restart_run.restarts
    ]


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


def write_solver_manifest(
    output_root: Path,
    run_id: str,
    selected_solver: str = SELECTED_SOLVER,
) -> Path:
    output_root = Path(output_root)
    search_date = datetime.now(timezone.utc).date().isoformat()
    candidates = [
        {
            "name": "PyNX",
            "source_url": "https://pynx.esrf.fr/en/latest/modules/cdi/index.html",
            "package_version": _package_version("pynx"),
            "license": "not_frozen_in_repo_manifest",
            "install_command": "conda env create --file https://gitlab.esrf.fr/favre/PyNX/-/raw/master/conda-environment.yaml",
            "api_entry_point": "pynx.cdi.CDI with HIO/ER operators",
            "accepted": False,
            "searched_sources": ["PyPI", "GitHub", "web"],
            "installed": _package_version("pynx") is not None,
            "evidence": [
                {
                    "source_url": "https://pynx.esrf.fr/en/latest/modules/cdi/index.html",
                    "note": "Public documentation describes 2D/3D CDI support with HIO and ER algorithms.",
                }
            ],
            "reason": "capable external CDI package, but not adopted in this bounded pass because install/license/runtime provenance was not frozen in the local environment",
        },
        {
            "name": "CDIutils",
            "source_url": "https://pypi.org/project/cdiutils/",
            "package_version": _package_version("cdiutils"),
            "license": "MIT",
            "install_command": "pip install cdiutils; install PyNX separately for phase retrieval",
            "api_entry_point": "cdiutils pipeline helpers; phase retrieval delegated to PyNX",
            "accepted": False,
            "searched_sources": ["PyPI", "GitHub", "web"],
            "installed": _package_version("cdiutils") is not None,
            "evidence": [
                {
                    "source_url": "https://pypi.org/project/cdiutils/",
                    "note": "PyPI page states CDIutils uses PyNX for phase retrieval and requires separate PyNX installation.",
                },
                {
                    "source_url": "https://github.com/clatlan/cdiutils",
                    "note": "GitHub repository reports MIT license.",
                },
            ],
            "reason": "not a direct bounded single-frame HIO/ER API for this pass; delegates phase retrieval to PyNX and requires a broader environment setup",
        },
        {
            "name": "phastphase",
            "source_url": "https://phastphase.readthedocs.io/en/latest/",
            "package_version": _package_version("phastphase"),
            "license": "MIT",
            "install_command": "pip install phastphase",
            "api_entry_point": "phastphase.retrieve",
            "accepted": False,
            "searched_sources": ["PyPI", "GitHub", "web"],
            "installed": _package_version("phastphase") is not None,
            "evidence": [
                {
                    "source_url": "https://phastphase.readthedocs.io/en/latest/",
                    "note": "Documentation describes support-constrained phase retrieval for near-Schwarz objects.",
                },
                {
                    "source_url": "https://github.com/cbrabes/phastphase",
                    "note": "GitHub repository reports MIT license and GPU-accelerated support-constrained phase retrieval.",
                },
            ],
            "reason": "not selected because it is a specialized near-Schwarz-object solver rather than the pre-registered HIO/ER support-constrained CDI baseline",
        },
        {
            "name": "Tike",
            "source_url": "local package import tike; scripts/reconstruction/run_tike_reconstruction.py",
            "package_version": _package_version("tike"),
            "license": "not_recorded_here",
            "install_command": "already installed in current environment" if _package_version("tike") else "pip/conda install tike",
            "api_entry_point": "tike.ptycho.reconstruct",
            "accepted": False,
            "searched_sources": ["repo", "environment"],
            "installed": _package_version("tike") is not None,
            "reason": "ptychographic multi-frame orientation, not a bounded single-frame support-constrained CDI HIO/ER baseline",
        },
        {
            "name": "PtyChi",
            "source_url": "local package import ptychi; scripts/reconstruction/ptychi_reconstruct_tike.py",
            "package_version": _package_version("ptychi"),
            "license": "not_recorded_here",
            "install_command": "already installed in current environment" if _package_version("ptychi") else "pip/conda install ptychi",
            "api_entry_point": "ptychi ptychographic reconstruction APIs",
            "accepted": False,
            "searched_sources": ["repo", "environment"],
            "installed": _package_version("ptychi") is not None,
            "reason": "ptychographic multi-frame orientation, not a bounded single-frame support-constrained CDI HIO/ER baseline",
        },
        {
            "name": SELECTED_SOLVER,
            "source_url": _repo_relative(SCRIPT_RELATIVE),
            "package_version": None,
            "license": "repository_local",
            "install_command": "none",
            "api_entry_point": "run_restarts / hio_update / er_cleanup",
            "accepted": True,
            "searched_sources": ["repo"],
            "installed": True,
            "reason": "study-local support-constrained HIO/ER implementation with explicit restart and support policy",
        },
    ]
    payload = {
        "run_id": run_id,
        "search_date": search_date,
        "searched_sources": ["repo", "environment", "PyPI", "GitHub", "web"],
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


def _data_generation_control_status(data_generation_control: str) -> str:
    if data_generation_control == "loader-compatible":
        return "implemented_loader_compatible"
    if data_generation_control == "study-local-seeded":
        return "unimplemented_for_metric_runs"
    raise ValueError(f"unknown data generation control: {data_generation_control}")


def write_data_identity_manifest(
    output_root: Path,
    branch: str,
    artifact_paths: Sequence[Path | str] | None = None,
    run_id: str | None = None,
    data_generation_control: str = "loader-compatible",
) -> Path:
    if branch not in {"frozen-artifact", "same-split-rerun"}:
        raise ValueError(f"unknown data identity branch: {branch}")
    control_status = _data_generation_control_status(data_generation_control)
    paths = [Path(item) for item in (_default_table2_artifact_paths() if artifact_paths is None else artifact_paths)]
    records = [_file_record(path) for path in paths]
    missing = [record["path"] for record in records if not record["exists"]]
    metric_inspection_allowed = False
    old_table2_same_data_allowed = False
    if branch == "same-split-rerun" and control_status != "implemented_loader_compatible":
        decision = f"{data_generation_control}_data_generation_control_unimplemented_metric_run_blocked"
    elif branch == "same-split-rerun":
        decision = "same_split_rerun_bundle_not_frozen"
    else:
        decision = "frozen_artifact_insufficient_exact_inputs"
    payload = {
        "run_id": run_id,
        "branch": branch,
        "data_generation_control": data_generation_control,
        "data_generation_control_status": control_status,
        "records": records,
        "missing_records": missing,
        "metric_inspection_allowed": bool(metric_inspection_allowed),
        "same_data_comparator_allowed": bool(old_table2_same_data_allowed),
        "old_table2_same_data_comparator_allowed": bool(old_table2_same_data_allowed),
        "old_table2_value_policy": "historical_context_only",
        "hio_metric_policy": (
            "blocked_until_same_split_generated_bundle_is_frozen"
            if branch == "same-split-rerun"
            else "blocked_until_exact_frozen_table2_inputs_are_located"
        ),
        "decision": decision,
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
    table2_compatibility = {
        "unresolved": "unresolved",
        "direct-stitch": "fresh_same_split_direct_stitch_not_historical_table2",
        "align-for-evaluation": "unimplemented_in_this_pass",
    }[mode]
    payload = {
        "run_id": run_id,
        "mode": mode,
        "table2_compatibility": table2_compatibility,
        "metric_inspection_allowed": mode == "direct-stitch",
        "table2_compatible": False,
        "main_row_policy": "direct_support_anchored_no_ground_truth_shift_twin_or_orientation_selection",
        "reconstruction_to_ground_truth_preparation": {
            "selected_path": mode,
            "direct_stitch_function": "ptycho.workflows.grid_lines_workflow.stitch_predictions",
            "global_offsets_source": None,
            "global_offsets_shape_checksum": None,
            "stitch_patch_size": None,
            "notes": "Direct-stitch mode is a fresh rerun/exploratory contract until the paper-side alignment/subsample notes are resolved.",
        },
        "metric_subset_policy": {
            "policy": "full_generated_smoke_or_test_subset",
            "nsamples": None,
            "seed": None,
            "rng_implementation": None,
            "sampling_population": "generated test frames bounded by --max-test-frames when supplied",
            "selected_indices_checksum": None,
            "reason": "full generated smoke/test subset, not paper nsamples=1000 subsample",
        },
        "evaluation_arguments": {
            "phase_align_method": "plane",
            "frc_sigma": 0,
            "ms_ssim_sigma": 1.0,
            "single_image_frc": False,
            "debug_save_images": False,
            "amplitude_mean_scaling": "eval_reconstruction internal mean scaling",
        },
        "paper_side_note_decisions": [
            {
                "id": "paper_json_align_for_evaluation",
                "decision": "unresolved",
                "evidence": {
                    "path": "/home/ollie/Documents/ptychopinnpaper2/data/sim_lines_4x_metrics.json",
                    "line_reference": "L140-L149",
                },
                "note": "Paper data JSON records align_for_evaluation/global_offsets/stitch_patch_size=20 provenance that is not yet reconciled with the current direct grid-lines workflow.",
            },
            {
                "id": "table_script_subsample_1000_seed_7",
                "decision": "unresolved",
                "evidence": {
                    "path": "/home/ollie/Documents/ptychopinnpaper2/tables/scripts/generate_sim_lines_4x_metrics.py",
                    "line_reference": "L36-L41",
                },
                "note": "Table script/caption records nsamples=1000 seed=7, but this study script currently uses the generated smoke/full subset directly.",
            },
            {
                "id": "current_grid_lines_direct_stitch_workflow",
                "decision": "authoritative_for_fresh_rerun_only",
                "evidence": {
                    "path": "ptycho/workflows/grid_lines_workflow.py",
                    "line_reference": "L1489-L1496",
                },
                "note": "Current grid-lines workflow stitches predictions and calls eval_reconstruction directly.",
            },
        ],
        "deviation": {
            "table2_compatible": False,
            "result_role": "fresh_same_split_exploratory_or_rerun_comparator",
            "explanation": "The historical Table 2 crop/alignment and subsample notes remain unresolved; metrics from this script must not be merged into old Table 2 rows as same-data results.",
        },
        "compatibility_annotations": [
            {
                "id": "metric_contract_current_workflow",
                "status": "fresh_rerun_selected" if mode == "direct-stitch" else "informational",
                "note": "Current grid-lines workflow uses direct stitching followed by eval_reconstruction.",
            },
            {
                "id": "metric_contract_paper_json_alignment_note",
                "status": "unresolved",
                "note": "Paper-side historical JSON records align_for_evaluation provenance; do not silently compare if this remains unresolved.",
            },
            {
                "id": "table_script_subsample_policy",
                "status": "unresolved",
                "note": "Historical table script uses a fixed subsample policy for reported rows; fresh CDI rows must label any deviation.",
            },
        ],
    }
    if mode == "unresolved":
        payload["promotion_policy"] = "exploratory_only_until_metric_contract_is_resolved"
    return _write_json(Path(output_root) / "metric_contract_manifest.json", payload)


def write_pinn_randomness_manifest(
    output_root: Path,
    run_id: str,
    data_generation_control: str,
    metric_contract_manifest: Path,
    data_bundle_manifest: Path | None = None,
    primary_training_seed: int = 2026041211,
) -> Path:
    control_status = _data_generation_control_status(data_generation_control)
    payload = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pinn_comparator_status": "not_run_in_this_pass",
        "manifest_purpose": "pre_register_same_split_ptychopinn_seed_policy_before_model_construction",
        "data_generation_control": data_generation_control,
        "data_generation_control_status": control_status,
        "loader_compatible_data_seeds": {"train": 1, "test": 2}
        if data_generation_control == "loader-compatible"
        else None,
        "data_generation_seed": None,
        "primary_training_seed": int(primary_training_seed),
        "stochastic_fallback_seeds": [2026041211, 2026041212, 2026041213],
        "determinism_mode": "deterministic_primary_seed_pre_registered",
        "determinism_status": "not_validated_no_model_constructed",
        "fresh_process_requirements": {
            "PYTHONHASHSEED": str(primary_training_seed),
            "TF_DETERMINISTIC_OPS": "1 when supported",
        },
        "seed_api_policy": {
            "python_random": int(primary_training_seed),
            "numpy": int(primary_training_seed),
            "tensorflow": int(primary_training_seed),
            "preferred_tf_api": "tf.keras.utils.set_random_seed plus tf.config.experimental.enable_op_determinism when available",
        },
        "training_recipe": {
            "nepochs": 60,
            "batch_size": 16,
            "shuffle_control": "must be recorded by the future comparator run before model.fit",
            "if_uncontrolled": "switch_to_stochastic_repeated_rerun_mode",
        },
        "model_construction_started": False,
        "training_started": False,
        "initial_model_weight_checksums": None,
        "final_model_or_checkpoint_checksums": None,
        "training_history": None,
        "metric_contract_manifest": str(metric_contract_manifest),
        "metric_contract_sha256": _sha256_file(Path(metric_contract_manifest)),
        "data_bundle_manifest": str(data_bundle_manifest) if data_bundle_manifest else None,
        "data_bundle_sha256": _sha256_file(Path(data_bundle_manifest)) if data_bundle_manifest else None,
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "env": {
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
                "PTYCHO_DISABLE_MEMOIZE": os.environ.get("PTYCHO_DISABLE_MEMOIZE"),
            },
            "package_versions": {
                "numpy": np.__version__,
                "tensorflow": _package_version("tensorflow"),
                "cuda_runtime": _package_version("nvidia-cuda-runtime-cu12"),
                "cudnn": _package_version("nvidia-cudnn-cu12"),
            },
        },
    }
    return _write_json(Path(output_root) / "pinn_randomness_manifest.json", payload)


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


_SAME_SPLIT_EXPECTED_KEYS = [
    "X",
    "Y_I",
    "Y_phi",
    "YY_full",
    "YY_ground_truth",
    "norm_Y_I",
    "probeGuess",
    "coords_nominal",
    "coords_true",
    "coords_offsets",
]

_NPZ_CANONICAL_KEYS = {"diffraction": "X"}


def _npz_key_records(path: Path, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    present: set[str] = set()
    with np.load(path, allow_pickle=True) as loaded:
        for npz_key in sorted(loaded.files):
            if npz_key == "_metadata":
                continue
            canonical_key = _NPZ_CANONICAL_KEYS.get(npz_key, npz_key)
            if canonical_key not in _SAME_SPLIT_EXPECTED_KEYS:
                continue
            present.add(canonical_key)
            records.append(
                _array_record(
                    canonical_key,
                    loaded[npz_key],
                    split=split,
                    npz_key=npz_key,
                )
            )
    for expected_key in _SAME_SPLIT_EXPECTED_KEYS:
        if expected_key not in present:
            records.append(
                _array_record(
                    expected_key,
                    None,
                    split=split,
                    npz_key="diffraction" if expected_key == "X" else expected_key,
                )
            )
    return records


def _memoization_policy_for_fresh_bundle() -> dict[str, Any]:
    disabled_value = os.environ.get("PTYCHO_DISABLE_MEMOIZE")
    if disabled_value != "1":
        raise RuntimeError(
            "Same-split bundle generation requires PTYCHO_DISABLE_MEMOIZE=1 until cache identity "
            "recording is implemented for this CDI benchmark."
        )
    return {
        "PTYCHO_DISABLE_MEMOIZE": disabled_value,
        "PTYCHO_MEMOIZE_KEY_MODE": os.environ.get("PTYCHO_MEMOIZE_KEY_MODE"),
        "cache_mode": "disabled_fresh_generation",
        "cache_reused": False,
        "cache_key": None,
        "cache_file": None,
        "cache_file_sha256": None,
    }


def persist_same_split_data_bundle(
    output_root: Path,
    run_id: str,
    cfg: Any,
    data: dict[str, Any],
    config: Any,
    probe: np.ndarray,
    probe_transform_pipeline: str,
    probe_transform_steps: Sequence[dict[str, Any]],
    data_generation_control: str = "loader-compatible",
) -> dict[str, Any]:
    if data_generation_control != "loader-compatible":
        raise NotImplementedError("study-local-seeded data generation is not implemented for metric runs")

    from ptycho.workflows.grid_lines_workflow import save_split_npz

    memoization = _memoization_policy_for_fresh_bundle()
    output_root = Path(output_root)
    train_payload = dict(data["train"])
    test_payload = dict(data["test"])
    train_payload["probeGuess"] = np.asarray(probe)
    test_payload["probeGuess"] = np.asarray(probe)

    train_npz = save_split_npz(
        cfg,
        "train",
        train_payload,
        config,
        probe_transform_pipeline=probe_transform_pipeline,
        probe_transform_steps=list(probe_transform_steps),
    )
    test_npz = save_split_npz(
        cfg,
        "test",
        test_payload,
        config,
        probe_transform_pipeline=probe_transform_pipeline,
        probe_transform_steps=list(probe_transform_steps),
    )

    probe_manifest = _write_json(
        output_root / "probe_transform_manifest.json",
        {
            "run_id": run_id,
            "probe_transform_pipeline": probe_transform_pipeline,
            "probe_transform_steps": list(probe_transform_steps),
            "probe_shape": list(np.asarray(probe).shape),
            "probe_dtype": str(np.asarray(probe).dtype),
            "probe_sha256": _array_sha256(np.asarray(probe)),
        },
    )
    data_generation_manifest = _write_json(
        output_root / "data_generation_manifest.json",
        {
            "run_id": run_id,
            "branch": "same-split-rerun",
            "data_generation_control": data_generation_control,
            "data_generation_control_status": _data_generation_control_status(data_generation_control),
            "generator": "ptycho.workflows.grid_lines_workflow.simulate_grid_data",
            "control_branch": "loader-compatible",
            "loader_compatible_data_seeds": {"train": 1, "test": 2},
            "data_generation_seed": None,
            "memoization": memoization,
            "table2_constants": {
                "N": int(cfg.N),
                "gridsize": int(cfg.gridsize),
                "data_source": "lines",
                "size": int(cfg.size),
                "offset": int(cfg.offset),
                "outer_offset_train": int(cfg.outer_offset_train),
                "outer_offset_test": int(cfg.outer_offset_test),
                "nimgs_train": int(cfg.nimgs_train),
                "nimgs_test": int(cfg.nimgs_test),
                "nphotons": float(cfg.nphotons),
                "probe_source": cfg.probe_source,
                "probe_scale_mode": cfg.probe_scale_mode,
                "probe_smoothing_sigma": float(cfg.probe_smoothing_sigma),
                "probe_transform_pipeline": probe_transform_pipeline,
            },
        },
    )

    key_records = _npz_key_records(Path(train_npz), "train") + _npz_key_records(Path(test_npz), "test")
    data_bundle_manifest = _write_json(
        output_root / "data_bundle_manifest.json",
        {
            "run_id": run_id,
            "branch": "same-split-rerun",
            "train_npz": str(train_npz),
            "test_npz": str(test_npz),
            "data_generation_manifest": str(data_generation_manifest),
            "probe_transform_manifest": str(probe_manifest),
            "memoization": memoization,
            "npz_records": {
                "train": _file_record(Path(train_npz)),
                "test": _file_record(Path(test_npz)),
            },
            "key_records": key_records,
        },
    )
    return {
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "data_generation_manifest": str(data_generation_manifest),
        "data_bundle_manifest": str(data_bundle_manifest),
        "probe_transform_manifest": str(probe_manifest),
        "memoization": memoization,
    }


def update_data_identity_manifest_with_generated_bundle(
    data_identity_manifest: Path,
    bundle: dict[str, Any],
) -> Path:
    path = Path(data_identity_manifest)
    payload = _read_manifest(path)
    bundle_manifest = _read_manifest(Path(bundle["data_bundle_manifest"]))
    payload["generated_data_bundle"] = {
        "train_npz": bundle["train_npz"],
        "test_npz": bundle["test_npz"],
        "data_generation_manifest": bundle["data_generation_manifest"],
        "data_bundle_manifest": bundle["data_bundle_manifest"],
        "probe_transform_manifest": bundle["probe_transform_manifest"],
        "memoization": bundle["memoization"],
        "npz_records": bundle_manifest["npz_records"],
    }
    payload["key_level_checksums"] = bundle_manifest["key_records"]
    payload["metric_inspection_allowed"] = (
        payload.get("branch") == "same-split-rerun"
        and payload.get("data_generation_control_status") == "implemented_loader_compatible"
        and bundle["memoization"]["cache_mode"] == "disabled_fresh_generation"
    )
    payload["decision"] = "same_split_generated_bundle_frozen_for_comparator"
    payload["hio_metric_policy"] = "allowed_on_frozen_same_split_generated_bundle_only"
    return _write_json(path, payload)


def assert_metric_gates_allow_metrics(
    data_identity_manifest: Path,
    metric_contract_manifest: Path,
) -> dict[str, Any]:
    data_identity = _read_manifest(Path(data_identity_manifest))
    metric_contract = _read_manifest(Path(metric_contract_manifest))
    if not data_identity.get("metric_inspection_allowed", False):
        raise RuntimeError(
            "Data identity gate blocked metric inspection: "
            f"{data_identity.get('decision')}; use --data-identity-branch same-split-rerun "
            "or locate exact frozen Table 2 inputs first."
        )
    if not metric_contract.get("metric_inspection_allowed", False):
        raise RuntimeError(
            "Metric contract gate blocked metric inspection: "
            f"{metric_contract.get('table2_compatibility')}"
        )
    table2_compatible = bool(
        data_identity.get("old_table2_same_data_comparator_allowed", False)
        and metric_contract.get("table2_compatible", False)
    )
    return {
        "metric_inspection_allowed": True,
        "table2_compatible": table2_compatible,
        "data_identity_decision": data_identity.get("decision"),
        "metric_contract": metric_contract.get("table2_compatibility"),
        "old_table2_value_policy": data_identity.get("old_table2_value_policy"),
    }


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
    data_generation_manifest: Path | None = None,
    data_bundle_manifest: Path | None = None,
    probe_transform_manifest: Path | None = None,
    pinn_randomness_manifest: Path | None = None,
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
        "data_generation_manifest": str(data_generation_manifest) if data_generation_manifest else None,
        "data_bundle_manifest": str(data_bundle_manifest) if data_bundle_manifest else None,
        "probe_transform_manifest": str(probe_transform_manifest) if probe_transform_manifest else None,
        "pinn_randomness_manifest": str(pinn_randomness_manifest) if pinn_randomness_manifest else None,
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


def _label_amp_phase(y_i: np.ndarray, y_phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amp = np.asarray(y_i, dtype=np.float64)
    phase = np.asarray(y_phi, dtype=np.float64)
    if amp.ndim == 3:
        amp = amp[..., 0]
    if phase.ndim == 3:
        phase = phase[..., 0]
    if amp.ndim != 2 or phase.ndim != 2:
        raise ValueError("simulated labels must resolve to 2D amplitude and phase arrays")
    if amp.shape != phase.shape:
        raise ValueError("simulated label amplitude and phase shapes must match")
    if not np.all(np.isfinite(amp)) or not np.all(np.isfinite(phase)):
        raise ValueError("simulated labels must contain only finite values")
    return amp, phase


def object_patch_from_simulated_labels(
    y_i: np.ndarray,
    y_phi: np.ndarray,
    probe: np.ndarray,
    epsilon_ratio: float = DEFAULT_EPSILON_RATIO,
) -> np.ndarray:
    """Recover normalized object labels from stored illuminated-amplitude labels."""
    amp, phase = _label_amp_phase(y_i, y_phi)
    probe_array = _validate_2d_complex("probe", probe)
    if probe_array.shape != amp.shape:
        raise ValueError("probe shape must match simulated labels")
    probe_amp = np.abs(probe_array)
    max_amp = float(probe_amp.max(initial=0.0))
    if max_amp <= 0:
        raise ValueError("zero-amplitude probe cannot be used for label recovery")
    epsilon = float(epsilon_ratio) * max_amp
    object_amp = np.zeros_like(amp, dtype=np.float64)
    valid = probe_amp >= epsilon
    object_amp[valid] = amp[valid] / probe_amp[valid]
    return (object_amp * np.exp(1j * phase)).astype(np.complex64)


def _exit_wave_from_simulated_labels(
    y_i: np.ndarray,
    y_phi: np.ndarray,
    probe: np.ndarray,
) -> np.ndarray:
    """Reconstruct the simulated exit wave from stored labels and probe phase."""
    amp, phase = _label_amp_phase(y_i, y_phi)
    probe_array = _validate_2d_complex("probe", probe)
    if probe_array.shape != amp.shape:
        raise ValueError("probe shape must match simulated labels")
    return (amp * np.exp(1j * phase) * np.exp(1j * np.angle(probe_array))).astype(np.complex64)


def check_forward_amplitude_self_consistency(
    target_magnitude: np.ndarray,
    y_i: np.ndarray,
    y_phi: np.ndarray,
    probe: np.ndarray,
    tolerance: float = 5e-3,
) -> dict[str, Any]:
    """Compare stored normalized X with the known ground-truth exit-wave amplitude."""
    target = _validate_target_magnitude(np.asarray(target_magnitude, dtype=np.float64), np.asarray(target_magnitude).shape)
    exit_wave = _exit_wave_from_simulated_labels(y_i, y_phi, probe)
    predicted = forward_amplitude(exit_wave)
    denominator = max(float(np.linalg.norm(target)), 1e-12)
    residual = float(np.linalg.norm(predicted - target) / denominator)
    return {
        "status": "ok" if math.isfinite(residual) and residual <= float(tolerance) else "failed",
        "normalized_residual": residual,
        "tolerance": float(tolerance),
        "exit_wave_source": "stored_label_amplitude_plus_object_phase_plus_probe_phase",
        "target_magnitude_shape": list(target.shape),
    }


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


def build_metrics_artifact_context(
    row_label: str,
    data_identity_manifest: Path,
    metric_contract_manifest: Path,
    support_manifest: Path,
    residual_payload: dict[str, Any],
    residual_path: Path,
) -> dict[str, Any]:
    """Summarize the manifests and residual content required by the row contract."""
    data_identity = _read_manifest(Path(data_identity_manifest))
    metric_contract = _read_manifest(Path(metric_contract_manifest))
    support_payload = _read_manifest(Path(support_manifest))

    selected_restarts: list[dict[str, Any]] = []
    selected_residual_curves: list[dict[str, Any]] = []
    restart_final_residuals_by_patch: list[dict[str, Any]] = []
    for patch in residual_payload.get("patches", []):
        selected_restarts.append(
            {
                "patch_index": patch.get("patch_index"),
                "selected_seed": patch.get("selected_seed"),
                "selected_base_seed": patch.get("selected_base_seed"),
                "selected_restart_index": patch.get("selected_restart_index"),
                "selected_final_residual": patch.get("selected_final_residual"),
            }
        )
        selected_residual_curves.append(
            {
                "patch_index": patch.get("patch_index"),
                "selected_residual_curve": patch.get("selected_residual_curve"),
            }
        )
        restart_final_residuals_by_patch.append(
            {
                "patch_index": patch.get("patch_index"),
                "restart_final_residuals": patch.get("restart_final_residuals", []),
            }
        )

    return {
        "data_identity": {
            "manifest": str(data_identity_manifest),
            "manifest_sha256": _sha256_file(Path(data_identity_manifest)),
            "branch": data_identity.get("branch"),
            "decision": data_identity.get("decision"),
            "data_generation_control": data_identity.get("data_generation_control"),
            "data_generation_control_status": data_identity.get("data_generation_control_status"),
            "old_table2_value_policy": data_identity.get("old_table2_value_policy"),
            "old_table2_same_data_comparator_allowed": data_identity.get(
                "old_table2_same_data_comparator_allowed"
            ),
        },
        "metric_contract": {
            "manifest": str(metric_contract_manifest),
            "manifest_sha256": _sha256_file(Path(metric_contract_manifest)),
            "mode": metric_contract.get("mode"),
            "table2_compatibility": metric_contract.get("table2_compatibility"),
            "table2_compatible": metric_contract.get("table2_compatible"),
            "metric_inspection_allowed": metric_contract.get("metric_inspection_allowed"),
            "main_row_policy": metric_contract.get("main_row_policy"),
        },
        "support_threshold_grid_status": support_payload.get("support_records", []),
        "selected_restart": {
            "row_label": row_label,
            "selection_metric": "final_fourier_amplitude_residual",
            "selection_uses_ground_truth": False,
            "per_patch": selected_restarts,
        },
        "residuals": {
            "path": str(residual_path),
            "sha256": _sha256_file(Path(residual_path)),
            "condition": residual_payload.get("condition"),
            "support_threshold": residual_payload.get("support_threshold"),
            "restart_seeds": residual_payload.get("restart_seeds"),
            "full_curve_storage": "residual_json",
            "selected_residual_curves": selected_residual_curves,
            "restart_final_residuals_by_patch": restart_final_residuals_by_patch,
        },
    }


def run_smoke_benchmark(
    output_root: Path,
    args: argparse.Namespace,
    probe: np.ndarray,
    support: np.ndarray,
    support_record: dict[str, Any],
    data_identity_manifest: Path | None = None,
    metric_contract_manifest: Path | None = None,
    support_manifest: Path | None = None,
    cfg: Any | None = None,
    data: dict[str, Any] | None = None,
) -> tuple[list[Path], list[Path], list[Path]]:
    from ptycho.evaluation import eval_reconstruction
    from ptycho.workflows.grid_lines_workflow import save_recon_artifact, stitch_predictions

    start = time.perf_counter()
    if cfg is None or data is None:
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
            condition_id=_condition_label(args),
            patch_index=index,
        )
        reconstructed = recover_object_patch(
            restart_run.selected.psi,
            probe,
            support,
            epsilon_ratio=DEFAULT_EPSILON_RATIO,
        )
        self_consistency_check = check_forward_amplitude_self_consistency(
            target,
            y_i_test[index],
            y_phi_test[index],
            probe,
        )
        self_consistency_check.update(
            {
                "patch_index": index,
                "selected_seed": restart_run.selected.seed,
                "selected_final_residual": restart_run.selected.final_residual,
            }
        )
        self_consistency_checks.append(self_consistency_check)
        reconstructed_patches.append(reconstructed[..., None])
        ground_truth_patches.append(
            object_patch_from_simulated_labels(y_i_test[index], y_phi_test[index], probe)[..., None]
        )
        residual_payload["patches"].append(
            {
                "patch_index": index,
                "selected_seed": restart_run.selected.seed,
                "selected_base_seed": restart_run.selected.base_seed,
                "selected_restart_index": restart_run.selected.restart_index,
                "selected_final_residual": restart_run.selected.final_residual,
                "restart_final_residuals": [
                    {
                        "restart_index": item.restart_index,
                        "base_seed": item.base_seed,
                        "seed": item.seed,
                        "final_residual": item.final_residual,
                    }
                    for item in restart_run.restarts
                ],
                "restarts": _restart_records(restart_run),
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

    artifact_context: dict[str, Any] = {}
    if data_identity_manifest and metric_contract_manifest and support_manifest:
        artifact_context = build_metrics_artifact_context(
            row_label=row_label,
            data_identity_manifest=data_identity_manifest,
            metric_contract_manifest=metric_contract_manifest,
            support_manifest=support_manifest,
            residual_payload=residual_payload,
            residual_path=residual_path,
        )

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
            "status": "ok" if all(item["status"] == "ok" for item in self_consistency_checks) else "failed",
            "checks": self_consistency_checks,
        },
        "timing_seconds": time.perf_counter() - start,
        "recon_npz": str(recon_path),
        "residual_json": str(residual_path),
    }
    metrics_payload.update(artifact_context)
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
    parser.add_argument("--residual-period", default=10, type=int)
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
    data_generation_manifest: Path | None = None
    data_bundle_manifest: Path | None = None
    probe_transform_manifest: Path | None = None
    pinn_randomness_manifest: Path | None = None
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

    gate_report: dict[str, Any] | None = None
    prepared_cfg: Any | None = None
    prepared_data: dict[str, Any] | None = None
    if args.data_identity_branch == "same-split-rerun":
        from ptycho.workflows.grid_lines_workflow import configure_legacy_params

        if args.data_generation_control != "loader-compatible":
            assert_metric_gates_allow_metrics(data_identity_manifest, metric_contract_manifest)
        _memoization_policy_for_fresh_bundle()
        prepared_cfg, prepared_data = _generate_table2_smoke_data(args, probe)
        config = configure_legacy_params(prepared_cfg, probe)
        bundle = persist_same_split_data_bundle(
            output_root=output_root,
            run_id=args.run_id,
            cfg=prepared_cfg,
            data=prepared_data,
            config=config,
            probe=probe,
            probe_transform_pipeline=probe_pipeline,
            probe_transform_steps=probe_steps,
            data_generation_control=args.data_generation_control,
        )
        data_identity_manifest = update_data_identity_manifest_with_generated_bundle(
            data_identity_manifest,
            bundle,
        )
        data_generation_manifest = Path(bundle["data_generation_manifest"])
        data_bundle_manifest = Path(bundle["data_bundle_manifest"])
        probe_transform_manifest = Path(bundle["probe_transform_manifest"])
        pinn_randomness_manifest = write_pinn_randomness_manifest(
            output_root,
            run_id=args.run_id,
            data_generation_control=args.data_generation_control,
            metric_contract_manifest=metric_contract_manifest,
            data_bundle_manifest=data_bundle_manifest,
        )
        gate_report = assert_metric_gates_allow_metrics(data_identity_manifest, metric_contract_manifest)
    else:
        gate_report = assert_metric_gates_allow_metrics(data_identity_manifest, metric_contract_manifest)

    metrics_paths, residual_paths, recon_paths = run_smoke_benchmark(
        output_root,
        args,
        probe,
        support,
        support_record,
        data_identity_manifest=data_identity_manifest,
        metric_contract_manifest=metric_contract_manifest,
        support_manifest=support_manifest,
        cfg=prepared_cfg,
        data=prepared_data,
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
        data_generation_manifest=data_generation_manifest,
        data_bundle_manifest=data_bundle_manifest,
        probe_transform_manifest=probe_transform_manifest,
        pinn_randomness_manifest=pinn_randomness_manifest,
    )
    _write_json(output_root / "metric_gate_report.json", gate_report)
    print(f"benchmark complete: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

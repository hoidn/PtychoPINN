"""Train-only legacy amplitude-gain calibration preparation and selection.

This module owns dataset splitting, generic-driver manifest derivation, and
profile-wide gain selection. Model training, reconstruction, and metrics remain
owned by :mod:`scripts.studies.torch_ablation_driver` and its runtime modules.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset_content import array_sha256, canonicalize_probe, probe_geometry_sha256
from .dataset_schema import DatasetError
from .datasets import load_checked_dataset_bundle, probe_l2_norm
from .configuration import ConfigResolutionError, resolve_torch_configs
from .manifest import Manifest, ResolvedRun, load_manifest, resolve_manifest
from .reporting import ReportingError, verify_completed_report


DEFAULT_ARCHITECTURES = ("cnn", "hybrid_resnet")
DEFAULT_LOSS_PROFILES = ("legacy_mae", "legacy_nll")
DEFAULT_GAINS = (1.0, 4.0, 16.0, 64.0)
_DATASET_ID = "legacy_gain_calibration"
_INVARIANT_KEYS = frozenset({"objectGuess", "probeGuess", "probeGeometry"})
_PER_SAMPLE_KEYS = frozenset(
    {
        "xcoords",
        "ycoords",
        "xcoords_start",
        "ycoords_start",
        "diff3d",
        "scan_index",
        "ground_truth_patches",
    }
)
_REQUIRED_SELECTION_METRICS = (
    "truth_quality.amp_ssim",
    "truth_quality.phase_ssim",
    "stability.finite",
    "stability.amp_variance",
    "stability.amp_dynamic_range",
    "stability.phase_variance",
    "stability.phase_dynamic_range",
)
_SELECTORS = {
    "aggregate": "median_across_architectures",
    "primary": "amplitude_ssim",
    "tie_break": ["phase_ssim", "smaller_gain"],
}
_REPO_ROOT = Path(__file__).resolve().parents[3]


class CalibrationError(ValueError):
    """Raised when calibration preparation or selection is not trustworthy."""


@dataclass(frozen=True)
class SplitRequest:
    source_train: Path
    output_root: Path
    seed: int
    calibration_fraction: float


@dataclass(frozen=True)
class SplitPreparation:
    optimization_npz: Path
    calibration_npz: Path
    evidence_path: Path


@dataclass(frozen=True)
class CalibrationRequest:
    source_train: Path
    base_spec: Path
    output_root: Path
    architectures: tuple[str, ...] = DEFAULT_ARCHITECTURES
    loss_profiles: tuple[str, ...] = DEFAULT_LOSS_PROFILES
    gains: tuple[float, ...] = DEFAULT_GAINS
    seed: int = 3
    epochs: int = 80
    calibration_fraction: float = 0.2
    expected_dataset_id: str = "lines_legacy_amp"


@dataclass(frozen=True)
class CalibrationPreparation:
    output_root: Path
    split: SplitPreparation
    provenance_path: Path
    spec_path: Path
    request_path: Path
    driver_output_root: Path
    candidate_rows_path: Path
    selection_path: Path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise CalibrationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _publish_bytes(path: Path, data: bytes, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise CalibrationError(
                f"cannot inspect existing {label} {path}: {error}"
            ) from error
        if existing == data:
            return
        raise CalibrationError(f"overwrite mismatch for existing {label}: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _publish_json(path: Path, payload: Mapping[str, Any], *, label: str) -> None:
    _publish_bytes(path, _json_bytes(payload), label=label)


def _identity_component(value: Any) -> str:
    item = np.asarray(value).item()
    if isinstance(item, float):
        if not math.isfinite(item):
            raise CalibrationError("scan positions must be finite")
        return float(item).hex()
    if isinstance(item, (int, np.integer)):
        return str(int(item))
    return str(item)


def _scan_identities(arrays: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    required = ("scan_index", "xcoords", "ycoords")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise CalibrationError(
            f"source TRAIN NPZ requires scan_index, xcoords, and ycoords; missing {missing!r}"
        )
    scans = np.asarray(arrays["scan_index"])
    xcoords = np.asarray(arrays["xcoords"])
    ycoords = np.asarray(arrays["ycoords"])
    if (
        scans.ndim != 1
        or xcoords.ndim != 1
        or ycoords.ndim != 1
        or scans.shape != xcoords.shape
        or xcoords.shape != ycoords.shape
    ):
        raise CalibrationError(
            "source TRAIN scan_index/xcoords/ycoords must be matching 1D arrays"
        )
    identities: list[str] = []
    for index, (xcoord, ycoord) in enumerate(zip(xcoords, ycoords, strict=True)):
        identities.append(
            f"scan={_identity_component(scans[index])}"
            f"|x={_identity_component(xcoord)}|y={_identity_component(ycoord)}"
        )
    if not identities:
        raise CalibrationError(
            "source TRAIN NPZ must contain at least two scan identities"
        )
    return tuple(identities)


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _identity_digest(values: Sequence[str]) -> str:
    canonical = json.dumps(list(values), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _split_assignment(
    identities: Sequence[str], *, seed: int, calibration_fraction: float
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CalibrationError("seed must be a nonnegative integer")
    if not math.isfinite(calibration_fraction) or not 0.0 < calibration_fraction < 1.0:
        raise CalibrationError(
            "calibration_fraction must be finite and between 0 and 1"
        )
    unique = _ordered_unique(identities)
    if len(unique) < 2:
        raise CalibrationError(
            "source TRAIN NPZ needs at least two unique scan identities"
        )
    count = int(round(len(unique) * calibration_fraction))
    count = min(max(count, 1), len(unique) - 1)
    ranked = sorted(
        unique,
        key=lambda identity: (
            hashlib.sha256(f"{seed}\0{identity}".encode()).digest(),
            identity,
        ),
    )
    calibration_ids = frozenset(ranked[:count])
    calibration_mask = np.asarray(
        [identity in calibration_ids for identity in identities], dtype=np.bool_
    )
    optimization_mask = ~calibration_mask
    ordered_optimization = tuple(item for item in unique if item not in calibration_ids)
    ordered_calibration = tuple(item for item in unique if item in calibration_ids)
    return (
        optimization_mask,
        calibration_mask,
        ordered_optimization,
        ordered_calibration,
    )


def _split_payload(
    arrays: Mapping[str, np.ndarray], mask: np.ndarray, *, sample_count: int
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if key in _PER_SAMPLE_KEYS:
            if array.ndim == 0 or array.shape[0] != sample_count:
                raise CalibrationError(
                    f"per-sample array {key!r} must have leading dimension "
                    f"{sample_count}"
                )
            payload[key] = np.asarray(array[mask])
        elif key in _INVARIANT_KEYS or key == "_metadata" or array.ndim == 0:
            payload[key] = np.array(array, copy=True)
        elif array.shape[0] == sample_count:
            raise CalibrationError(
                f"ambiguous per-sample array {key!r}; add it to the explicit v3 "
                "split contract before calibration"
            )
        else:
            payload[key] = np.array(array, copy=True)
    return payload


def _write_npz_atomic(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_npz_arrays(path: Path, *, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise CalibrationError(f"cannot load {label} NPZ {path}: {error}") from error


def _payload_matches(
    observed: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray]
) -> bool:
    if set(observed) != set(expected):
        return False
    for key, expected_value in expected.items():
        observed_value = observed[key]
        numeric = np.issubdtype(expected_value.dtype, np.number)
        if not np.array_equal(
            observed_value,
            expected_value,
            equal_nan=numeric,
        ):
            return False
    return True


def prepare_split(request: SplitRequest) -> SplitPreparation:
    """Create deterministic optimization/calibration archives from TRAIN only."""
    source = Path(request.source_train).resolve()
    root = Path(request.output_root).resolve()
    result = SplitPreparation(
        optimization_npz=root / "optimization.npz",
        calibration_npz=root / "calibration.npz",
        evidence_path=root / "split_evidence.json",
    )
    existing = (
        result.optimization_npz.exists(),
        result.calibration_npz.exists(),
        result.evidence_path.exists(),
    )
    if any(existing):
        if not all(existing):
            raise CalibrationError(
                f"overwrite mismatch for incomplete existing split under {root}"
            )
        try:
            evidence = json.loads(result.evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationError(
                f"overwrite mismatch for split evidence: {error}"
            ) from error
        arrays = _load_npz_arrays(source, label="source TRAIN")
        identities = _scan_identities(arrays)
        optimization_mask, calibration_mask, optimization_ids, calibration_ids = (
            _split_assignment(
                identities,
                seed=request.seed,
                calibration_fraction=request.calibration_fraction,
            )
        )
        expected_optimization = _split_payload(
            arrays, optimization_mask, sample_count=len(identities)
        )
        expected_calibration = _split_payload(
            arrays, calibration_mask, sample_count=len(identities)
        )
        try:
            observed_optimization = _load_npz_arrays(
                result.optimization_npz, label="optimization"
            )
            observed_calibration = _load_npz_arrays(
                result.calibration_npz, label="calibration"
            )
        except CalibrationError as error:
            raise CalibrationError(
                f"overwrite mismatch for existing split under {root}"
            ) from error
        expected_request = (
            evidence.get("source_train_sha256") == _file_sha256(source)
            and evidence.get("seed") == request.seed
            and evidence.get("calibration_fraction") == request.calibration_fraction
            and evidence.get("identity_policy") == "scan_index+xcoords+ycoords"
            and evidence.get("optimization_scan_ids") == list(optimization_ids)
            and evidence.get("calibration_scan_ids") == list(calibration_ids)
            and evidence.get("optimization_scan_ids_sha256")
            == _identity_digest(optimization_ids)
            and evidence.get("calibration_scan_ids_sha256")
            == _identity_digest(calibration_ids)
            and evidence.get("optimization_rows")
            == int(np.count_nonzero(optimization_mask))
            and evidence.get("calibration_rows")
            == int(np.count_nonzero(calibration_mask))
            and evidence.get("source_rows") == len(identities)
            and evidence.get("disjoint") is True
            and evidence.get("coverage") is True
            and evidence.get("output_sha256")
            == {
                "optimization": _file_sha256(result.optimization_npz),
                "calibration": _file_sha256(result.calibration_npz),
            }
            and _payload_matches(observed_optimization, expected_optimization)
            and _payload_matches(observed_calibration, expected_calibration)
        )
        if not expected_request:
            raise CalibrationError(
                f"overwrite mismatch for existing split under {root}"
            )
        return result

    arrays = _load_npz_arrays(source, label="source TRAIN")
    identities = _scan_identities(arrays)
    optimization_mask, calibration_mask, optimization_ids, calibration_ids = (
        _split_assignment(
            identities,
            seed=request.seed,
            calibration_fraction=request.calibration_fraction,
        )
    )
    optimization_payload = _split_payload(
        arrays, optimization_mask, sample_count=len(identities)
    )
    calibration_payload = _split_payload(
        arrays, calibration_mask, sample_count=len(identities)
    )
    try:
        _write_npz_atomic(result.optimization_npz, optimization_payload)
        _write_npz_atomic(result.calibration_npz, calibration_payload)
        evidence = {
            "schema_version": "legacy_gain_train_split_v1",
            "source_train_sha256": _file_sha256(source),
            "seed": request.seed,
            "calibration_fraction": request.calibration_fraction,
            "identity_policy": "scan_index+xcoords+ycoords",
            "optimization_scan_ids": list(optimization_ids),
            "calibration_scan_ids": list(calibration_ids),
            "optimization_scan_ids_sha256": _identity_digest(optimization_ids),
            "calibration_scan_ids_sha256": _identity_digest(calibration_ids),
            "optimization_rows": int(np.count_nonzero(optimization_mask)),
            "calibration_rows": int(np.count_nonzero(calibration_mask)),
            "source_rows": len(identities),
            "disjoint": set(optimization_ids).isdisjoint(calibration_ids),
            "coverage": set(optimization_ids) | set(calibration_ids) == set(identities),
            "output_sha256": {
                "optimization": _file_sha256(result.optimization_npz),
                "calibration": _file_sha256(result.calibration_npz),
            },
        }
        _publish_json(result.evidence_path, evidence, label="split evidence")
    except Exception:
        if not result.evidence_path.exists():
            result.optimization_npz.unlink(missing_ok=True)
            result.calibration_npz.unlink(missing_ok=True)
        raise
    return result


def _validated_tuple(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise CalibrationError(f"{name} must contain nonempty ids")
    if len(set(result)) != len(result):
        raise CalibrationError(f"{name} must not contain duplicates")
    return result


def _validated_gains(values: Sequence[float]) -> tuple[float, ...]:
    gains = tuple(float(value) for value in values)
    if not gains or any(not math.isfinite(value) or value <= 0.0 for value in gains):
        raise CalibrationError("gains must be finite and positive")
    if len(set(gains)) != len(gains):
        raise CalibrationError("gains must not contain duplicates")
    return gains


def _match_source_dataset(
    raw: Mapping[str, Any], source_hash: str, *, expected_dataset_id: str
) -> tuple[str, dict[str, Any]]:
    if not isinstance(expected_dataset_id, str) or not expected_dataset_id:
        raise CalibrationError("expected_dataset_id must be a nonempty id")
    datasets = raw.get("datasets")
    if not isinstance(datasets, Mapping):
        raise CalibrationError("base spec has no datasets table")
    matches = [
        (str(dataset_id), copy.deepcopy(dict(value)))
        for dataset_id, value in datasets.items()
        if isinstance(value, Mapping) and value.get("train_sha256") == source_hash
    ]
    if len(matches) != 1:
        raise CalibrationError(
            "source TRAIN SHA256 must match exactly one base-spec train descriptor"
        )
    dataset_id, dataset = matches[0]
    if dataset_id != expected_dataset_id:
        raise CalibrationError(
            f"source TRAIN descriptor id {dataset_id!r} does not match "
            f"expected dataset id {expected_dataset_id!r}"
        )
    if (
        dataset.get("scale_contract_version") != "legacy_v1"
        or dataset.get("measurement_domain") != "normalized_amplitude"
    ):
        raise CalibrationError(
            "gain calibration accepts legacy normalized-amplitude TRAIN data only"
        )
    if dataset.get("truth_location") != "embedded_test":
        raise CalibrationError(
            "calibration requires truth embedded in the source TRAIN archive"
        )
    probe = dataset.get("probe")
    if not isinstance(probe, Mapping) or (
        probe.get("calibration") != "legacy_normalized"
        or probe.get("gauge") != "legacy_normalized"
    ):
        raise CalibrationError(
            "gain calibration requires legacy-compatible probe calibration and gauge"
        )
    return dataset_id, dataset


def _dimension_values(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    matrix = raw.get("matrix")
    dimensions = matrix.get("dimensions") if isinstance(matrix, Mapping) else None
    if not isinstance(dimensions, list):
        raise CalibrationError("base spec has no matrix dimensions")
    values: dict[str, dict[str, Any]] = {}
    for dimension in dimensions:
        if not isinstance(dimension, Mapping) or not isinstance(
            dimension.get("values"), list
        ):
            raise CalibrationError("base spec has malformed matrix dimensions")
        for value in dimension["values"]:
            if isinstance(value, Mapping) and isinstance(value.get("id"), str):
                identifier = value["id"]
                if identifier in values:
                    raise CalibrationError(
                        f"base spec repeats matrix value id {identifier!r}"
                    )
                values[identifier] = copy.deepcopy(dict(value))
    return values


def _selected_values(
    available: Mapping[str, dict[str, Any]], requested: Sequence[str], *, kind: str
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for identifier in requested:
        if identifier not in available:
            raise CalibrationError(
                f"base spec does not declare requested {kind} {identifier!r}"
            )
        selected.append(copy.deepcopy(available[identifier]))
    return selected


def _run_matrix_record(run: ResolvedRun) -> dict[str, Any]:
    return {
        "run_id": run.id,
        "arm_id": run.arm_id,
        "dataset_id": run.dataset_id,
        "seed": run.seed,
        "dimensions": json.loads(json.dumps(dict(run.dimensions))),
        "overrides": json.loads(json.dumps(dict(run.overrides))),
    }


def _validate_legacy_run_semantics(
    runs: Sequence[ResolvedRun],
    *,
    require_all_explicit: bool,
    dataset: Mapping[str, Any],
) -> None:
    probe = dataset.get("probe")
    if not isinstance(probe, Mapping) or (
        probe.get("calibration") != "legacy_normalized"
        or probe.get("gauge") != "legacy_normalized"
    ):
        raise CalibrationError(
            "generated runs violate legacy semantics: probe calibration and gauge "
            "must be legacy_normalized"
        )
    allowed_losses = {("MAE", "mae"), ("Poisson", "poisson")}
    for run in runs:
        try:
            resolved = resolve_torch_configs(
                dict(run.overrides),
                require_all_explicit=require_all_explicit,
            )
        except ConfigResolutionError as error:
            raise CalibrationError(
                f"generated run {run.id!r} violates legacy semantics: {error}"
            ) from error
        legacy = (
            resolved.data_config.scale_contract_version == "legacy_v1"
            and resolved.data_config.measurement_domain == "normalized_amplitude"
            and resolved.model_config.physics_forward_mode == "amplitude"
            and resolved.model_config.rect_s1s2_trainable is False
            and resolved.inference_config.varpro_scaling is False
            and resolved.ci_scaling_active is False
            and (
                resolved.model_config.loss_function,
                resolved.training_config.torch_loss_mode,
            )
            in allowed_losses
        )
        if not legacy:
            raise CalibrationError(
                f"generated run {run.id!r} violates required legacy semantics"
            )


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and len(commit) == 40 else "0" * 40


def _probe_record(probe: np.ndarray) -> tuple[str, float, float, list[float], str]:
    canonical = np.asarray(probe).astype(np.complex128, copy=False)
    mode_energies = np.sum(np.abs(canonical) ** 2, axis=(-2, -1), dtype=np.float64)
    total = float(mode_energies.sum())
    return (
        array_sha256(probe),
        probe_l2_norm(probe),
        total,
        [float(item) for item in mode_energies],
        probe_geometry_sha256(probe),
    )


def _build_provenance(
    *,
    dataset: Mapping[str, Any],
    split: SplitPreparation,
    seed: int,
) -> tuple[dict[str, Any], str]:
    try:
        with (
            np.load(split.optimization_npz, allow_pickle=False) as optimization,
            np.load(split.calibration_npz, allow_pickle=False) as held,
        ):
            x_key = str(dataset["x_key"])
            y_key = str(dataset["y_key"])
            probe_key = str(dataset["probe_key"])
            truth_key = str(dataset["truth_key"])
            detector_shape = tuple(int(item) for item in dataset["detector_shape"])
            optimization_probe = canonicalize_probe(
                optimization[probe_key], detector_shape
            )
            calibration_probe = canonicalize_probe(held[probe_key], detector_shape)
            train_probe = _probe_record(optimization_probe)
            calibration_probe_record = _probe_record(calibration_probe)
            source_object_hash = array_sha256(np.asarray(held[truth_key]))
            coordinate_record = {
                "train_x_sha256": array_sha256(np.asarray(optimization[x_key])),
                "train_y_sha256": array_sha256(np.asarray(optimization[y_key])),
                "test_x_sha256": array_sha256(np.asarray(held[x_key])),
                "test_y_sha256": array_sha256(np.asarray(held[y_key])),
            }
    except (OSError, KeyError, ValueError) as error:
        raise CalibrationError(
            f"cannot derive calibration provenance: {error}"
        ) from error
    if train_probe[0] != calibration_probe_record[0]:
        raise CalibrationError(
            "split invariant probe changed between optimization and calibration"
        )
    probe_descriptor = dict(dataset["probe"])
    record = {
        "kind": dataset["kind"],
        "format": dataset["format"],
        "scale_contract_version": dataset["scale_contract_version"],
        "measurement_domain": dataset["measurement_domain"],
        "truth": dataset["truth"],
        "truth_location": dataset["truth_location"],
        "measurement_key": dataset["measurement_key"],
        "probe_key": dataset["probe_key"],
        "x_key": dataset["x_key"],
        "y_key": dataset["y_key"],
        "truth_key": dataset["truth_key"],
        "coords_convention": dataset["coords_convention"],
        "detector_shape": list(dataset["detector_shape"]),
        "grouping_max_C": dataset["grouping_max_C"],
        "probe_modes": dataset["probe_modes"],
        "source_object_id": "source_train_object",
        "coordinate_set_id": "train_calibration_coordinates",
        "probe_geometry_id": "source_train_probe",
        "dose_family_id": None,
        "base_dataset_id": None,
        "dose_multiplier": None,
        "files": {
            "train": _file_sha256(split.optimization_npz),
            "test": _file_sha256(split.calibration_npz),
        },
        "probe": {
            "source": probe_descriptor["source"],
            "calibration": "legacy_normalized",
            "gauge": "legacy_normalized",
            "mask_policy": probe_descriptor["mask_policy"],
            "train_sha256": train_probe[0],
            "test_sha256": calibration_probe_record[0],
            "train_l2_norm": train_probe[1],
            "test_l2_norm": calibration_probe_record[1],
            "train_total_energy": train_probe[2],
            "test_total_energy": calibration_probe_record[2],
            "train_mode_energies": train_probe[3],
            "test_mode_energies": calibration_probe_record[3],
        },
    }
    payload = {
        "schema_version": 1,
        "materializer_id": "legacy_gain_calibration",
        "materializer_version": 1,
        "generator_commit": _git_commit(),
        "expected_dataset_ids": [_DATASET_ID],
        "seeds": {
            "object": seed,
            "train_coordinates": seed,
            "test_coordinates": seed,
            "measurements": {_DATASET_ID: {"train": seed, "test": seed}},
        },
        "source_objects": {"source_train_object": {"sha256": source_object_hash}},
        "coordinate_sets": {"train_calibration_coordinates": coordinate_record},
        "probe_geometries": {"source_train_probe": {"sha256": train_probe[4]}},
        "datasets": {_DATASET_ID: record},
    }
    return payload, train_probe[0]


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CalibrationError("generated TOML cannot contain nonfinite values")
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise CalibrationError(f"unsupported generated TOML value {value!r}")


def _append_table(lines: list[str], header: str, values: Mapping[str, Any]) -> None:
    lines.extend((f"[{header}]",))
    for key, value in values.items():
        lines.append(
            f"{json.dumps(str(key)) if '.' in str(key) else key} = {_toml_value(value)}"
        )
    lines.append("")


def _manifest_text(
    *,
    base_raw: Mapping[str, Any],
    dataset: Mapping[str, Any],
    architecture_values: Sequence[Mapping[str, Any]],
    profile_values: Sequence[Mapping[str, Any]],
    gains: Sequence[float],
    seed: int,
    epochs: int,
    driver_output_root: Path,
) -> str:
    study = base_raw.get("study", {})
    base = base_raw.get("base", {})
    overrides = (
        copy.deepcopy(dict(base.get("overrides", {})))
        if isinstance(base, Mapping)
        else {}
    )
    overrides["dataset.id"] = _DATASET_ID
    overrides["training.epochs"] = epochs
    overrides.pop("model.amplitude_physics_gain", None)
    lines = ["[schema]", "version = 1", ""]
    _append_table(
        lines,
        "study",
        {
            "id": "legacy-gain-calibration",
            "seeds": [seed],
            "output_root": str(driver_output_root),
            "require_all_explicit": bool(study.get("require_all_explicit", False)),
        },
    )
    _append_table(lines, "base.overrides", overrides)
    dataset_scalars = {key: value for key, value in dataset.items() if key != "probe"}
    _append_table(lines, f"datasets.{_DATASET_ID}", dataset_scalars)
    _append_table(lines, f"datasets.{_DATASET_ID}.probe", dict(dataset["probe"]))
    for name, values in (
        ("architecture", architecture_values),
        ("loss_profile", profile_values),
    ):
        lines.extend(("[[matrix.dimensions]]", f"name = {_toml_value(name)}", ""))
        for value in values:
            lines.extend(
                ("[[matrix.dimensions.values]]", f"id = {_toml_value(value['id'])}")
            )
            value_overrides = value.get("overrides", {})
            if value_overrides:
                lines.append("[matrix.dimensions.values.overrides]")
                for key, item in value_overrides.items():
                    lines.append(f"{json.dumps(key)} = {_toml_value(item)}")
            lines.append("")
    lines.extend(("[[matrix.dimensions]]", 'name = "gain"', ""))
    for gain in gains:
        lines.extend(
            (
                "[[matrix.dimensions.values]]",
                f"id = {_toml_value(f'gain_{gain:g}')}",
                "[matrix.dimensions.values.overrides]",
                f'"model.amplitude_physics_gain" = {_toml_value(gain)}',
                "",
            )
        )
    return "\n".join(lines)


def _repository_relative(path: Path, *, field: str) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT.resolve()))
    except ValueError as error:
        raise CalibrationError(
            f"{field} must be inside repository root {_REPO_ROOT} for the generic driver"
        ) from error


def _preparation(root: Path) -> CalibrationPreparation:
    root = root.resolve()
    return CalibrationPreparation(
        output_root=root,
        split=SplitPreparation(
            root / "inputs" / "optimization.npz",
            root / "inputs" / "calibration.npz",
            root / "inputs" / "split_evidence.json",
        ),
        provenance_path=root / "calibration_provenance.json",
        spec_path=root / "calibration_spec.toml",
        request_path=root / "calibration_request.json",
        driver_output_root=root / "driver",
        candidate_rows_path=root / "candidate_rows.json",
        selection_path=root / "selection.json",
    )


def prepare_calibration(request: CalibrationRequest) -> CalibrationPreparation:
    """Prepare the TRAIN-only split and a generic-driver calibration matrix."""
    source = Path(request.source_train).resolve()
    base_spec = Path(request.base_spec).resolve()
    root = Path(request.output_root).resolve()
    _repository_relative(root, field="output_root")
    architectures = _validated_tuple(request.architectures, name="architectures")
    profiles = _validated_tuple(request.loss_profiles, name="loss_profiles")
    gains = _validated_gains(request.gains)
    if (
        isinstance(request.epochs, bool)
        or not isinstance(request.epochs, int)
        or request.epochs <= 0
    ):
        raise CalibrationError("epochs must be a positive integer")
    try:
        raw = tomllib.loads(base_spec.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise CalibrationError(
            f"cannot load base study spec {base_spec}: {error}"
        ) from error
    source_hash = _file_sha256(source)
    source_dataset_id, source_dataset = _match_source_dataset(
        raw,
        source_hash,
        expected_dataset_id=request.expected_dataset_id,
    )
    available = _dimension_values(raw)
    architecture_values = _selected_values(
        available, architectures, kind="architecture"
    )
    profile_values = _selected_values(available, profiles, kind="loss profile")
    prepared = _preparation(root)
    split = prepare_split(
        SplitRequest(
            source, root / "inputs", request.seed, request.calibration_fraction
        )
    )
    provenance, probe_hash = _build_provenance(
        dataset=source_dataset,
        split=split,
        seed=request.seed,
    )
    _publish_json(prepared.provenance_path, provenance, label="calibration provenance")
    dataset = copy.deepcopy(source_dataset)
    dataset.update(
        {
            "train": _repository_relative(split.optimization_npz, field="output_root"),
            "test": _repository_relative(split.calibration_npz, field="output_root"),
            "provenance": _repository_relative(
                prepared.provenance_path, field="output_root"
            ),
            "train_sha256": _file_sha256(split.optimization_npz),
            "test_sha256": _file_sha256(split.calibration_npz),
            "provenance_sha256": _file_sha256(prepared.provenance_path),
        }
    )
    dataset.pop("dose", None)
    dataset["probe"] = {
        "source": source_dataset["probe"]["source"],
        "calibration": "legacy_normalized",
        "gauge": "legacy_normalized",
        "mask_policy": source_dataset["probe"]["mask_policy"],
        "sha256": probe_hash,
    }
    try:
        load_checked_dataset_bundle({_DATASET_ID: dataset}, repo_root=_REPO_ROOT)
    except DatasetError as error:
        raise CalibrationError(
            f"generated calibration dataset failed preflight: {error}"
        ) from error
    spec_text = _manifest_text(
        base_raw=raw,
        dataset=dataset,
        architecture_values=architecture_values,
        profile_values=profile_values,
        gains=gains,
        seed=request.seed,
        epochs=request.epochs,
        driver_output_root=prepared.driver_output_root,
    )
    _publish_bytes(prepared.spec_path, spec_text.encode(), label="calibration spec")
    try:
        manifest = load_manifest(prepared.spec_path)
        study = resolve_manifest(manifest)
    except Exception as error:
        raise CalibrationError(
            f"generated calibration spec is invalid: {error}"
        ) from error
    _validate_legacy_run_semantics(
        study.runs,
        require_all_explicit=manifest.require_all_explicit,
        dataset=dataset,
    )
    manifest_sha256 = hashlib.sha256(manifest.canonical_json.encode()).hexdigest()
    split_evidence = json.loads(split.evidence_path.read_text(encoding="utf-8"))
    request_payload = {
        "schema_version": "legacy_gain_calibration_request_v1",
        "source_train": str(source),
        "source_train_sha256": source_hash,
        "expected_dataset_id": request.expected_dataset_id,
        "source_dataset_id": source_dataset_id,
        "base_spec": str(base_spec),
        "base_spec_sha256": _file_sha256(base_spec),
        "output_root": str(root),
        "driver_output_root": str(prepared.driver_output_root),
        "generated_spec_sha256": _file_sha256(prepared.spec_path),
        "generated_manifest_sha256": manifest_sha256,
        "generated_study_id": manifest.study_id,
        "calibration_dataset_id": _DATASET_ID,
        "expected_run_matrix": [_run_matrix_record(run) for run in study.runs],
        "architectures": list(architectures),
        "loss_profiles": list(profiles),
        "gains": list(gains),
        "seed": request.seed,
        "epochs": request.epochs,
        "calibration_fraction": request.calibration_fraction,
        "split_evidence_sha256": _file_sha256(split.evidence_path),
        "split_evidence": split_evidence,
        "selectors": _SELECTORS,
    }
    _publish_json(prepared.request_path, request_payload, label="calibration request")
    return prepared


def driver_command(
    prepared: CalibrationPreparation, *, dry_run: bool
) -> tuple[str, ...]:
    command = (
        "python",
        "-m",
        "scripts.studies.torch_ablation_driver",
        "--spec",
        str(prepared.spec_path),
        "--output-root",
        str(prepared.driver_output_root),
    )
    return command + (("--dry-run",) if dry_run else ())


def run_driver(prepared: CalibrationPreparation, *, dry_run: bool) -> int:
    """Invoke the generic ablation driver without owning any training logic."""
    completed = subprocess.run(
        driver_command(prepared, dry_run=dry_run), check=False, cwd=_REPO_ROOT
    )
    return int(completed.returncode)


def _normalized_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return {
            "architecture": str(row["architecture"]),
            "loss_profile": str(row["loss_profile"]),
            "gain": float(row["gain"]),
            "amplitude_ssim": float(row["amplitude_ssim"]),
            "phase_ssim": float(row["phase_ssim"]),
            "collapsed": bool(row["collapsed"]),
            "status": str(row["status"]),
            **({"run_id": str(row["run_id"])} if "run_id" in row else {}),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise CalibrationError(f"malformed candidate row: {error}") from error


def select_profile_gains(
    rows: Sequence[Mapping[str, Any]],
    *,
    architectures: Sequence[str],
    loss_profiles: Sequence[str],
    gains: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """Select one gain per legacy loss profile across all architectures."""
    architecture_ids = _validated_tuple(architectures, name="architectures")
    profiles = _validated_tuple(loss_profiles, name="loss_profiles")
    candidates = _validated_gains(gains)
    normalized = [_normalized_candidate(row) for row in rows]
    expected = set(product(architecture_ids, profiles, candidates))
    indexed: dict[tuple[str, str, float], dict[str, Any]] = {}
    for row in normalized:
        key = (row["architecture"], row["loss_profile"], row["gain"])
        if key not in expected:
            raise CalibrationError(f"unexpected candidate row {key!r}")
        if key in indexed:
            raise CalibrationError(f"duplicate candidate row {key!r}")
        indexed[key] = row
    missing = expected - set(indexed)
    if missing:
        raise CalibrationError(
            f"complete row matrix required; missing {sorted(missing)!r}"
        )
    for key, row in indexed.items():
        if row["status"] != "success":
            raise CalibrationError(f"candidate row {key!r} is not successful")
        if row["collapsed"]:
            raise CalibrationError(f"candidate row {key!r} is collapsed")
        if not math.isfinite(row["amplitude_ssim"]) or not math.isfinite(
            row["phase_ssim"]
        ):
            raise CalibrationError(
                f"candidate row {key!r} must have finite SSIM metrics"
            )
    result: dict[str, dict[str, Any]] = {}
    minimum, maximum = min(candidates), max(candidates)
    for profile in profiles:
        scores = []
        for gain in candidates:
            profile_rows = [
                indexed[(architecture, profile, gain)]
                for architecture in architecture_ids
            ]
            scores.append(
                {
                    "gain": gain,
                    "median_amplitude_ssim": float(
                        statistics.median(row["amplitude_ssim"] for row in profile_rows)
                    ),
                    "median_phase_ssim": float(
                        statistics.median(row["phase_ssim"] for row in profile_rows)
                    ),
                }
            )
        winner = max(
            scores,
            key=lambda item: (
                item["median_amplitude_ssim"],
                item["median_phase_ssim"],
                -item["gain"],
            ),
        )
        selected_gain = float(winner["gain"])
        boundary = None
        if selected_gain == minimum == maximum:
            boundary = "both"
        elif selected_gain == minimum:
            boundary = "lower"
        elif selected_gain == maximum:
            boundary = "upper"
        result[profile] = {
            "selected_gain": selected_gain,
            "status": "unbracketed" if boundary is not None else "selected",
            "boundary": boundary,
            "candidate_scores": scores,
        }
    return result


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(
            f"cannot load driver artifact {path}: {error}"
        ) from error


def _expansion_matrix_record(run: Mapping[str, Any]) -> dict[str, Any]:
    try:
        dimensions = run["dimensions"]
        overrides = run["overrides"]
        if not isinstance(dimensions, Mapping) or not isinstance(overrides, Mapping):
            raise TypeError("dimensions and overrides must be mappings")
        return {
            "run_id": str(run["id"]),
            "arm_id": str(run["arm_id"]),
            "dataset_id": str(run["dataset_id"]),
            "seed": int(run["seed"]),
            "dimensions": json.loads(json.dumps(dict(dimensions))),
            "overrides": json.loads(json.dumps(dict(overrides))),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise CalibrationError(f"malformed driver run matrix row: {error}") from error


def _bind_completed_report(
    prepared: CalibrationPreparation,
) -> tuple[dict[str, Any], list[dict[str, Any]], Manifest]:
    generated_spec_sha256 = _file_sha256(prepared.spec_path)
    if (
        _file_sha256(prepared.driver_output_root / "source_manifest.toml")
        != generated_spec_sha256
    ):
        raise CalibrationError("sealed driver report has the wrong source manifest")
    try:
        manifest = load_manifest(prepared.spec_path)
        study = resolve_manifest(manifest)
    except Exception as error:
        raise CalibrationError(
            f"cannot validate generated manifest: {error}"
        ) from error
    manifest_sha256 = hashlib.sha256(manifest.canonical_json.encode()).hexdigest()
    generated_matrix = [_run_matrix_record(run) for run in study.runs]
    expansion = _load_json(prepared.driver_output_root / "expansion.json")
    if (
        not isinstance(expansion, Mapping)
        or expansion.get("schema_version") != "ablation_expansion_v1"
    ):
        raise CalibrationError("sealed driver expansion must be an object")
    profiles = expansion.get("dataset_materialization_profiles")
    if (
        expansion.get("study_id") != manifest.study_id
        or expansion.get("manifest_sha256") != manifest_sha256
        or expansion.get("requested_seeds") != list(manifest.seeds)
        or not isinstance(profiles, Mapping)
        or set(profiles) != {_DATASET_ID}
    ):
        raise CalibrationError(
            "sealed driver expansion does not match the generated manifest"
        )
    selected_runs = expansion.get("selected_runs")
    if not isinstance(selected_runs, list):
        raise CalibrationError("sealed driver expansion has no run matrix")
    report_matrix = [_expansion_matrix_record(run) for run in selected_runs]
    if report_matrix != generated_matrix:
        raise CalibrationError("sealed driver expansion has the wrong run matrix")
    return dict(expansion), generated_matrix, manifest


def _validated_request_provenance(
    prepared: CalibrationPreparation,
    request: Mapping[str, Any],
    manifest: Manifest,
    generated_matrix: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        source = Path(request["source_train"]).resolve()
        base_spec = Path(request["base_spec"]).resolve()
    except (KeyError, TypeError) as error:
        raise CalibrationError(
            f"calibration request has invalid paths: {error}"
        ) from error
    if request.get("source_train") != str(source):
        raise CalibrationError("calibration request source_train is not canonical")
    if request.get("base_spec") != str(base_spec):
        raise CalibrationError("calibration request base_spec is not canonical")

    split_evidence = _load_json(prepared.split.evidence_path)
    if not isinstance(split_evidence, Mapping):
        raise CalibrationError("split evidence must be an object")
    try:
        architectures = tuple(
            dict.fromkeys(
                str(row["dimensions"]["architecture"]) for row in generated_matrix
            )
        )
        loss_profiles = tuple(
            dict.fromkeys(
                str(row["dimensions"]["loss_profile"]) for row in generated_matrix
            )
        )
        gains = tuple(
            dict.fromkeys(
                float(row["overrides"]["model.amplitude_physics_gain"])
                for row in generated_matrix
            )
        )
        epochs_values = {
            int(row["overrides"]["training.epochs"]) for row in generated_matrix
        }
    except (KeyError, TypeError, ValueError) as error:
        raise CalibrationError(f"generated run matrix is malformed: {error}") from error
    if len(manifest.seeds) != 1 or len(epochs_values) != 1:
        raise CalibrationError("generated manifest must have one seed and epoch budget")
    seed = manifest.seeds[0]
    epochs = next(iter(epochs_values))
    expected_combinations = set(product(architectures, loss_profiles, gains))
    actual_combinations = {
        (
            str(row["dimensions"]["architecture"]),
            str(row["dimensions"]["loss_profile"]),
            float(row["overrides"]["model.amplitude_physics_gain"]),
        )
        for row in generated_matrix
    }
    if (
        actual_combinations != expected_combinations
        or len(generated_matrix) != len(expected_combinations)
        or any(row["dataset_id"] != _DATASET_ID for row in generated_matrix)
        or any(row["seed"] != seed for row in generated_matrix)
    ):
        raise CalibrationError("generated manifest has the wrong run matrix")

    calibration_fraction = split_evidence.get("calibration_fraction")
    prepare_split(
        SplitRequest(
            source,
            prepared.split.evidence_path.parent,
            seed,
            calibration_fraction,
        )
    )
    source_sha256 = _file_sha256(source)
    try:
        base_raw = tomllib.loads(base_spec.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise CalibrationError(f"cannot validate base study spec: {error}") from error
    source_dataset_id, _ = _match_source_dataset(
        base_raw,
        source_sha256,
        expected_dataset_id=request.get("expected_dataset_id"),
    )
    actual_request = {
        "schema_version": "legacy_gain_calibration_request_v1",
        "source_train": str(source),
        "source_train_sha256": source_sha256,
        "expected_dataset_id": source_dataset_id,
        "source_dataset_id": source_dataset_id,
        "base_spec": str(base_spec),
        "base_spec_sha256": _file_sha256(base_spec),
        "output_root": str(prepared.output_root),
        "driver_output_root": str(prepared.driver_output_root),
        "generated_spec_sha256": _file_sha256(prepared.spec_path),
        "generated_manifest_sha256": hashlib.sha256(
            manifest.canonical_json.encode()
        ).hexdigest(),
        "generated_study_id": manifest.study_id,
        "calibration_dataset_id": _DATASET_ID,
        "expected_run_matrix": list(generated_matrix),
        "architectures": list(architectures),
        "loss_profiles": list(loss_profiles),
        "gains": list(gains),
        "seed": seed,
        "epochs": epochs,
        "calibration_fraction": calibration_fraction,
        "split_evidence_sha256": _file_sha256(prepared.split.evidence_path),
        "split_evidence": dict(split_evidence),
        "selectors": json.loads(json.dumps(_SELECTORS)),
    }
    if set(request) != set(actual_request):
        raise CalibrationError("calibration request has unexpected or missing fields")
    for field, actual_value in actual_request.items():
        if request.get(field) != actual_value:
            raise CalibrationError(
                f"calibration request field {field!r} does not match actual artifacts"
            )
    return actual_request


def _collect_candidate_rows(
    prepared: CalibrationPreparation,
    expansion: Mapping[str, Any],
    expected_matrix: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    statuses_payload = _load_json(prepared.driver_output_root / "arm_seed_status.json")
    metrics_payload = _load_json(prepared.driver_output_root / "aggregate_metrics.json")
    if (
        not isinstance(statuses_payload, Mapping)
        or statuses_payload.get("schema_version") != "ablation_status_v1"
        or not isinstance(statuses_payload.get("rows"), list)
    ):
        raise CalibrationError("sealed driver status artifact has an invalid schema")
    if (
        not isinstance(metrics_payload, Mapping)
        or metrics_payload.get("schema_version") != "ablation_metrics_v1"
        or not isinstance(metrics_payload.get("rows"), list)
    ):
        raise CalibrationError("sealed driver metrics artifact has an invalid schema")
    expected_identities = {
        str(row["run_id"]): (
            str(row["run_id"]),
            str(row["arm_id"]),
            str(row["dataset_id"]),
            int(row["seed"]),
        )
        for row in expected_matrix
    }

    def exact_identity(
        row: Mapping[str, Any], *, label: str
    ) -> tuple[str, str, str, int]:
        run_id = row.get("run_id")
        arm_id = row.get("arm_id")
        dataset_id = row.get("dataset_id")
        seed = row.get("seed")
        if (
            not isinstance(run_id, str)
            or not isinstance(arm_id, str)
            or not isinstance(dataset_id, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise CalibrationError(f"{label} has a malformed run identity")
        identity = (run_id, arm_id, dataset_id, seed)
        if expected_identities.get(run_id) != identity:
            raise CalibrationError(
                f"{label} identity does not match the expected run matrix"
            )
        return identity

    statuses: dict[str, Mapping[str, Any]] = {}
    for row in statuses_payload["rows"]:
        if not isinstance(row, Mapping):
            raise CalibrationError("sealed driver status artifact has a malformed row")
        run_id = exact_identity(row, label="sealed driver status row")[0]
        if run_id in statuses:
            raise CalibrationError(f"duplicate status row for {run_id}")
        statuses[run_id] = row
    metrics: dict[str, dict[str, float]] = {}
    for row in metrics_payload["rows"]:
        if not isinstance(row, Mapping):
            raise CalibrationError("sealed driver metrics artifact has a malformed row")
        path = row.get("metric_path")
        if path not in _REQUIRED_SELECTION_METRICS:
            continue
        run_id = exact_identity(row, label="required selection metric row")[0]
        if path in metrics.setdefault(run_id, {}):
            raise CalibrationError(f"duplicate metric {path!r} for {run_id}")
        try:
            metrics[run_id][path] = float(row.get("value"))
        except (TypeError, ValueError) as error:
            raise CalibrationError(
                f"nonnumeric metric {path!r} for {run_id}"
            ) from error
    expected_run_ids = {str(row["run_id"]) for row in expected_matrix}
    if set(statuses) != expected_run_ids or set(metrics) != expected_run_ids:
        raise CalibrationError(
            "sealed driver report rows do not match the expected run matrix"
        )
    candidates: list[dict[str, Any]] = []
    for run in expansion.get("selected_runs", []):
        run_id = str(run["id"])
        dimensions = run.get("dimensions", {})
        overrides = run.get("overrides", {})
        status = statuses.get(run_id)
        if status is None:
            row_status = "missing"
        elif (
            status.get("status") == "success" and status.get("completion") == "terminal"
        ):
            row_status = "success"
        else:
            row_status = str(status.get("status", "failed"))
        values = metrics.get(run_id, {})
        missing_metrics = [
            name for name in _REQUIRED_SELECTION_METRICS if name not in values
        ]
        if missing_metrics and row_status == "success":
            raise CalibrationError(
                f"successful run {run_id} lacks metrics {missing_metrics!r}"
            )
        collapsed = True
        if not missing_metrics:
            collapsed = not (
                values["stability.finite"] == 1.0
                and values["stability.amp_variance"] > 0.0
                and values["stability.amp_dynamic_range"] > 0.0
                and values["stability.phase_variance"] > 0.0
                and values["stability.phase_dynamic_range"] > 0.0
            )
        candidates.append(
            {
                "run_id": run_id,
                "architecture": dimensions.get("architecture"),
                "loss_profile": dimensions.get("loss_profile"),
                "gain": overrides.get("model.amplitude_physics_gain"),
                "amplitude_ssim": values.get("truth_quality.amp_ssim"),
                "phase_ssim": values.get("truth_quality.phase_ssim"),
                "collapsed": collapsed,
                "status": row_status,
            }
        )
    return candidates


def finalize_selection(output_root: Path) -> dict[str, Any]:
    """Collate generic-driver artifacts and persist profile-wide selections."""
    prepared = _preparation(Path(output_root))
    request = _load_json(prepared.request_path)
    if not isinstance(request, Mapping):
        raise CalibrationError("calibration request must be an object")
    try:
        verify_completed_report(prepared.driver_output_root)
    except ReportingError as error:
        raise CalibrationError(
            f"driver report completion is invalid: {error}"
        ) from error
    expansion, expected_matrix, manifest = _bind_completed_report(prepared)
    provenance = _validated_request_provenance(
        prepared, request, manifest, expected_matrix
    )
    rows = _collect_candidate_rows(prepared, expansion, expected_matrix)
    candidate_payload = {
        "schema_version": "legacy_gain_candidate_rows_v1",
        "rows": rows,
    }
    _publish_json(
        prepared.candidate_rows_path, candidate_payload, label="candidate rows"
    )
    selected = select_profile_gains(
        rows,
        architectures=tuple(provenance["architectures"]),
        loss_profiles=tuple(provenance["loss_profiles"]),
        gains=tuple(provenance["gains"]),
    )
    result = {
        "schema_version": "legacy_gain_selection_v1",
        "source_train_sha256": provenance["source_train_sha256"],
        "split_evidence_sha256": provenance["split_evidence_sha256"],
        "split_evidence": provenance["split_evidence"],
        "architectures": list(provenance["architectures"]),
        "loss_profiles": list(provenance["loss_profiles"]),
        "gains": list(provenance["gains"]),
        "seed": provenance["seed"],
        "epochs": provenance["epochs"],
        "calibration_fraction": provenance["calibration_fraction"],
        "selectors": provenance["selectors"],
        "candidate_rows": rows,
        "selected_gains": selected,
    }
    _publish_json(prepared.selection_path, result, label="gain selection")
    return result


def _csv_strings(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("CSV value must not be empty")
    return values


def _csv_gains(text: str) -> tuple[float, ...]:
    try:
        return tuple(float(item) for item in _csv_strings(text))
    except ValueError as error:
        raise argparse.ArgumentTypeError("gains must be a CSV of numbers") from error


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-npz", type=Path, required=True, help="source TRAIN NPZ"
    )
    parser.add_argument("--base-spec", type=Path, required=True, help="base study TOML")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--expected-dataset-id",
        default="lines_legacy_amp",
        help="base-spec dataset id whose TRAIN hash must match",
    )
    parser.add_argument(
        "--architectures", type=_csv_strings, default=DEFAULT_ARCHITECTURES
    )
    parser.add_argument(
        "--loss-profiles", type=_csv_strings, default=DEFAULT_LOSS_PROFILES
    )
    parser.add_argument("--gains", type=_csv_gains, default=DEFAULT_GAINS)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--split-fraction", type=float, default=0.2, dest="calibration_fraction"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="legacy_gain_calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "dry-run", "execute"):
        _add_prepare_arguments(subparsers.add_parser(command))
    select = subparsers.add_parser("select")
    select.add_argument("--output-root", type=Path, required=True)
    return parser


def _request_from_args(args: argparse.Namespace) -> CalibrationRequest:
    return CalibrationRequest(
        source_train=args.train_npz,
        base_spec=args.base_spec,
        output_root=args.output_root,
        architectures=tuple(args.architectures),
        loss_profiles=tuple(args.loss_profiles),
        gains=tuple(args.gains),
        seed=args.seed,
        epochs=args.epochs,
        calibration_fraction=args.calibration_fraction,
        expected_dataset_id=args.expected_dataset_id,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "select":
            result = finalize_selection(args.output_root)
            print(f"selection {Path(args.output_root).resolve() / 'selection.json'}")
            for profile, selected in result["selected_gains"].items():
                print(
                    f"{profile} gain={selected['selected_gain']:g} status={selected['status']}"
                )
            return 0
        prepared = prepare_calibration(_request_from_args(args))
        print(f"spec {prepared.spec_path}")
        if args.command == "prepare":
            return 0
        return run_driver(prepared, dry_run=args.command == "dry-run")
    except CalibrationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

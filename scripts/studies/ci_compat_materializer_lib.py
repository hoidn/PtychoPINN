"""Deterministic primitives for the Task10 CI-compatibility materializer.

Sibling helper module of ``materialize_ci_compatibility_datasets.py`` (split
to respect the 500-line module budget). Holds the spec value object,
byte-deterministic NPZ serialization, dose/probe statistics that match the
``scripts.studies.ablation`` preflight reductions bit-for-bit, the legacy
normalized-amplitude twin builder, and the guarded no-clobber writer.
Diffraction physics is reused from ``make_synthetic_truth_datasets``.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _entry in (str(REPO), str(REPO / "scripts/studies")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import make_synthetic_truth_datasets as M  # noqa: E402

from ptycho_torch.scaling_contract import (  # noqa: E402
    LEGACY_SCALE_CONTRACT,
    NORMALIZED_AMPLITUDE,
)
from scripts.studies.ablation.datasets import (  # noqa: E402
    TASK10_RAW_PROBE_GEOMETRY_KEY,
    array_sha256,
    probe_l2_norm,
)

_MIN_POSITIONS = 7  # canonical Nearest/K=6 preflight needs K+1 distinct points
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_NPZ_KEY_ORDER = (
    "xcoords",
    "ycoords",
    "xcoords_start",
    "ycoords_start",
    "diff3d",
    "probeGuess",
    "objectGuess",
    "scan_index",
    "ground_truth_patches",
    TASK10_RAW_PROBE_GEOMETRY_KEY,
    "_metadata",
)


class MaterializationError(RuntimeError):
    """Raised for invalid specs, overwrite refusals, and parity failures."""


@dataclass(frozen=True)
class MaterializationSpec:
    """Complete deterministic identity of one materialized bundle."""

    detector_size: int
    object_resolution: int
    object_seed: int
    train_positions: int
    test_positions: int
    train_coordinate_seed: int
    test_coordinate_seed: int
    scan_jitter: float
    measurement_seed: int
    probe_source: str

    def __post_init__(self) -> None:
        for field in ("detector_size", "object_resolution"):
            _require_positive_int(getattr(self, field), field)
        if self.object_resolution <= 2 * self.detector_size:
            raise MaterializationError(
                "object_resolution must exceed twice detector_size so scan "
                "positions fit inside the object with a probe buffer"
            )
        for field in ("train_positions", "test_positions"):
            value = _require_positive_int(getattr(self, field), field)
            if value < _MIN_POSITIONS:
                raise MaterializationError(
                    f"{field} must be at least {_MIN_POSITIONS} for the "
                    f"canonical Nearest/K=6 grouping preflight"
                )
        for field in (
            "object_seed",
            "train_coordinate_seed",
            "test_coordinate_seed",
            "measurement_seed",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise MaterializationError(f"{field} must be a nonnegative integer")
        if not isinstance(self.scan_jitter, (int, float)) or not (
            np.isfinite(self.scan_jitter) and self.scan_jitter >= 0
        ):
            raise MaterializationError("scan_jitter must be finite and nonnegative")
        if not isinstance(self.probe_source, str) or not self.probe_source.strip():
            raise MaterializationError("probe_source must be a nonempty label")


def _require_positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise MaterializationError(f"{field} must be a positive integer")
    return value


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize an NPZ with fixed zip metadata for byte determinism."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for key, value in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload.getvalue())
    return buffer.getvalue()


def ordered_payload(
    payload: dict[str, np.ndarray],
    raw_geometry: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Fix the NPZ key order and store metadata as a pickle-free JSON scalar."""
    arrays = dict(payload)
    arrays[TASK10_RAW_PROBE_GEOMETRY_KEY] = raw_geometry
    arrays["_metadata"] = np.array(json.dumps(metadata, sort_keys=True))
    return {key: arrays[key] for key in _NPZ_KEY_ORDER}


def dose_statistics(measurement: np.ndarray) -> dict[str, int | float]:
    """Match ``ablation.dataset_content._computed_dose`` bit-for-bit."""
    photons = measurement.astype(np.float64).sum(axis=(-2, -1))
    dtype_max = int(np.iinfo(measurement.dtype).max)
    return {
        "counts_mean": float(measurement.mean(dtype=np.float64)),
        "photons_per_image_min": float(photons.min()),
        "photons_per_image_mean": float(photons.mean()),
        "max_observed_count": int(measurement.max()),
        "dtype_max": dtype_max,
        "saturation_fraction": float(
            np.count_nonzero(measurement == dtype_max) / measurement.size
        ),
    }


def probe_statistics(canonical_probe: np.ndarray) -> dict[str, Any]:
    """Match the provenance probe reductions used by bundle preflight."""
    mode_energies = np.sum(
        np.abs(canonical_probe.astype(np.complex128, copy=False)) ** 2,
        axis=(-2, -1),
        dtype=np.float64,
    )
    return {
        "sha256": array_sha256(canonical_probe),
        "l2_norm": probe_l2_norm(canonical_probe),
        "total_energy": float(mode_energies.sum()),
        "mode_energies": [float(value) for value in mode_energies],
    }


def legacy_split_payload(
    obj: np.ndarray,
    raw_geometry: np.ndarray,
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    *,
    detector_size: int,
    target_mean_count: float,
    poisson_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Normalized-amplitude twin from the same calibrated count builder.

    Historical legacy_v1 datasets store ``X = sqrt(Poisson(s^2 I)) / s`` with
    the object amplitude scaled by ``s`` and the probe left uncalibrated; the
    CI builder produces the identical detector statistics by scaling the probe
    with ``s = dose_amplitude_scale`` instead, so inverting that one scalar on
    fresh counts reproduces the legacy normalized amplitude exactly.
    """
    patches = M.extract_object_patches(obj, xcoords, ycoords, detector_size)
    measurement = M.generate_ci_count_measurement(
        patches,
        raw_geometry,
        target_mean_count=target_mean_count,
        rng=np.random.default_rng(poisson_seed),
    )
    amplitude = (
        np.sqrt(measurement.counts.astype(np.float64))
        / measurement.dose_amplitude_scale
    ).astype(np.float32)
    metadata = {
        "schema_version": "1.0.0",
        "scale_contract_version": LEGACY_SCALE_CONTRACT,
        "measurement_domain": NORMALIZED_AMPLITUDE,
        "probe_gauge": "legacy_normalized",
        "object_units": M.ABSOLUTE_OBJECT_UNITS,
        "normalization": {
            "method": "sqrt_fresh_poisson_counts_over_dose_amplitude_scale",
            "target_mean_count": float(target_mean_count),
            "dose_amplitude_scale": measurement.dose_amplitude_scale,
            "poisson_seed": int(poisson_seed),
        },
    }
    payload = {
        "xcoords": xcoords,
        "ycoords": ycoords,
        "xcoords_start": xcoords.copy(),
        "ycoords_start": ycoords.copy(),
        "diff3d": amplitude,
        "probeGuess": raw_geometry,
        "objectGuess": np.asarray(obj),
        "scan_index": np.zeros(len(xcoords), dtype=np.int64),
        "ground_truth_patches": patches[..., None],
    }
    return payload, metadata


def generator_commit() -> str:
    """Resolve the full generating commit recorded in provenance."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MaterializationError(
            f"cannot resolve generator commit via git rev-parse: {exc}"
        ) from exc
    commit = result.stdout.strip()
    if not _COMMIT_RE.fullmatch(commit):
        raise MaterializationError(f"unexpected generator commit {commit!r}")
    return commit


def write_guarded(output_root: Path, outputs: dict[str, bytes]) -> None:
    """Write missing outputs; refuse to clobber any checksum-mismatched file."""
    output_root.mkdir(parents=True, exist_ok=True)
    mismatched = []
    for name, blob in outputs.items():
        path = output_root / name
        if not path.exists():
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != (
            hashlib.sha256(blob).hexdigest()
        ):
            mismatched.append(name)
    if mismatched:
        raise MaterializationError(
            "refusing to overwrite existing output(s) whose checksum differs "
            f"from the deterministic materialization: {sorted(mismatched)}"
        )
    for name, blob in outputs.items():
        path = output_root / name
        if not path.exists():
            path.write_bytes(blob)

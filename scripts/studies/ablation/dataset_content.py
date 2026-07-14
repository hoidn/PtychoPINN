"""Cached immutable artifact hashing, NPZ analysis, dose, probe, and grouping."""

from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .dataset_schema import (
    TASK10_RAW_PROBE_GEOMETRY_KEY,
    DatasetDescriptor,
    DatasetError,
    DoseStatistics,
    MeasurementDomain,
    _PROBE_SCALAR_PHASE_REL_TOL,
    _PROBE_SCALAR_RESIDUAL_REL_TOL,
    _RUNTIME_GROUPING_GRID_SIZE,
    _RUNTIME_GROUPING_K,
)


@dataclass(frozen=True)
class _SplitContent:
    probe: np.ndarray
    probe_l2_norm: float
    probe_total_energy: float
    probe_mode_energies: tuple[float, ...]
    xcoords: np.ndarray
    ycoords: np.ndarray
    x_sha256: str
    y_sha256: str
    measurement_sha256: str
    initialization_sha256: str
    raw_probe_geometry: np.ndarray | None
    raw_probe_geometry_sha256: str | None
    truth_sha256: str | None
    truth_dtype: str | None
    truth_is_complex: bool | None
    dose: DoseStatistics | None


@dataclass(frozen=True)
class _ContentValidation:
    descriptor: DatasetDescriptor
    train: _SplitContent
    test: _SplitContent
    truth_sha256: str | None


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> FileIdentity:
        return cls(
            device=int(getattr(value, "st_dev", 0)),
            inode=int(getattr(value, "st_ino", 0)),
            size=int(value.st_size),
            mtime_ns=int(
                getattr(value, "st_mtime_ns", round(value.st_mtime * 1_000_000_000))
            ),
            ctime_ns=int(
                getattr(value, "st_ctime_ns", round(value.st_ctime * 1_000_000_000))
            ),
        )


class SnapshotReader:
    """Hash and parse one stable file snapshot through one open descriptor."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def read_json(self, expected: str, field: str) -> Any:
        with self._open() as (handle, before):
            digest, payload = self._hash(handle, capture=True)
            self._check_digest(digest, expected, field)
            self._after_hash(handle)
            try:
                parsed = json.loads(payload)
            except (UnicodeError, json.JSONDecodeError) as exc:
                self._ensure_stable(handle, before)
                raise DatasetError(f"provenance must be valid JSON: {exc}") from exc
            self._ensure_stable(handle, before)
            return parsed

    def read_npz(
        self,
        expected: str,
        field: str,
        required_keys: set[str],
        optional_keys: AbstractSet[str] = frozenset(),
    ) -> dict[str, np.ndarray]:
        with self._open() as (handle, before):
            digest, _ = self._hash(handle, capture=False)
            self._check_digest(digest, expected, field)
            self._after_hash(handle)
            handle.seek(0)
            try:
                with np.load(handle, allow_pickle=False) as archive:
                    available = set(archive.files)
                    selected = (required_keys | optional_keys) & available
                    arrays = {
                        key: np.ascontiguousarray(np.asarray(archive[key]))
                        for key in selected
                    }
            except DatasetError:
                self._ensure_stable(handle, before)
                raise
            except (OSError, ValueError, TypeError, zipfile.BadZipFile) as exc:
                self._ensure_stable(handle, before)
                raise DatasetError(
                    f"cannot load NPZ snapshot without pickle {self.path}: {exc}"
                ) from exc
            self._ensure_stable(handle, before)
            return arrays

    def _open(self) -> _OpenSnapshot:
        return _OpenSnapshot(self.path)

    def _hash(self, handle: BinaryIO, *, capture: bool) -> tuple[str, bytes]:
        handle.seek(0)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            if capture:
                chunks.append(block)
        return digest.hexdigest(), b"".join(chunks)

    def _check_digest(self, observed: str, expected: str, field: str) -> None:
        if observed != expected:
            raise DatasetError(
                f"{field} mismatch for {self.path}: expected {expected}, observed {observed}"
            )

    def _after_hash(self, _handle: BinaryIO) -> None:
        """Race-test hook invoked after hashing and before parsing."""

    def _ensure_stable(self, handle: BinaryIO, before: FileIdentity) -> None:
        try:
            after = FileIdentity.from_stat(os.fstat(handle.fileno()))
            current = FileIdentity.from_stat(self.path.stat())
        except OSError as exc:
            raise DatasetError(
                f"immutable snapshot path changed during validation: {self.path}"
            ) from exc
        if after != before or current != before:
            raise DatasetError(
                f"immutable snapshot changed or was replaced during validation: {self.path}"
            )


class _OpenSnapshot:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: BinaryIO | None = None

    def __enter__(self) -> tuple[BinaryIO, FileIdentity]:
        try:
            handle = self.path.open("rb")
        except OSError as exc:
            raise DatasetError(
                f"cannot open immutable snapshot {self.path}: {exc}"
            ) from exc
        self.handle = handle
        return handle, FileIdentity.from_stat(os.fstat(handle.fileno()))

    def __exit__(self, *_args: object) -> None:
        if self.handle is not None:
            self.handle.close()


@dataclass
class ArtifactCache:
    """Preflight cache of compact analysis keyed by immutable artifact identity."""

    def __post_init__(self) -> None:
        self._splits: dict[tuple[object, ...], _SplitContent] = {}
        self._images: dict[tuple[object, ...], str] = {}
        self._json: dict[tuple[Path, str], Any] = {}
        self._neighbors: dict[tuple[object, ...], np.ndarray] = {}

    def split(self, identity: tuple[object, ...]) -> _SplitContent | None:
        return self._splits.get(identity)

    def remember_split(
        self, identity: tuple[object, ...], content: _SplitContent
    ) -> _SplitContent:
        self._splits[identity] = content
        return content

    def json(self, path: Path, expected: str, field: str) -> Any:
        identity = (path.resolve(), expected)
        if identity not in self._json:
            self._json[identity] = SnapshotReader(path).read_json(expected, field)
        return self._json[identity]

    def image(self, identity: tuple[object, ...]) -> str | None:
        return self._images.get(identity)

    def remember_image(self, identity: tuple[object, ...], digest: str) -> str:
        self._images[identity] = digest
        return digest

    def neighbors(self, identity: tuple[object, ...]) -> np.ndarray | None:
        return self._neighbors.get(identity)

    def remember_neighbors(
        self, identity: tuple[object, ...], neighbors: np.ndarray
    ) -> np.ndarray:
        self._neighbors[identity] = neighbors
        return neighbors


def file_sha256(path: Path) -> str:
    """Hash a file in bounded blocks."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise DatasetError(f"cannot read immutable dataset file {path}: {exc}") from exc
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Return the repository dtype-, shape-, and byte-aware array digest."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(b"|")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def canonicalize_probe(
    probe: np.ndarray, detector_shape: tuple[int, int]
) -> np.ndarray:
    """Canonicalize accepted ``(N,N)``, ``(N,N,1)``, and ``(P,N,N)`` probes."""
    array = np.asarray(probe)
    height, width = detector_shape
    if array.ndim == 2 and tuple(array.shape) == detector_shape:
        canonical = array[None]
    elif array.ndim == 3 and tuple(array.shape) == (height, width, 1):
        canonical = np.moveaxis(array, -1, 0)
    elif (
        array.ndim == 3
        and array.shape[0] > 0
        and tuple(array.shape[1:]) == detector_shape
    ):
        canonical = array
    else:
        raise DatasetError(
            "probe must have shape (N,N), (N,N,1), or (P,N,N); "
            f"got {tuple(array.shape)} for detector_shape={detector_shape}"
        )
    if not np.issubdtype(canonical.dtype, np.number):
        raise DatasetError("probe must have a numeric dtype")
    if not _all_finite(canonical):
        raise DatasetError("probe must contain only finite values")
    return np.ascontiguousarray(canonical)


def probe_l2_norm(probe: np.ndarray) -> float:
    """Return the deterministic float64 L2 norm used by provenance v1."""
    canonical = np.asarray(probe).astype(np.complex128, copy=False)
    return float(np.sqrt(np.sum(np.abs(canonical) ** 2, dtype=np.float64)))


def probe_geometry_sha256(probe: np.ndarray) -> str:
    """Hash scale-invariant geometry after unit-energy normalization.

    Seven decimal places absorb float32 roundoff from calibrated amplitude
    scaling while remaining far tighter than the bundle geometry tolerance.
    """
    canonical = np.asarray(probe).astype(np.complex128, copy=False)
    energy = float(np.sum(np.abs(canonical) ** 2, dtype=np.float64))
    if not math.isfinite(energy) or energy <= 0:
        raise DatasetError("probe geometry requires positive finite total energy")
    normalized = canonical / math.sqrt(energy)
    quantized = np.round(normalized.real, 7) + 1j * np.round(normalized.imag, 7)
    return array_sha256(np.asarray(quantized, dtype=np.complex128))


def validate_positive_real_scalar_multiple(
    calibrated_value: np.ndarray,
    raw_value: np.ndarray,
    *,
    label: str,
) -> None:
    """Require one dimensionless positive-real scale across a whole probe."""
    calibrated = np.asarray(calibrated_value).astype(np.complex128, copy=False)
    raw = np.asarray(raw_value).astype(np.complex128, copy=False)
    if calibrated.shape != raw.shape:
        raise DatasetError(f"{label} shape/modes do not match declared geometry")
    raw_norm = float(np.linalg.norm(raw.ravel()))
    calibrated_norm = float(np.linalg.norm(calibrated.ravel()))
    if not _positive_finite(raw_norm) or not _positive_finite(calibrated_norm):
        raise DatasetError(f"{label} norms must be positive and finite")
    alpha = np.complex128(np.vdot(raw, calibrated) / (raw_norm * raw_norm))
    magnitude = float(abs(alpha))
    if (
        not _positive_finite(magnitude)
        or alpha.real <= 0
        or abs(float(alpha.imag)) / magnitude > _PROBE_SCALAR_PHASE_REL_TOL
    ):
        raise DatasetError(
            f"{label} is not a positive-real scalar multiple of declared geometry"
        )
    expected = alpha * raw
    _validate_scalar_residual(calibrated, expected, calibrated_norm, label)
    for mode in range(calibrated.shape[0]):
        mode_norm = float(np.linalg.norm(calibrated[mode].ravel()))
        if not _positive_finite(mode_norm):
            raise DatasetError(f"{label} mode {mode} must have positive finite norm")
        _validate_scalar_residual(
            calibrated[mode], expected[mode], mode_norm, f"{label} mode {mode}"
        )


def _positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0


def _validate_scalar_residual(
    calibrated: np.ndarray,
    expected: np.ndarray,
    calibrated_norm: float,
    label: str,
) -> None:
    residual = float(np.linalg.norm((calibrated - expected).ravel()) / calibrated_norm)
    if math.isfinite(residual) and residual <= _PROBE_SCALAR_RESIDUAL_REL_TOL:
        return
    raise DatasetError(
        f"{label} does not share one positive-real scalar; geometry or mode power changed"
    )


def nearest_neighbor_indices(
    xcoords: np.ndarray, ycoords: np.ndarray, *, K: int = 6
) -> np.ndarray:
    """Match the runtime Nearest primitive without importing ptycho_torch."""
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]

    points = np.column_stack((xcoords, ycoords))
    tree = cKDTree(points)
    _, indices = tree.query(points, k=K + 1)
    return np.asarray(indices)


def analyze_descriptor_content(
    descriptor: DatasetDescriptor,
    cache: ArtifactCache,
    *,
    requires_raw_probe_geometry: bool,
) -> _ContentValidation:
    """Validate one descriptor's artifacts using one bundle-scoped cache."""
    train_content = _validate_split(
        descriptor,
        "train",
        descriptor.train,
        descriptor.train_sha256,
        cache,
        truth_key=None,
        requires_raw_probe_geometry=requires_raw_probe_geometry,
    )
    test_content = _validate_split(
        descriptor,
        "test",
        descriptor.test,
        descriptor.test_sha256,
        cache,
        truth_key=(
            descriptor.truth_key
            if descriptor.truth_location == "embedded_test"
            else None
        ),
        requires_raw_probe_geometry=requires_raw_probe_geometry,
    )
    probes_identical = array_sha256(train_content.probe) == array_sha256(
        test_content.probe
    )
    if descriptor.probe.sha256 is not None and not probes_identical:
        raise DatasetError(
            "probe sha256 shorthand requires identical canonical train/test probes"
        )
    if descriptor.probe.sha256 is None and probes_identical:
        raise DatasetError(
            "split probe hashes require differing canonical train/test probes; "
            "use sha256 shorthand for identical probes"
        )

    truth_sha256: str | None = None
    if descriptor.truth_location == "embedded_test":
        truth_sha256 = test_content.truth_sha256
    elif descriptor.truth_location == "external_npz":
        assert (
            descriptor.reference is not None
            and descriptor.reference_sha256 is not None
            and descriptor.truth_key is not None
        )
        truth_sha256 = _validate_npz_image(
            descriptor.reference,
            descriptor.reference_sha256,
            descriptor.truth_key,
            "external reference",
            require_complex=(
                descriptor.kind == "synthetic" and descriptor.truth == "object_truth"
            ),
            cache=cache,
        )

    if descriptor.dose is not None:
        assert train_content.dose is not None and test_content.dose is not None
        _compare_dose(descriptor.dose.train, train_content.dose, "dose.train")
        _compare_dose(descriptor.dose.test, test_content.dose, "dose.test")
    return _ContentValidation(
        descriptor=descriptor,
        train=train_content,
        test=test_content,
        truth_sha256=truth_sha256,
    )


def _validate_split(
    descriptor: DatasetDescriptor,
    split: str,
    path: Path,
    expected_file_hash: str,
    cache: ArtifactCache,
    *,
    truth_key: str | None,
    requires_raw_probe_geometry: bool,
) -> _SplitContent:
    expected_probe_hash = (
        descriptor.probe.sha256
        if descriptor.probe.sha256 is not None
        else descriptor.probe.train_sha256
        if split == "train"
        else descriptor.probe.test_sha256
    )
    identity = (
        path.resolve(),
        expected_file_hash,
        descriptor.measurement_key,
        descriptor.probe_key,
        descriptor.x_key,
        descriptor.y_key,
        descriptor.detector_shape,
        descriptor.measurement_domain,
        descriptor.probe_modes,
        expected_probe_hash,
        truth_key,
        requires_raw_probe_geometry,
    )
    cached = cache.split(identity)
    if cached is not None:
        _validate_cached_split_constraints(cached, descriptor, split, truth_key)
        return cached
    required_keys = {
        descriptor.measurement_key,
        descriptor.probe_key,
        descriptor.x_key,
        descriptor.y_key,
        "objectGuess",
    }
    if truth_key is not None:
        required_keys.add(truth_key)
    raw_keys = (
        {TASK10_RAW_PROBE_GEOMETRY_KEY} if requires_raw_probe_geometry else frozenset()
    )
    archive = SnapshotReader(path).read_npz(
        expected_file_hash,
        f"{split}_sha256",
        required_keys,
        raw_keys,
    )
    measurement = _npz_array(
        archive, descriptor.measurement_key, "measurement_key", split
    )
    probe = _npz_array(archive, descriptor.probe_key, "probe_key", split)
    xcoords = _npz_array(archive, descriptor.x_key, "x_key", split)
    ycoords = _npz_array(archive, descriptor.y_key, "y_key", split)
    initialization = _npz_array(archive, "objectGuess", "runtime initialization", split)
    raw_probe_geometry = archive.get(TASK10_RAW_PROBE_GEOMETRY_KEY)

    canonical_measurement = _canonical_measurement(
        measurement, descriptor.detector_shape, split
    )
    sample_count = canonical_measurement.shape[0]
    _validate_coordinate(xcoords, sample_count, descriptor.x_key, split)
    _validate_coordinate(ycoords, sample_count, descriptor.y_key, split)
    _validate_initialization_object(initialization, split)
    _validate_measurement(canonical_measurement, descriptor.measurement_domain, split)
    canonical_probe = canonicalize_probe(probe, descriptor.detector_shape)
    if canonical_probe.shape[0] != descriptor.probe_modes:
        raise DatasetError(
            f"{split} probe_modes mismatch: declared {descriptor.probe_modes}, "
            f"observed {canonical_probe.shape[0]}"
        )
    observed_probe_hash = array_sha256(canonical_probe)
    assert expected_probe_hash is not None
    if observed_probe_hash != expected_probe_hash:
        raise DatasetError(
            f"{split} probe sha256 mismatch: expected {expected_probe_hash}, "
            f"observed {observed_probe_hash}"
        )
    mode_energies_array = np.sum(
        np.abs(canonical_probe.astype(np.complex128, copy=False)) ** 2,
        axis=(-2, -1),
        dtype=np.float64,
    )
    if not np.isfinite(mode_energies_array).all() or np.any(mode_energies_array <= 0):
        raise DatasetError(f"{split} probe mode energy must be finite and positive")
    total_energy = float(mode_energies_array.sum())
    if not math.isfinite(total_energy) or total_energy <= 0:
        raise DatasetError(f"{split} probe total energy must be finite and positive")
    dose = (
        _computed_dose(canonical_measurement)
        if descriptor.measurement_domain == "count_intensity"
        else None
    )
    canonical_raw_geometry = (
        _validate_raw_probe_geometry(
            raw_probe_geometry,
            descriptor.detector_shape,
            descriptor.probe_modes,
            split,
        )
        if raw_probe_geometry is not None
        else None
    )
    truth_array = archive.get(truth_key) if truth_key is not None else None
    truth_sha256 = (
        _validate_image_array(
            truth_array,
            truth_key,
            f"{split} truth",
            require_complex=False,
        )
        if truth_key is not None and truth_array is not None
        else None
    )
    content = _SplitContent(
        probe=canonical_probe,
        probe_l2_norm=probe_l2_norm(canonical_probe),
        probe_total_energy=total_energy,
        probe_mode_energies=tuple(float(value) for value in mode_energies_array),
        xcoords=np.ascontiguousarray(xcoords),
        ycoords=np.ascontiguousarray(ycoords),
        x_sha256=array_sha256(xcoords),
        y_sha256=array_sha256(ycoords),
        measurement_sha256=array_sha256(canonical_measurement),
        initialization_sha256=array_sha256(initialization),
        raw_probe_geometry=canonical_raw_geometry,
        raw_probe_geometry_sha256=(
            array_sha256(canonical_raw_geometry)
            if canonical_raw_geometry is not None
            else None
        ),
        truth_sha256=truth_sha256,
        truth_dtype=str(truth_array.dtype) if truth_array is not None else None,
        truth_is_complex=(
            bool(np.issubdtype(truth_array.dtype, np.complexfloating))
            if truth_array is not None
            else None
        ),
        dose=dose,
    )
    _validate_cached_split_constraints(content, descriptor, split, truth_key)
    return cache.remember_split(identity, content)


def _validate_cached_split_constraints(
    content: _SplitContent,
    descriptor: DatasetDescriptor,
    split: str,
    truth_key: str | None,
) -> None:
    expected_truth_key = (
        descriptor.truth_key
        if split == "test"
        and descriptor.truth != "none"
        and descriptor.truth_location == "embedded_test"
        else None
    )
    if truth_key != expected_truth_key:
        raise DatasetError(
            f"{split} cached truth key does not match descriptor truth role/location"
        )
    if truth_key is None:
        return
    if content.truth_sha256 is None or content.truth_dtype is None:
        raise DatasetError(f"{split} NPZ is missing truth_key key {truth_key!r}")
    if (
        descriptor.kind == "synthetic"
        and descriptor.truth == "object_truth"
        and content.truth_is_complex is not True
    ):
        raise DatasetError("synthetic object_truth must use a complex finite array")


def _validate_raw_probe_geometry(
    array: np.ndarray,
    detector_shape: tuple[int, int],
    probe_modes: int,
    split: str,
) -> np.ndarray:
    raw = np.asarray(array)
    expected_shape = (probe_modes, *detector_shape)
    if raw.shape != expected_shape:
        raise DatasetError(
            f"{split} {TASK10_RAW_PROBE_GEOMETRY_KEY} must have canonical shape "
            f"{expected_shape}; got {raw.shape}"
        )
    if not np.issubdtype(raw.dtype, np.number) or not _all_finite(raw):
        raise DatasetError(
            f"{split} {TASK10_RAW_PROBE_GEOMETRY_KEY} must be numeric and finite"
        )
    mode_energy = np.sum(
        np.abs(raw.astype(np.complex128, copy=False)) ** 2,
        axis=(-2, -1),
        dtype=np.float64,
    )
    if np.any(mode_energy <= 0) or not np.isfinite(mode_energy).all():
        raise DatasetError(
            f"{split} {TASK10_RAW_PROBE_GEOMETRY_KEY} modes need positive energy"
        )
    return np.ascontiguousarray(raw)


def _validate_runtime_grouping(
    content: _ContentValidation, cache: ArtifactCache
) -> int:
    """Prove canonical Nearest/K=6 candidate viability without sampling groups."""
    descriptor = content.descriptor
    if descriptor.grouping_max_C == 1:
        return 1
    for split, split_content in (("train", content.train), ("test", content.test)):
        candidates = _nearest_candidate_rows(split_content, cache)
        _validate_candidate_rows(
            descriptor.id,
            split,
            candidates,
            len(split_content.xcoords),
            descriptor.grouping_max_C,
        )
    return descriptor.grouping_max_C


def _nearest_candidate_rows(content: _SplitContent, cache: ArtifactCache) -> np.ndarray:
    identity = (
        content.x_sha256,
        content.y_sha256,
        "Nearest",
        _RUNTIME_GROUPING_K,
        _RUNTIME_GROUPING_GRID_SIZE,
        1,
    )
    cached = cache.neighbors(identity)
    if cached is not None:
        return cached
    try:
        candidates = nearest_neighbor_indices(
            content.xcoords,
            content.ycoords,
            K=_RUNTIME_GROUPING_K,
        )
    except (IndexError, TypeError, ValueError) as exc:
        raise DatasetError(f"canonical Nearest grouping query failed: {exc}") from exc
    return cache.remember_neighbors(identity, candidates)


def _validate_candidate_rows(
    dataset_id: str,
    split: str,
    candidates: np.ndarray,
    sample_count: int,
    requested_C: int,
) -> None:
    if sample_count < requested_C:
        raise DatasetError(
            f"dataset {dataset_id!r} {split} has {sample_count} samples, fewer than "
            f"claimed grouping C={requested_C}"
        )
    if candidates.ndim != 2 or candidates.shape[0] != sample_count:
        raise DatasetError(
            f"dataset {dataset_id!r} {split} runtime grouping C={requested_C} "
            f"returned candidate shape {candidates.shape}"
        )
    for row in candidates:
        in_bounds = row[(row >= 0) & (row < sample_count)]
        if np.unique(in_bounds).size < requested_C:
            raise DatasetError(
                f"dataset {dataset_id!r} {split} runtime grouping C={requested_C} "
                "does not have enough distinct in-bounds nearest candidates"
            )


def _canonical_measurement(
    measurement: np.ndarray, detector_shape: tuple[int, int], split: str
) -> np.ndarray:
    array = np.asarray(measurement)
    if array.ndim != 3 or tuple(array.shape[-2:]) != detector_shape:
        raise DatasetError(
            f"{split} measurement must have exact runtime shape (S,H,W), sample-first, "
            f"with detector_shape={detector_shape}; got {tuple(array.shape)}"
        )
    if array.shape[0] <= 0:
        raise DatasetError(f"{split} measurement must contain at least one sample")
    return array


def _validate_coordinate(array: np.ndarray, samples: int, key: str, split: str) -> None:
    if array.ndim != 1 or len(array) != samples:
        raise DatasetError(
            f"{split} coordinate {key!r} must be 1D with length {samples}; got {array.shape}"
        )
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or not _all_finite(array)
    ):
        raise DatasetError(
            f"{split} coordinate {key!r} must be real, non-bool, numeric, and finite"
        )


def _validate_initialization_object(array: np.ndarray, split: str) -> None:
    if array.ndim != 2 or array.size == 0:
        raise DatasetError(
            f"{split} runtime initialization objectGuess must be a nonempty 2D array"
        )
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or not _all_finite(array)
    ):
        raise DatasetError(
            f"{split} runtime initialization objectGuess must be finite real/complex numeric data"
        )


def _validate_measurement(
    measurement: np.ndarray, domain: MeasurementDomain, split: str
) -> None:
    if not _all_finite(measurement):
        raise DatasetError(f"{split} measurements must be finite")
    if domain == "count_intensity":
        if not np.issubdtype(measurement.dtype, np.integer) or np.issubdtype(
            measurement.dtype, np.bool_
        ):
            raise DatasetError(f"{split} count measurements require an integer dtype")
    elif not np.issubdtype(measurement.dtype, np.floating):
        raise DatasetError(f"{split} normalized amplitudes require a floating dtype")
    if np.any(measurement < 0):
        raise DatasetError(f"{split} measurements must be nonnegative")


def _computed_dose(measurement: np.ndarray) -> DoseStatistics:
    photons = measurement.astype(np.float64).sum(axis=(-2, -1))
    dtype_max = int(np.iinfo(measurement.dtype).max)
    return DoseStatistics(
        counts_mean=float(measurement.mean(dtype=np.float64)),
        photons_per_image_min=float(photons.min()),
        photons_per_image_mean=float(photons.mean()),
        max_observed_count=int(measurement.max()),
        dtype_max=dtype_max,
        saturation_fraction=float(
            np.count_nonzero(measurement == dtype_max) / measurement.size
        ),
    )


def deterministic_float_equal(declared: float, observed: float) -> bool:
    """Compare deterministic reductions exactly after JSON/TOML float round-trip."""
    return math.isfinite(declared) and math.isfinite(observed) and declared == observed


def _compare_dose(
    expected: DoseStatistics, observed: DoseStatistics, path: str
) -> None:
    for field in ("max_observed_count", "dtype_max"):
        if getattr(expected, field) != getattr(observed, field):
            raise DatasetError(
                f"{path}.{field} mismatch: declared {getattr(expected, field)}, "
                f"observed {getattr(observed, field)}"
            )
    for field in (
        "counts_mean",
        "photons_per_image_min",
        "photons_per_image_mean",
        "saturation_fraction",
    ):
        declared = getattr(expected, field)
        measured = getattr(observed, field)
        if not deterministic_float_equal(declared, measured):
            raise DatasetError(
                f"{path}.{field} mismatch: declared {declared}, observed {measured}"
            )


def _validate_npz_image(
    path: Path,
    expected_file_hash: str,
    key: str,
    label: str,
    *,
    require_complex: bool,
    cache: ArtifactCache,
) -> str:
    identity = (path.resolve(), expected_file_hash, key, require_complex)
    cached = cache.image(identity)
    if cached is not None:
        return cached
    archive = SnapshotReader(path).read_npz(
        expected_file_hash, f"{label}_sha256", {key}
    )
    array = _npz_array(archive, key, "truth_key", label)
    digest = _validate_image_array(array, key, label, require_complex=require_complex)
    return cache.remember_image(identity, digest)


def _validate_image_array(
    array: np.ndarray, key: str, label: str, *, require_complex: bool
) -> str:
    if array.ndim != 2 or array.size == 0:
        raise DatasetError(f"{label} key {key!r} must be a nonempty 2D array")
    if not np.issubdtype(array.dtype, np.number) or not _all_finite(array):
        raise DatasetError(f"{label} key {key!r} must be numeric and finite")
    if require_complex and not np.issubdtype(array.dtype, np.complexfloating):
        raise DatasetError("synthetic object_truth must use a complex finite array")
    return array_sha256(array)


def _npz_array(archive: Any, key: str, descriptor_field: str, split: str) -> np.ndarray:
    if key not in archive:
        raise DatasetError(f"{split} NPZ is missing {descriptor_field} key {key!r}")
    try:
        return np.asarray(archive[key])
    except ValueError as exc:
        raise DatasetError(
            f"{split} NPZ key {key!r} cannot load without pickle: {exc}"
        ) from exc


def _verify_file_hash(path: Path, expected: str, field: str) -> None:
    observed = file_sha256(path)
    if observed != expected:
        raise DatasetError(
            f"{field} mismatch for {path}: expected {expected}, observed {observed}"
        )


def _all_finite(array: np.ndarray) -> bool:
    try:
        return bool(np.isfinite(array).all())
    except TypeError:
        return False

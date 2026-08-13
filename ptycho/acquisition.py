"""Framework-neutral records for acquisition data crossing backend boundaries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional
import json
import warnings
import zipfile

import numpy as np


_DIFFRACTION_KEYS = ("diff3d", "diffraction")
_MEASUREMENT_PAIRS = {
    (None, None),
    ("legacy_v1", "normalized_amplitude"),
    ("ci_intensity_v2", "count_intensity"),
}


def _finite_numeric_vector(
    array: np.ndarray, *, name: str, source: str
) -> np.ndarray:
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{source}: {name} must be a finite numeric vector.")
    return array


def _validate_measurement_values(
    array: np.ndarray, *, name: str, source: str, nonnegative: bool = False
) -> None:
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{source}: {name} must be a finite numeric array.")
    if nonnegative and (np.iscomplexobj(array) or np.any(array < 0)):
        raise ValueError(f"{source}: {name} must contain nonnegative real values.")


def _canonical_diffraction_shape(
    shape: tuple[int, ...], n_coords: int, source: str
) -> tuple[tuple[int, int, int], bool]:
    if len(shape) == 4 and shape[-1] == 1:
        shape = shape[:-1]
    if len(shape) != 3:
        raise ValueError(
            f"{source}: diffraction data must be 3D (M, H, W), legacy "
            f"(H, W, M), or have one trailing singleton channel; got {shape}."
        )

    canonical_square = shape[1] == shape[2]
    legacy_square = shape[0] == shape[1]
    canonical_match = shape[0] == n_coords
    legacy_match = shape[2] == n_coords
    if canonical_square != legacy_square:
        legacy = legacy_square
    elif canonical_match != legacy_match:
        legacy = legacy_match
    else:
        legacy = shape[2] > max(shape[0], shape[1])
    canonical = (shape[2], shape[0], shape[1]) if legacy else tuple(shape)
    return canonical, legacy


def _canonical_diffraction(array: np.ndarray, n_coords: int, source: str) -> np.ndarray:
    canonical_shape, legacy = _canonical_diffraction_shape(
        array.shape, n_coords, source
    )
    if array.ndim == 4:
        array = array[..., 0]
    canonical = np.transpose(array, (2, 0, 1)) if legacy else array
    assert canonical.shape == canonical_shape
    return canonical


def _align_coordinates(
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    n_diffraction: int,
    source: str,
    policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    if policy not in {"strict", "trailing"}:
        raise ValueError("coordinate_policy must be 'strict' or 'trailing'")
    if xcoords.ndim != 1 or ycoords.ndim != 1:
        raise ValueError(
            f"{source}: xcoords shape {xcoords.shape} and ycoords shape "
            f"{ycoords.shape} must be one-dimensional."
        )
    _finite_numeric_vector(xcoords, name="xcoords", source=source)
    _finite_numeric_vector(ycoords, name="ycoords", source=source)
    if len(xcoords) != len(ycoords):
        raise ValueError(
            f"{source}: xcoords={len(xcoords)} and ycoords={len(ycoords)} "
            "must have equal lengths."
        )
    n_coords = len(xcoords)
    if n_coords == n_diffraction:
        return xcoords, ycoords
    if n_coords < n_diffraction:
        raise ValueError(
            f"{source}: {n_coords} scan positions for {n_diffraction} diffraction "
            "patterns. Every pattern needs a position."
        )
    if policy == "strict":
        raise ValueError(
            f"{source}: {n_coords} scan positions for {n_diffraction} diffraction "
            "patterns under strict coordinate policy."
        )
    warnings.warn(
        f"{source}: {n_coords} scan positions for {n_diffraction} diffraction "
        f"patterns; dropping the trailing {n_coords - n_diffraction} positions.",
        RuntimeWarning,
        stacklevel=2,
    )
    return xcoords[:n_diffraction], ycoords[:n_diffraction]


def _aligned_vector(
    value: np.ndarray,
    *,
    name: str,
    n_diffraction: int,
    n_coordinates: int,
    source: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) not in {n_diffraction, n_coordinates}:
        raise ValueError(
            f"{source}: {name} must have shape ({n_diffraction},)"
            + (
                f" or ({n_coordinates},) under trailing coordinate policy"
                if n_coordinates != n_diffraction
                else ""
            )
            + f"; got {array.shape}."
        )
    return array[:n_diffraction]


def _identity_vector(
    value: Optional[np.ndarray],
    *,
    name: str,
    n_diffraction: int,
    n_coordinates: int,
    source: str,
) -> np.ndarray:
    array = (
        None
        if value is None
        else _aligned_vector(
            value,
            name=name,
            n_diffraction=n_diffraction,
            n_coordinates=n_coordinates,
            source=source,
        )
    )
    return canonicalize_identity_index(
        array, name=name, length=n_diffraction, source=source
    )


def canonicalize_identity_index(
    values: Optional[np.ndarray], *, name: str, length: int, source: str = ""
) -> np.ndarray:
    """Validate one frame-identity vector before canonical int64 storage."""

    prefix = f"{source}: " if source else ""
    if values is None:
        return np.zeros(length, dtype=np.int64)
    array = np.asarray(values)
    if array.shape != (length,):
        raise ValueError(f"{prefix}{name} must have shape ({length},).")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
        array.dtype, np.integer
    ):
        raise ValueError(f"{prefix}{name} must contain nonnegative integers.")
    if np.issubdtype(array.dtype, np.unsignedinteger) and array.size:
        if int(array.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"{prefix}{name} must contain nonnegative integers.")
    canonical = np.asarray(array, dtype=np.int64)
    if np.any(canonical < 0):
        raise ValueError(f"{prefix}{name} must contain nonnegative integers.")
    return canonical


def _optional_scalar(data: Any, key: str) -> Any:
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar; got shape {value.shape}.")
    return value.item()


def _experiment_id(value: Any, source: str) -> Optional[int]:
    if value is None:
        return None
    array = np.asarray(value)
    if (
        array.shape != ()
        or np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.integer)
        or array.item() < 0
    ):
        raise ValueError(f"{source}: experiment_id must be a nonnegative integer scalar.")
    return int(array.item())


def _object_amplitude_scale(data: Any, source: str) -> Optional[np.float64]:
    if "object_amplitude_scale" not in data:
        return None
    array = np.asarray(data["object_amplitude_scale"])
    if (
        array.shape != ()
        or array.dtype != np.dtype(np.float64)
        or not np.isfinite(array.item())
        or array.item() <= 0
    ):
        raise ValueError(
            f"{source}: object_amplitude_scale must be a positive finite "
            "float64 scalar."
        )
    return np.float64(array.item())


def _probe_spatial_shape(
    probe_shape: tuple[int, ...], source: str
) -> tuple[int, int]:
    if len(probe_shape) == 2:
        probe_spatial_shape = probe_shape
    elif len(probe_shape) == 3 and probe_shape[-1] == 1:
        probe_spatial_shape = probe_shape[:2]
    elif len(probe_shape) == 3:
        probe_spatial_shape = probe_shape[1:]
    else:
        probe_spatial_shape = None
    if probe_spatial_shape is None or probe_spatial_shape[0] != probe_spatial_shape[1]:
        raise ValueError(
            f"{source}: probeGuess shape {probe_shape} must be (N, N), "
            "(P, N, N), or (N, N, 1)."
        )
    return probe_spatial_shape


def _validate_probe_shape(
    probe_shape: tuple[int, ...],
    spatial_shape: tuple[int, int],
    source: str,
) -> None:
    height, width = spatial_shape
    if height != width:
        raise ValueError(
            f"{source}: diffraction/diff3d frames must be square; got {spatial_shape}."
        )
    if _probe_spatial_shape(probe_shape, source) != spatial_shape:
        raise ValueError(
            f"{source}: probeGuess shape {probe_shape} must match diffraction N."
        )


def _validate_raw_shapes(data: Any, diff3d: np.ndarray, source: str) -> None:
    spatial_shape = diff3d.shape[1:]
    _validate_probe_shape(data["probeGuess"].shape, spatial_shape, source)
    _validate_measurement_values(
        diff3d, name="diffraction", source=source, nonnegative=True
    )
    _validate_measurement_values(
        data["probeGuess"], name="probeGuess", source=source
    )

    if "objectGuess" in data and data["objectGuess"].ndim != 2:
        raise ValueError(
            f"{source}: objectGuess must be 2D; got {data['objectGuess'].shape}."
        )
    if "label" in data and data["label"].shape != diff3d.shape:
        raise ValueError(
            f"{source}: label must have shape {diff3d.shape}; "
            f"got {data['label'].shape}."
        )
    if "probe_simulated" in data and data["probe_simulated"].shape != spatial_shape:
        raise ValueError(
            f"{source}: probe_simulated must have shape {spatial_shape}; "
            f"got {data['probe_simulated'].shape}."
        )


def _truth_patches(
    data: Any,
    diffraction_shape: tuple[int, ...],
    source: str,
    policy: str,
) -> Optional[np.ndarray]:
    if policy not in {"strict", "drop_incompatible"}:
        raise ValueError("truth_policy must be 'strict' or 'drop_incompatible'")
    if "Y" not in data:
        return None
    if data["Y"].shape == diffraction_shape:
        return data["Y"]
    if policy == "drop_incompatible":
        warnings.warn(
            f"Ignoring NPZ 'Y' with incompatible shape {data['Y'].shape}; "
            f"expected {diffraction_shape}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    raise ValueError(
        f"{source}: Y must have shape {diffraction_shape}; got {data['Y'].shape}."
    )


def _metadata(data: Any, source: str) -> Any:
    if "_metadata" not in data:
        return None
    try:
        value = np.asarray(data["_metadata"])
    except ValueError as exc:
        raise ValueError(
            f"{source}: _metadata object arrays are unsupported without pickle."
        ) from exc
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(
            f"{source}: _metadata must be a scalar string or bytes JSON value."
        )
    encoded = value.item()
    if isinstance(encoded, bytes):
        try:
            encoded = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{source}: _metadata bytes must be UTF-8 JSON.") from exc
    try:
        return json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: _metadata contains invalid JSON: {exc}.") from exc


@dataclass(frozen=True)
class AcquisitionRecord:
    """The NumPy acquisition state needed to reconstruct a ``RawData`` adapter.

    This record deliberately contains no loading, grouping, tensor conversion, or
    backend behavior. Those operations remain owned by their existing adapters.
    """

    xcoords: np.ndarray
    ycoords: np.ndarray
    xcoords_start: Optional[np.ndarray]
    ycoords_start: Optional[np.ndarray]
    diff3d: Optional[np.ndarray]
    probeGuess: Optional[np.ndarray]
    scan_index: Optional[np.ndarray]
    objectGuess: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None
    norm_Y_I: Any = None
    metadata: Any = None
    sample_indices: Optional[np.ndarray] = None
    subsample_seed: Optional[int] = None
    object_index: Optional[np.ndarray] = None
    probe_simulated: Optional[np.ndarray] = None
    object_amplitude_scale: Optional[np.float64] = None
    label: Optional[np.ndarray] = None
    scale_contract_version: Optional[str] = None
    measurement_domain: Optional[str] = None
    experiment_id: Optional[int] = None

    @classmethod
    def from_raw_data(cls, raw_data: Any) -> "AcquisitionRecord":
        """Snapshot the acquisition fields consumed by the RawData bridge."""

        return cls(
            xcoords=raw_data.xcoords,
            ycoords=raw_data.ycoords,
            xcoords_start=raw_data.xcoords_start,
            ycoords_start=raw_data.ycoords_start,
            diff3d=raw_data.diff3d,
            probeGuess=raw_data.probeGuess,
            scan_index=raw_data.scan_index,
            object_index=getattr(raw_data, "object_index", None),
            objectGuess=raw_data.objectGuess,
            Y=getattr(raw_data, "Y", None),
            probe_simulated=getattr(raw_data, "probe_simulated", None),
            object_amplitude_scale=getattr(raw_data, "object_amplitude_scale", None),
            label=getattr(raw_data, "label", None),
            scale_contract_version=getattr(raw_data, "scale_contract_version", None),
            measurement_domain=getattr(raw_data, "measurement_domain", None),
            experiment_id=getattr(raw_data, "experiment_id", None),
            norm_Y_I=getattr(raw_data, "norm_Y_I", None),
            metadata=getattr(raw_data, "metadata", None),
            sample_indices=getattr(raw_data, "sample_indices", None),
            subsample_seed=getattr(raw_data, "subsample_seed", None),
        )


@dataclass(frozen=True)
class AcquisitionSelection:
    """Deterministic source rows and their selection provenance."""

    source_indices: np.ndarray
    seed: Optional[int]
    mode: str


def _selection_result(
    source_indices: np.ndarray, seed: Optional[int], mode: str
) -> AcquisitionSelection:
    indices = np.array(source_indices, dtype=np.int64, copy=True)
    indices.setflags(write=False)
    return AcquisitionSelection(indices, seed, mode)


@dataclass(frozen=True)
class AcquisitionHeader:
    """Small NPZ metadata needed to size and validate acquisition adapters."""

    diffraction_shape: tuple[int, int, int]
    xcoords: np.ndarray
    ycoords: np.ndarray
    object_index: np.ndarray
    probe_shape: Optional[tuple[int, ...]]
    object_shape: Optional[tuple[int, ...]]
    label_shape: Optional[tuple[int, ...]]


def read_npz_array_shape(source: Any, key: str) -> Optional[tuple[int, ...]]:
    """Read one embedded NPY shape without loading its array payload."""

    try:
        with zipfile.ZipFile(source) as archive:
            try:
                npy = archive.open(f"{key}.npy")
            except KeyError:
                return None
            with npy:
                version = np.lib.format.read_magic(npy)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(npy)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(npy)
                elif version == (3, 0):
                    shape, _, _ = np.lib.format.read_array_header_3_0(npy)
                else:
                    raise ValueError(f"{source}: unsupported NPY version {version}.")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{source}: invalid NPZ archive.") from exc
    return tuple(shape)


def inspect_probe_size(source: Any) -> int:
    """Read the detector size from a raw or grouped NPZ probe header."""

    shape = read_npz_array_shape(source, "probeGuess")
    if shape is None:
        raise ValueError(f"{source}: missing required key probeGuess.")
    return _probe_spatial_shape(shape, str(source))[0]


def inspect_acquisition(
    source: Any, *, coordinate_policy: str = "strict"
) -> AcquisitionHeader:
    """Inspect acquisition shapes and aligned coordinates without loading images."""

    with np.load(source) as data:
        try:
            xcoords = data["xcoords"]
            ycoords = data["ycoords"]
        except KeyError as exc:
            raise ValueError(f"{source}: missing required coordinate key {exc}.") from exc
        object_index = data["object_index"] if "object_index" in data else None

    shapes = {
        key: shape
        for key in _DIFFRACTION_KEYS
        if (shape := read_npz_array_shape(source, key)) is not None
    }
    if not shapes:
        raise ValueError(
            f"{source}: missing diffraction data; expected 'diff3d' or 'diffraction'."
        )
    probe_shape = read_npz_array_shape(source, "probeGuess")
    if probe_shape is None:
        raise ValueError(f"{source}: missing required key probeGuess.")
    canonical_shapes = {
        key: _canonical_diffraction_shape(shape, len(xcoords), str(source))[0]
        for key, shape in shapes.items()
    }
    if len(set(canonical_shapes.values())) != 1:
        raise ValueError(
            f"{source}: 'diff3d' and 'diffraction' have conflicting canonical shapes."
        )
    diffraction_shape = canonical_shapes[
        next(key for key in _DIFFRACTION_KEYS if key in canonical_shapes)
    ]
    _validate_probe_shape(probe_shape, diffraction_shape[1:], str(source))
    n_coordinates = len(xcoords)
    xcoords, ycoords = _align_coordinates(
        xcoords,
        ycoords,
        diffraction_shape[0],
        str(source),
        coordinate_policy,
    )
    return AcquisitionHeader(
        diffraction_shape=diffraction_shape,
        xcoords=xcoords,
        ycoords=ycoords,
        object_index=_identity_vector(
            object_index,
            name="object_index",
            n_diffraction=diffraction_shape[0],
            n_coordinates=n_coordinates,
            source=str(source),
        ),
        probe_shape=probe_shape,
        object_shape=read_npz_array_shape(source, "objectGuess"),
        label_shape=read_npz_array_shape(source, "label"),
    )


def decode_acquisition(
    source: Any,
    *,
    coordinate_policy: str = "strict",
    truth_policy: str = "strict",
    experiment_id: Optional[int] = None,
) -> AcquisitionRecord:
    """Decode a canonical standalone NPZ into a neutral acquisition record."""

    with np.load(source) as data:
        missing = [key for key in ("xcoords", "ycoords", "probeGuess") if key not in data]
        if missing:
            raise ValueError(f"{source}: missing required key(s): {', '.join(missing)}.")
        xcoords = data["xcoords"]
        ycoords = data["ycoords"]
        diffraction = {
            key: _canonical_diffraction(data[key], len(xcoords), str(source))
            for key in _DIFFRACTION_KEYS
            if key in data
        }
        if not diffraction:
            raise ValueError(
                f"{source}: missing diffraction data; expected 'diff3d' or 'diffraction'."
            )
        if len(diffraction) == 2 and not np.array_equal(
            diffraction["diff3d"], diffraction["diffraction"]
        ):
            raise ValueError(
                f"{source}: canonical 'diff3d' and compatibility alias "
                "'diffraction' contain conflicting diffraction stacks."
            )
        diff3d = diffraction[next(key for key in _DIFFRACTION_KEYS if key in diffraction)]
        _validate_raw_shapes(data, diff3d, str(source))
        truth = _truth_patches(data, diff3d.shape, str(source), truth_policy)
        n_diffraction = len(diff3d)
        n_coordinates = len(xcoords)
        xcoords, ycoords = _align_coordinates(
            xcoords,
            ycoords,
            n_diffraction,
            str(source),
            coordinate_policy,
        )

        xcoords_start = _finite_numeric_vector(
            _aligned_vector(
                data["xcoords_start"] if "xcoords_start" in data else xcoords,
                name="xcoords_start",
                n_diffraction=n_diffraction,
                n_coordinates=n_coordinates,
                source=str(source),
            ),
            name="xcoords_start",
            source=str(source),
        )
        ycoords_start = _finite_numeric_vector(
            _aligned_vector(
                data["ycoords_start"] if "ycoords_start" in data else ycoords,
                name="ycoords_start",
                n_diffraction=n_diffraction,
                n_coordinates=n_coordinates,
                source=str(source),
            ),
            name="ycoords_start",
            source=str(source),
        )
        stored_experiment_id = _experiment_id(
            data["experiment_id"] if "experiment_id" in data else None, str(source)
        )
        assigned_experiment_id = _experiment_id(experiment_id, str(source))
        if assigned_experiment_id is None:
            assigned_experiment_id = stored_experiment_id
        elif (
            stored_experiment_id is not None
            and stored_experiment_id != assigned_experiment_id
        ):
            raise ValueError(
                f"{source}: stored experiment_id {stored_experiment_id!r} conflicts "
                f"with assigned experiment_id {assigned_experiment_id!r}."
            )
        scale_contract_version = _optional_scalar(data, "scale_contract_version")
        measurement_domain = _optional_scalar(data, "measurement_domain")
        if (scale_contract_version, measurement_domain) not in _MEASUREMENT_PAIRS:
            raise ValueError(
                f"{source}: unsupported scale_contract_version/measurement_domain "
                f"pair ({scale_contract_version!r}, {measurement_domain!r})."
            )
        return AcquisitionRecord(
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords_start,
            ycoords_start=ycoords_start,
            diff3d=diff3d,
            probeGuess=data["probeGuess"],
            scan_index=_identity_vector(
                data["scan_index"] if "scan_index" in data else None,
                name="scan_index",
                n_diffraction=n_diffraction,
                n_coordinates=n_coordinates,
                source=str(source),
            ),
            object_index=_identity_vector(
                data["object_index"] if "object_index" in data else None,
                name="object_index",
                n_diffraction=n_diffraction,
                n_coordinates=n_coordinates,
                source=str(source),
            ),
            objectGuess=data["objectGuess"] if "objectGuess" in data else None,
            Y=truth,
            probe_simulated=(
                data["probe_simulated"] if "probe_simulated" in data else None
            ),
            object_amplitude_scale=_object_amplitude_scale(data, str(source)),
            label=data["label"] if "label" in data else None,
            scale_contract_version=scale_contract_version,
            measurement_domain=measurement_domain,
            experiment_id=assigned_experiment_id,
            metadata=_metadata(data, str(source)),
        )


def transform_coordinates(
    record: AcquisitionRecord,
    *,
    flip_x: bool = False,
    flip_y: bool = False,
    swap_xy: bool = False,
    scale: float = 1.0,
) -> AcquisitionRecord:
    """Return a record with the legacy coordinate transforms applied purely."""

    xcoords = -record.xcoords if flip_x else record.xcoords
    ycoords = -record.ycoords if flip_y else record.ycoords
    xcoords_start = (
        -record.xcoords_start
        if flip_x and record.xcoords_start is not None
        else record.xcoords_start
    )
    ycoords_start = (
        -record.ycoords_start
        if flip_y and record.ycoords_start is not None
        else record.ycoords_start
    )
    if swap_xy:
        xcoords, ycoords = ycoords, xcoords
        xcoords_start, ycoords_start = ycoords_start, xcoords_start
    return replace(
        record,
        xcoords=np.asarray(xcoords) * scale,
        ycoords=np.asarray(ycoords) * scale,
        xcoords_start=(
            None if xcoords_start is None else np.asarray(xcoords_start) * scale
        ),
        ycoords_start=(
            None if ycoords_start is None else np.asarray(ycoords_start) * scale
        ),
    )


def select_acquisition(
    source: int | AcquisitionRecord,
    count: Optional[int] = None,
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> AcquisitionSelection:
    """Choose sorted source rows without touching NumPy's ambient RNG."""

    if seed is not None and rng is not None:
        raise ValueError("seed and rng are mutually exclusive")
    size = source if isinstance(source, int) else len(source.xcoords)
    if size < 0 or count is not None and count < 0:
        raise ValueError("source size and count must be nonnegative")
    if count is None or count >= size:
        return _selection_result(np.arange(size, dtype=np.int64), seed, "all")
    local_rng = rng if rng is not None else np.random.default_rng(seed)
    indices = np.sort(local_rng.choice(size, size=count, replace=False)).astype(
        np.int64, copy=False
    )
    return _selection_result(indices, seed, "random_without_replacement")

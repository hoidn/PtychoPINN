"""Backend-neutral index plans for the maintained acquisition groupers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from ptycho.config.config import TrainingConfig
    from ptycho.raw_data import RawData


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class GroupingPlan:
    """Immutable source-row plan shared by RAM and mmap materializers."""

    neighbor_indices: np.ndarray
    center_indices: np.ndarray
    source_indices: np.ndarray
    object_index: np.ndarray
    experiment_id: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "neighbor_indices": _readonly(self.neighbor_indices, np.int64),
            "center_indices": _readonly(self.center_indices, np.int64),
            "source_indices": _readonly(self.source_indices, np.int64),
            "object_index": _readonly(self.object_index, np.int64),
            "experiment_id": _readonly(self.experiment_id, np.int64),
        }
        neighbors = arrays["neighbor_indices"]
        if neighbors.ndim != 2:
            raise ValueError("neighbor_indices must have shape (rows, group_size)")
        rows = len(neighbors)
        for name in ("center_indices", "object_index", "experiment_id"):
            if arrays[name].shape != (rows,):
                raise ValueError(f"{name} must have shape ({rows},)")
        if arrays["source_indices"].ndim != 1:
            raise ValueError("source_indices must be one-dimensional")
        source_count = len(arrays["source_indices"])
        for name in ("neighbor_indices", "center_indices"):
            if np.any(arrays[name] < 0) or np.any(arrays[name] >= source_count):
                raise ValueError(f"{name} contains an out-of-range source row")
        if np.any(arrays["source_indices"] < 0):
            raise ValueError("source_indices must be nonnegative")
        if len(np.unique(arrays["source_indices"])) != source_count:
            raise ValueError("source_indices must be distinct")
        for name in ("object_index", "experiment_id"):
            if np.any(arrays[name] < 0):
                raise ValueError(f"{name} must be nonnegative")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)


def _coordinates(
    xcoords: np.ndarray, ycoords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    xcoords = np.asarray(xcoords)
    ycoords = np.asarray(ycoords)
    if xcoords.ndim != 1 or ycoords.ndim != 1 or xcoords.shape != ycoords.shape:
        raise ValueError("xcoords and ycoords must be equal-length vectors")
    if not np.issubdtype(xcoords.dtype, np.number) or not np.issubdtype(
        ycoords.dtype, np.number
    ):
        raise ValueError("xcoords and ycoords must be numeric")
    if not np.all(np.isfinite(xcoords)) or not np.all(np.isfinite(ycoords)):
        raise ValueError("xcoords and ycoords must be finite")
    return xcoords, ycoords


def _identity(value: object, length: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(length, dtype=np.int64)
    array = np.asarray(value)
    if array.shape == () and name == "experiment_id":
        array = np.full(length, array.item())
    if (
        array.shape != (length,)
        or np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(f"{name} must be a nonnegative integer scalar/vector")
    canonical = np.asarray(array, dtype=np.int64)
    if np.any(canonical < 0):
        raise ValueError(f"{name} must be a nonnegative integer scalar/vector")
    return canonical


def _source_indices(value: object, length: int) -> np.ndarray:
    if value is None:
        return np.arange(length, dtype=np.int64)
    array = np.asarray(value)
    if (
        array.shape != (length,)
        or np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(f"source_indices must be a ({length},) integer vector")
    canonical = np.asarray(array, dtype=np.int64)
    if np.any(canonical < 0) or len(np.unique(canonical)) != length:
        raise ValueError("source_indices must contain distinct nonnegative rows")
    return canonical


def _rng(
    seed: Optional[int],
    rng: Optional[np.random.Generator],
    *,
    default_seed: Optional[int],
) -> np.random.Generator:
    if seed is not None and rng is not None:
        raise ValueError("seed and rng are mutually exclusive")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        return rng
    return np.random.default_rng(default_seed if seed is None else seed)


def _plan(
    neighbors: np.ndarray,
    centers: np.ndarray,
    objects: np.ndarray,
    experiments: np.ndarray,
    source_indices: np.ndarray,
    group_size: int,
) -> GroupingPlan:
    neighbors = np.asarray(neighbors, dtype=np.int64).reshape(-1, group_size)
    centers = np.asarray(centers, dtype=np.int64).reshape(-1)
    if len(neighbors) and np.any(objects[neighbors] != objects[centers, None]):
        raise ValueError("neighbor rows must remain within one object_index partition")
    return GroupingPlan(
        neighbor_indices=neighbors,
        center_indices=centers,
        source_indices=source_indices,
        object_index=objects[centers],
        experiment_id=experiments[centers],
    )


def group_from_config(
    raw_data: RawData,
    config: TrainingConfig,
    *,
    dataset_path: Optional[str] = None,
) -> dict:
    """Decide the shared grouping semantics once and delegate to ``RawData``.

    Both backend mirrors route through here so seed and count semantics live
    in exactly one place.  Requests the shared path cannot honor (e.g. more
    groups than candidate rows) surface as ``ValueError`` from
    ``generate_grouped_data``.
    """
    return raw_data.generate_grouped_data(
        N=config.model.N,
        K=config.neighbor_count,
        nsamples=config.training_groups,
        dataset_path=dataset_path,
        seed=config.subsample_seed,
        sequential_sampling=config.sequential_sampling,
        gridsize=config.model.gridsize,
    )


CENTERED_NEAREST_GROUPING_CONTRACT = "centered-nearest-v1"


def _index_vector(value: object, name: str, *, size: int) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(f"{name} must be a one-dimensional integer vector")
    canonical = np.asarray(array, dtype=np.int64)
    if np.any(canonical < 0) or np.any(canonical >= size):
        raise ValueError(f"{name} contains an out-of-range source row")
    if len(np.unique(canonical)) != len(canonical):
        raise ValueError(f"{name} must contain distinct rows")
    return canonical


def plan_nearest_groups(
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    *,
    center_indices: np.ndarray,
    candidate_indices: np.ndarray,
    group_size: int,
    neighbor_count: int,
    repeats: int = 1,
    object_index: Optional[np.ndarray] = None,
    experiment_id: object = None,
    source_indices: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> GroupingPlan:
    """Plan centered nearest groups over an explicit center and candidate pool.

    Every emitted row has C distinct same-object members and its designated
    center in column zero; the remaining members come from the K nearest
    non-center candidates of the center's object partition.
    """
    xcoords, ycoords = _coordinates(xcoords, ycoords)
    size = len(xcoords)
    if group_size < 1:
        raise ValueError(f"C={group_size} must be at least 1")
    if repeats < 1:
        raise ValueError(f"repeats={repeats} must be at least 1")
    if neighbor_count < group_size - 1:
        raise ValueError(
            f"K={neighbor_count} must be at least C-1={group_size - 1}"
        )
    centers = _index_vector(center_indices, "center_indices", size=size)
    candidates = _index_vector(candidate_indices, "candidate_indices", size=size)
    missing = np.setdiff1d(centers, candidates)
    if len(missing):
        raise ValueError(
            "center_indices must be a subset of candidate_indices; "
            f"center row(s) {missing.tolist()} are outside the pool"
        )
    objects = _identity(object_index, size, "object_index")
    experiments = _identity(experiment_id, size, "experiment_id")
    sources = _source_indices(source_indices, size)
    local_rng = _rng(seed, rng, default_seed=None)

    if group_size == 1:
        ordered = np.repeat(centers, repeats)
        return _plan(
            ordered.reshape(-1, 1),
            ordered,
            objects,
            experiments,
            sources,
            group_size,
        )

    coords = np.column_stack((xcoords, ycoords))
    trees: dict[int, tuple[np.ndarray, cKDTree]] = {}
    rows: list[np.ndarray] = []
    row_centers: list[int] = []
    for center in centers:
        key = int(objects[center])
        if key not in trees:
            partition = candidate_indices[objects[candidate_indices] == key]
            if len(partition) - 1 < group_size - 1:
                raise ValueError(
                    f"object partition {key} for center {center} has only "
                    f"{len(partition) - 1} non-center candidates but "
                    f"C-1={group_size - 1} required"
                )
            trees[key] = partition, cKDTree(coords[partition])
        partition, tree = trees[key]
        _, local = tree.query(
            coords[center], k=min(neighbor_count + 1, len(partition))
        )
        pool = partition[np.atleast_1d(local)]
        pool = pool[pool != center][:neighbor_count]
        for _ in range(repeats):
            neighbors = local_rng.choice(pool, size=group_size - 1, replace=False)
            rows.append(np.concatenate(([center], neighbors)))
            row_centers.append(int(center))
    neighbors = np.asarray(rows, dtype=np.int64).reshape(-1, group_size)
    return _plan(
        neighbors,
        np.asarray(row_centers, dtype=np.int64),
        objects,
        experiments,
        sources,
        group_size,
    )

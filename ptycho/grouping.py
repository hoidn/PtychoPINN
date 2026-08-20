"""Backend-neutral index plans for the maintained acquisition groupers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Optional

import numpy as np
from scipy.spatial import KDTree, cKDTree


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class GroupingPlan:
    """Immutable source-row plan shared by RAM and mmap materializers."""

    neighbor_indices: np.ndarray
    center_indices: np.ndarray
    center_available: np.ndarray
    eligible_indices: np.ndarray
    source_indices: np.ndarray
    object_index: np.ndarray
    experiment_id: np.ndarray
    policy: str
    coverage_complete: bool

    def __post_init__(self) -> None:
        arrays = {
            "neighbor_indices": _readonly(self.neighbor_indices, np.int64),
            "center_indices": _readonly(self.center_indices, np.int64),
            "center_available": _readonly(self.center_available, np.bool_),
            "eligible_indices": _readonly(self.eligible_indices, np.int64),
            "source_indices": _readonly(self.source_indices, np.int64),
            "object_index": _readonly(self.object_index, np.int64),
            "experiment_id": _readonly(self.experiment_id, np.int64),
        }
        neighbors = arrays["neighbor_indices"]
        if neighbors.ndim != 2:
            raise ValueError("neighbor_indices must have shape (rows, group_size)")
        rows = len(neighbors)
        for name in ("center_indices", "center_available", "object_index", "experiment_id"):
            if arrays[name].shape != (rows,):
                raise ValueError(f"{name} must have shape ({rows},)")
        for name in ("eligible_indices", "source_indices"):
            if arrays[name].ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
        centers = arrays["center_indices"]
        available = arrays["center_available"]
        if np.any(available & (centers < 0)) or np.any(~available & (centers != -1)):
            raise ValueError("center identity and availability disagree")
        source_count = len(arrays["source_indices"])
        for name in ("neighbor_indices", "eligible_indices"):
            if np.any(arrays[name] < 0) or np.any(arrays[name] >= source_count):
                raise ValueError(f"{name} contains an out-of-range source row")
        if np.any(available & (centers >= source_count)):
            raise ValueError("center_indices contains an out-of-range source row")
        if np.any(arrays["source_indices"] < 0):
            raise ValueError("source_indices must be nonnegative")
        if len(np.unique(arrays["source_indices"])) != source_count:
            raise ValueError("source_indices must be distinct")
        for name in ("object_index", "experiment_id"):
            if np.any(arrays[name] < 0):
                raise ValueError(f"{name} must be nonnegative")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "coverage_complete", bool(self.coverage_complete))


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


def _eligible(value: np.ndarray, length: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_:
        if array.shape != (length,):
            raise ValueError(f"eligible mask must have shape ({length},)")
        return np.flatnonzero(array).astype(np.int64, copy=False)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("eligible_indices must be an integer vector or boolean mask")
    canonical = np.asarray(array, dtype=np.int64)
    if np.any(canonical < 0) or np.any(canonical >= length):
        raise ValueError("eligible_indices contains an out-of-range source row")
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
    eligible: np.ndarray,
    objects: np.ndarray,
    experiments: np.ndarray,
    source_indices: np.ndarray,
    policy: str,
    group_size: int,
) -> GroupingPlan:
    neighbors = np.asarray(neighbors, dtype=np.int64).reshape(-1, group_size)
    centers = np.asarray(centers, dtype=np.int64).reshape(-1)
    covered = np.all(np.isin(eligible, neighbors.reshape(-1)))
    if len(neighbors) and np.any(objects[neighbors] != objects[centers, None]):
        raise ValueError("neighbor rows must remain within one object_index partition")
    return GroupingPlan(
        neighbor_indices=neighbors,
        center_indices=centers,
        center_available=np.ones(len(centers), dtype=np.bool_),
        eligible_indices=eligible,
        source_indices=source_indices,
        object_index=objects[centers],
        experiment_id=experiments[centers],
        policy=policy,
        coverage_complete=bool(covered),
    )


def _sample_then_group(
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    objects: np.ndarray,
    count: int,
    neighbor_count: int,
    group_size: int,
    local_rng: np.random.Generator,
    seed_indices: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    size = len(xcoords)
    if size < group_size:
        raise ValueError(
            f"Dataset has only {size} points but {group_size} coordinates per group requested."
        )
    if neighbor_count < group_size:
        raise ValueError(
            f"K={neighbor_count} must be >= C={group_size} "
            "(need at least C neighbors to form a group)"
        )
    if seed_indices is not None:
        local_rng = np.random.default_rng(0)
        centers = np.asarray(seed_indices[: min(len(seed_indices), size)], dtype=np.int64)
    else:
        actual = min(count, size)
        centers = (
            local_rng.choice(np.arange(size), size=actual, replace=False)
            if actual < size
            else np.arange(size)
        )
    if group_size == 1:
        return centers.reshape(-1, 1), centers

    coords = np.column_stack((xcoords, ycoords))
    groups = np.empty((len(centers), group_size), dtype=np.int64)
    trees: dict[int, tuple[np.ndarray, cKDTree]] = {}
    for row, center in enumerate(centers):
        key = int(objects[center])
        if key not in trees:
            partition = np.flatnonzero(objects == key)
            if len(partition) < group_size:
                raise ValueError(
                    "object_index partition has only "
                    f"{len(partition)} points but {group_size} coordinates per group "
                    "were requested"
                )
            trees[key] = partition, cKDTree(coords[partition])
        partition, tree = trees[key]
        _, local_neighbors = tree.query(
            coords[center], k=min(neighbor_count + 1, len(partition))
        )
        neighbors = partition[np.atleast_1d(local_neighbors)]
        neighbors = (
            neighbors[1 : neighbor_count + 1]
            if len(neighbors) > neighbor_count
            else neighbors[:neighbor_count]
        )
        available = (
            np.concatenate(([center], neighbors))
            if len(neighbors) < group_size
            else neighbors
        )
        groups[row] = local_rng.choice(
            available, size=group_size, replace=len(available) < group_size
        )
    return groups, centers


def _oversample(
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    count: int,
    neighbor_count: int,
    group_size: int,
    local_rng: np.random.Generator,
    seed_indices: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    size = len(xcoords)
    if size < group_size:
        raise ValueError(
            f"Dataset has only {size} points but {group_size} coordinates per group requested."
        )
    if neighbor_count < group_size:
        raise ValueError(
            f"K={neighbor_count} must be >= C={group_size} "
            "(need at least C neighbors to form a group)"
        )
    combinations_per_seed = comb(neighbor_count, group_size)
    minimum_seeds = max(1, (count + combinations_per_seed - 1) // combinations_per_seed)
    seed_count = min(minimum_seeds * 2, size)
    if seed_indices is not None and len(seed_indices) >= minimum_seeds:
        seeds = np.asarray(seed_indices[:seed_count], dtype=np.int64)
    elif seed_count < size:
        seeds = local_rng.choice(np.arange(size), size=seed_count, replace=False)
    else:
        seeds = np.arange(size)

    coords = np.column_stack((xcoords, ycoords))
    tree = cKDTree(coords)
    pool: list[np.ndarray] = []
    pool_centers: list[int] = []
    for center in seeds:
        _, neighbor_indices = tree.query(
            coords[center : center + 1], k=min(neighbor_count + 1, size)
        )
        neighbors = neighbor_indices[0]
        neighbors = (
            neighbors[1 : neighbor_count + 1]
            if len(neighbors) > neighbor_count
            else neighbors[1:] if len(neighbors) > 1 else neighbors
        )
        available = (
            np.concatenate(([center], neighbors))
            if len(neighbors) < group_size
            else neighbors[:neighbor_count]
        )
        if len(available) >= group_size:
            for group in combinations(available, group_size):
                pool.append(np.asarray(group, dtype=np.int64))
                pool_centers.append(int(center))
                if len(pool) >= count * 2:
                    break
        if len(pool) >= count * 2:
            break
    if not pool:
        raise ValueError("No valid combinations could be generated")
    selected = local_rng.choice(
        len(pool), size=count, replace=count > len(pool)
    )
    return np.asarray(pool)[selected], np.asarray(pool_centers, dtype=np.int64)[selected]


def group_from_config(
    raw_data: RawData,
    config: TrainingConfig,
    *,
    dataset_path: Optional[str] = None,
) -> dict:
    """Decide the shared grouping semantics once and delegate to ``RawData``.

    Both backend mirrors route through here so seed, oversampling, pool size,
    and count semantics live in exactly one place.  Requests the shared path
    cannot honor (e.g. K choose C oversampling required but not enabled)
    surface as ``ValueError`` from ``generate_grouped_data``.
    """
    sampling = config.sampling
    return raw_data.generate_grouped_data(
        N=config.model.N,
        K=sampling.neighbor_count,
        nsamples=sampling.n_groups,
        dataset_path=dataset_path,
        seed=sampling.subsample_seed,
        sequential_sampling=sampling.sequential_sampling,
        gridsize=config.model.gridsize,
        enable_oversampling=sampling.enable_oversampling,
        neighbor_pool_size=sampling.neighbor_pool_size,
    )


def plan_sample_then_group(
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    *,
    object_index: Optional[np.ndarray] = None,
    experiment_id: object = None,
    source_indices: Optional[np.ndarray] = None,
    count: int,
    neighbor_count: int,
    group_size: int,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    sequential: bool = False,
    enable_oversampling: bool = False,
    neighbor_pool_size: Optional[int] = None,
) -> GroupingPlan:
    """Plan RawData's random/sequential sample-then-group policy family."""

    xcoords, ycoords = _coordinates(xcoords, ycoords)
    size = len(xcoords)
    objects = _identity(object_index, size, "object_index")
    experiments = _identity(experiment_id, size, "experiment_id")
    sources = _source_indices(source_indices, size)
    if count < 0 or neighbor_count < 1 or group_size < 1:
        raise ValueError("count must be nonnegative and neighbor/group sizes positive")
    local_rng = _rng(seed, rng, default_seed=None)
    seed_indices = np.arange(min(count, size)) if sequential else None
    oversampling = count > size and group_size > 1
    effective_neighbors = (
        neighbor_count if neighbor_pool_size is None else neighbor_pool_size
    )
    if oversampling:
        if np.unique(objects).size > 1:
            raise ValueError(
                "K choose C oversampling is not defined across independent "
                "object_index partitions"
            )
        if not enable_oversampling:
            raise ValueError(
                "K choose C oversampling is required but not enabled; "
                "set enable_oversampling=True"
            )
        if effective_neighbors < group_size:
            raise ValueError("K choose C oversampling requires neighbor_pool_size >= C")
        neighbors, centers = _oversample(
            xcoords,
            ycoords,
            count,
            effective_neighbors,
            group_size,
            local_rng,
            seed_indices,
        )
        policy = "raw_k_choose_c_oversampling"
    else:
        if enable_oversampling and (
            group_size <= 1 or effective_neighbors < group_size
        ):
            raise ValueError(
                "enable_oversampling requires group_size>1 and "
                f"neighbor_pool_size>=group_size; got group_size={group_size!r}, "
                f"neighbor_pool_size={neighbor_pool_size!r}"
            )
        neighbors, centers = _sample_then_group(
            xcoords,
            ycoords,
            objects,
            count,
            neighbor_count,
            group_size,
            local_rng,
            seed_indices,
        )
        policy = (
            "raw_sequential_sample_then_group"
            if sequential
            else "raw_random_sample_then_group"
        )
    return _plan(
        neighbors,
        centers,
        np.arange(size, dtype=np.int64),
        objects,
        experiments,
        sources,
        policy,
        group_size,
    )


def _nearest_candidates(
    coords: np.ndarray,
    eligible: np.ndarray,
    objects: np.ndarray,
    neighbor_count: int,
) -> list[np.ndarray]:
    rows: list[Optional[np.ndarray]] = [None] * len(eligible)
    eligible_objects = objects[eligible]
    for key in dict.fromkeys(int(value) for value in eligible_objects):
        positions = np.flatnonzero(eligible_objects == key)
        source_rows = eligible[positions]
        tree = cKDTree(coords[source_rows])
        _, local = tree.query(
            coords[source_rows], k=min(neighbor_count + 1, len(source_rows))
        )
        local = np.asarray(local)
        if local.ndim == 1:
            local = local[:, None]
        for position, candidates in zip(positions, source_rows[local], strict=True):
            rows[int(position)] = np.asarray(candidates, dtype=np.int64)
    return [row for row in rows if row is not None]


def _min_distance_candidates(
    points: np.ndarray,
    neighbor_count: int,
    minimum_distance: float,
    maximum_distance: float,
) -> np.ndarray:
    tree = KDTree(points)
    sparse = tree.sparse_distance_matrix(tree, max_distance=maximum_distance)
    candidates: list[list[tuple[float, int]]] = [[] for _ in points]
    for (left, right), distance in sparse.items():
        if distance >= minimum_distance:
            candidates[left].append((distance, right))
            candidates[right].append((distance, left))
    result = np.full((len(points), neighbor_count), -1, dtype=np.int64)
    for row, values in enumerate(candidates):
        values.sort(key=lambda item: item[0])
        selected = [index for _, index in values[:neighbor_count]]
        result[row, : len(selected)] = selected
    return result


def _bounded_candidates(
    coords: np.ndarray,
    eligible: np.ndarray,
    objects: np.ndarray,
    neighbor_count: int,
    minimum_distance: float,
    maximum_distance: float,
) -> list[np.ndarray]:
    rows: list[Optional[np.ndarray]] = [None] * len(eligible)
    eligible_objects = objects[eligible]
    for key in dict.fromkeys(int(value) for value in eligible_objects):
        positions = np.flatnonzero(eligible_objects == key)
        source_rows = eligible[positions]
        local = _min_distance_candidates(
            coords[source_rows],
            neighbor_count,
            minimum_distance,
            maximum_distance,
        )
        mapped = np.full_like(local, -1)
        valid = local >= 0
        mapped[valid] = source_rows[local[valid]]
        for position, candidates in zip(positions, mapped, strict=True):
            rows[int(position)] = candidates
    return [row for row in rows if row is not None]


def _repair_coverage(groups: np.ndarray, centers: np.ndarray) -> np.ndarray:
    repaired = np.array(groups, dtype=np.int64, copy=True)
    if repaired.ndim != 2 or len(repaired) != len(centers):
        raise ValueError("group rows must align with paired center identities")
    if np.any(repaired < 0):
        raise ValueError("Nearest grouping produced an invalid scan index")
    if any(len(set(row.tolist())) != repaired.shape[1] for row in repaired):
        raise ValueError("Nearest grouping rows must contain distinct scan ids")
    values, frequencies = np.unique(repaired, return_counts=True)
    counts = dict(zip(values.tolist(), frequencies.tolist(), strict=True))
    for missing in (
        item for item in sorted(set(centers.tolist())) if counts.get(item, 0) == 0
    ):
        replacement = None
        for row in np.flatnonzero(centers == missing):
            for column, participant in enumerate(repaired[row]):
                if counts.get(int(participant), 0) > 1:
                    replacement = int(row), column, int(participant)
                    break
            if replacement is not None:
                break
        if replacement is None:
            raise ValueError(
                "cannot repair complete scan coverage without dropping an existing "
                f"participant: {missing}"
            )
        row, column, participant = replacement
        repaired[row, column] = missing
        counts[participant] -= 1
        counts[missing] = 1
    return repaired


def _quadrant_groups(
    coords: np.ndarray,
    eligible: np.ndarray,
    objects: np.ndarray,
    repeats: int,
    quadrant_neighbor_count: int,
    minimum_distance: float,
    maximum_distance: float,
    scan_pattern: str,
    local_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if quadrant_neighbor_count <= 4:
        raise ValueError("quadrant_neighbor_count must be greater than 4")
    if scan_pattern == "Isotropic":
        x_lower, x_upper, y_lower, y_upper, y_bound = 0, 10, 0, 10, 0
    elif scan_pattern == "Rectangular":
        x_lower, x_upper, y_lower, y_upper, y_bound = 0, 12, 0.7, 2, 2
    else:
        raise ValueError("scan_pattern must be 'Isotropic' or 'Rectangular'")

    trees: dict[int, tuple[np.ndarray, cKDTree, dict[int, int]]] = {}
    groups: list[list[int]] = []
    centers: list[int] = []
    for center in eligible:
        key = int(objects[center])
        if key not in trees:
            partition = np.flatnonzero(objects == key)
            trees[key] = (
                partition,
                cKDTree(coords[partition]),
                {int(source): local for local, source in enumerate(partition)},
            )
        partition, tree, local_indices = trees[key]
        distances, neighbors = tree.query(
            coords[center], k=quadrant_neighbor_count + 1
        )
        valid: list[int] = []
        for local, distance in zip(
            np.atleast_1d(neighbors), np.atleast_1d(distances), strict=True
        ):
            if local >= len(partition) or int(local) == local_indices[int(center)]:
                continue
            if minimum_distance <= distance < maximum_distance:
                valid.append(int(partition[int(local)]))
        if not valid:
            continue
        candidates = {name: [] for name in ("TL", "TR", "BL", "BR")}
        deltas = coords[valid] - coords[center]
        for neighbor, (dx, dy) in zip(valid, deltas, strict=True):
            if -x_upper < dx < x_lower and -y_lower < dy < y_upper:
                candidates["TL"].append(neighbor)
            elif x_lower < dx < x_upper and -y_lower < dy < y_upper:
                candidates["TR"].append(neighbor)
            elif -x_upper < dx < x_lower and dy < -y_bound:
                candidates["BL"].append(neighbor)
            elif x_lower < dx < x_upper and dy < -y_bound:
                candidates["BR"].append(neighbor)
        for values in candidates.values():
            values.append(int(center))
        if sum(len(values) == 1 for values in candidates.values()) >= 2:
            continue
        for _ in range(repeats):
            while True:
                group = [
                    int(local_rng.choice(candidates[name]))
                    for name in ("TL", "TR", "BL", "BR")
                ]
                if group.count(int(center)) <= 1:
                    break
            groups.append(group)
            centers.append(int(center))
    return np.asarray(groups, dtype=np.int64).reshape(-1, 4), np.asarray(
        centers, dtype=np.int64
    )


def plan_scan_centered(
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    *,
    eligible_indices: np.ndarray,
    object_index: Optional[np.ndarray] = None,
    experiment_id: object = None,
    source_indices: Optional[np.ndarray] = None,
    policy: str,
    group_size: int,
    neighbor_count: int,
    repeats: int = 1,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    min_neighbor_distance: float = 0.0,
    max_neighbor_distance: float = 3.0,
    quadrant_neighbor_count: int = 30,
    scan_pattern: str = "Isotropic",
    ensure_complete_coverage: bool = False,
) -> GroupingPlan:
    """Plan the maintained Nearest/Min_dist/4_quadrant scan-centered family."""

    if policy not in {"Nearest", "Min_dist", "4_quadrant"}:
        raise ValueError("policy must be 'Nearest', 'Min_dist', or '4_quadrant'")
    if group_size < 1 or neighbor_count < 1 or repeats < 0:
        raise ValueError("neighbor/group sizes must be positive and repeats nonnegative")
    if not isinstance(ensure_complete_coverage, bool):
        raise TypeError("ensure_complete_coverage must be a bool")
    xcoords, ycoords = _coordinates(xcoords, ycoords)
    size = len(xcoords)
    eligible = _eligible(eligible_indices, size)
    objects = _identity(object_index, size, "object_index")
    experiments = _identity(experiment_id, size, "experiment_id")
    sources = _source_indices(source_indices, size)
    local_rng = _rng(seed, rng, default_seed=0)
    coords = np.column_stack((xcoords, ycoords))

    if group_size == 1:
        centers = np.repeat(eligible, repeats)
        neighbors = centers.reshape(-1, 1)
    elif policy == "4_quadrant":
        if group_size != 4:
            raise ValueError("4_quadrant grouping requires group_size=4")
        neighbors, centers = _quadrant_groups(
            coords,
            eligible,
            objects,
            repeats,
            quadrant_neighbor_count,
            min_neighbor_distance,
            max_neighbor_distance,
            scan_pattern,
            local_rng,
        )
    else:
        candidates = (
            _nearest_candidates(
                coords, eligible, objects, neighbor_count
            )
            if policy == "Nearest"
            else _bounded_candidates(
                coords,
                eligible,
                objects,
                neighbor_count,
                min_neighbor_distance,
                max_neighbor_distance,
            )
        )
        grouped: list[np.ndarray] = []
        center_rows: list[int] = []
        for center, row in zip(eligible, candidates, strict=True):
            row = row[row >= 0]
            if len(row) < group_size:
                if policy == "Min_dist":
                    continue
                raise ValueError(
                    f"object_index partition cannot supply {group_size} candidates "
                    f"for center {int(center)}"
                )
            for _ in range(repeats):
                grouped.append(
                    local_rng.choice(row, size=group_size, replace=False)
                )
                center_rows.append(int(center))
        neighbors = np.asarray(grouped, dtype=np.int64).reshape(-1, group_size)
        centers = np.asarray(center_rows, dtype=np.int64)
        if ensure_complete_coverage:
            if policy != "Nearest":
                raise ValueError("complete coverage repair supports only Nearest grouping")
            neighbors = _repair_coverage(neighbors, centers)

    return _plan(
        neighbors,
        centers,
        eligible,
        objects,
        experiments,
        sources,
        policy,
        group_size,
    )

"""Torch compatibility adapters for backend-neutral grouping plans."""

from typing import Optional

import numpy as np

from ptycho.grouping import _nearest_candidates, plan_scan_centered
from ptycho_torch.config_params import DataConfig


def get_neighbor_indices(xcoords, ycoords, data_config, K=6):
    """Compatibility view of the owner's nearest-candidate rows."""

    del data_config
    points = np.column_stack((xcoords, ycoords))
    eligible = np.arange(len(points), dtype=np.int64)
    return np.asarray(
        _nearest_candidates(
            points,
            eligible,
            np.zeros(len(points), dtype=np.int64),
            K,
        )
    )


def group_coords(
    xcoords_full,
    ycoords_full,
    xcoords_bounded,
    ycoords_bounded,
    neighbor_function,
    valid_mask,
    data_config: DataConfig,
    C: int = None,
    return_center_indices: bool = False,
    rng: Optional[np.random.Generator] = None,
    groups_per_center: int = 1,
    ensure_complete_coverage: bool = False,
    object_index=None,
    experiment_id=None,
):
    """Materialize the legacy Torch tuple from the shared scan-centered plan."""

    del xcoords_bounded, ycoords_bounded, neighbor_function
    group_size = data_config.gridsize * data_config.gridsize if C is None else C
    randomness = (
        {"rng": rng}
        if rng is not None
        else {"seed": data_config.subsample_seed}
    )
    plan = plan_scan_centered(
        xcoords_full,
        ycoords_full,
        eligible_indices=valid_mask,
        object_index=object_index,
        experiment_id=experiment_id,
        policy=data_config.neighbor_function,
        group_size=group_size,
        neighbor_count=data_config.neighbor_count,
        repeats=groups_per_center,
        min_neighbor_distance=data_config.min_neighbor_distance,
        max_neighbor_distance=data_config.max_neighbor_distance,
        quadrant_neighbor_count=data_config.K_quadrant,
        scan_pattern=data_config.scan_pattern,
        ensure_complete_coverage=ensure_complete_coverage,
        **randomness,
    )
    nn_indices = np.array(plan.neighbor_indices, copy=True)
    coords_nn = np.stack(
        [np.asarray(xcoords_full)[nn_indices], np.asarray(ycoords_full)[nn_indices]],
        axis=2,
    )[:, :, None, :]
    if return_center_indices:
        return nn_indices, coords_nn, np.array(plan.center_indices, copy=True)
    return nn_indices, coords_nn


def get_relative_coords(coords_nn, local_offset_sign=-1):
    """Return group centroids and TF-sign relative coordinates."""

    assert np.ndim(coords_nn) == 4
    coords_offsets = np.mean(coords_nn, axis=1)[:, None, :, :]
    return coords_offsets, local_offset_sign * (coords_nn - coords_offsets)

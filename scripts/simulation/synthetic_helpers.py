from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from ptycho import params
from ptycho.config.config import ModelConfig, TrainingConfig, update_legacy_dict
from ptycho.diffsim import sim_object_image
from ptycho.nongrid_simulation import generate_simulated_data
from ptycho.probe import get_default_probe
from ptycho.raw_data import RawData


def make_lines_object(
    object_size: int,
    *,
    data_source: str = "lines",
    seed: int | None = None,
) -> np.ndarray:
    previous_source = params.get("data_source")
    if seed is not None:
        np.random.seed(seed)
    try:
        params.set("data_source", data_source)
        obj = sim_object_image(size=object_size, which="train")
    finally:
        params.set("data_source", previous_source)

    obj = np.squeeze(obj)
    if obj.ndim > 2:
        obj = obj[..., 0]
    return obj.astype(np.complex64)


def make_probe(
    N: int,
    *,
    mode: str = "idealized",
    path: Path | None = None,
    scale: float = 0.7,
) -> np.ndarray:
    if mode == "idealized":
        if params.cfg.get("default_probe_scale") is None:
            params.cfg["default_probe_scale"] = scale
        probe_guess = get_default_probe(N=N, fmt="np")
    elif mode == "custom":
        if path is None:
            raise ValueError("path is required for custom probe mode")
        if not path.exists():
            raise FileNotFoundError(f"Custom probe not found: {path}")
        with np.load(path) as data:
            if "probeGuess" not in data:
                raise KeyError(f"probeGuess missing in {path}")
            probe_guess = data["probeGuess"]
        if probe_guess.shape != (N, N):
            raise ValueError(f"Expected probe shape {(N, N)}, got {probe_guess.shape}")
    else:
        raise ValueError(f"Unknown probe mode: {mode}")

    return probe_guess.astype(np.complex64)


def simulate_nongrid_raw_data(
    object_guess: np.ndarray,
    probe_guess: np.ndarray,
    *,
    N: int,
    n_images: int,
    nphotons: float,
    seed: int,
    buffer: float | None = None,
    sim_gridsize: int = 1,
) -> RawData:
    if sim_gridsize != 1:
        raise ValueError("simulate_nongrid_raw_data only supports sim_gridsize=1")

    sim_config = TrainingConfig(
        model=ModelConfig(N=N, gridsize=1),
        n_groups=n_images,
        nphotons=nphotons,
    )
    update_legacy_dict(params.cfg, sim_config)

    if buffer is None:
        buffer = float(min(object_guess.shape)) * 0.35

    np.random.seed(seed)
    raw_data = generate_simulated_data(
        config=sim_config,
        objectGuess=object_guess,
        probeGuess=probe_guess,
        buffer=buffer,
        return_patches=False,
    )
    return raw_data


def split_raw_data_by_axis(
    raw_data: RawData,
    *,
    split_fraction: float = 0.5,
    axis: str = "y",
) -> Tuple[RawData, RawData]:
    if raw_data.diff3d is None:
        raise ValueError("raw_data.diff3d is required for splitting")
    if not 0 < split_fraction < 1:
        raise ValueError("split_fraction must be in (0, 1)")

    n_images = raw_data.diff3d.shape[0]
    test_count = int(round(n_images * split_fraction))
    if test_count <= 0 or test_count >= n_images:
        raise ValueError("split_fraction produces empty train/test split")

    if axis == "x":
        coords = raw_data.xcoords
    elif axis == "y":
        coords = raw_data.ycoords
    else:
        raise ValueError("axis must be 'x' or 'y'")

    sort_idx = np.argsort(coords)
    train_idx = sort_idx[: n_images - test_count]
    test_idx = sort_idx[n_images - test_count :]
    return _slice_raw_data(raw_data, train_idx), _slice_raw_data(raw_data, test_idx)


def _slice_raw_data(raw_data: RawData, indices: np.ndarray) -> RawData:
    scan_index = None
    if raw_data.scan_index is not None:
        scan_index = raw_data.scan_index[indices]
    Y = None
    if raw_data.Y is not None:
        Y = raw_data.Y[indices]

    xcoords_start = None
    if raw_data.xcoords_start is not None:
        xcoords_start = raw_data.xcoords_start[indices]
    ycoords_start = None
    if raw_data.ycoords_start is not None:
        ycoords_start = raw_data.ycoords_start[indices]

    return RawData(
        xcoords=raw_data.xcoords[indices],
        ycoords=raw_data.ycoords[indices],
        xcoords_start=xcoords_start,
        ycoords_start=ycoords_start,
        diff3d=raw_data.diff3d[indices],
        probeGuess=raw_data.probeGuess,
        scan_index=scan_index,
        objectGuess=raw_data.objectGuess,
        Y=Y,
        norm_Y_I=raw_data.norm_Y_I,
        metadata=raw_data.metadata,
    )

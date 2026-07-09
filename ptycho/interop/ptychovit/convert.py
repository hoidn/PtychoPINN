from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .contracts import PtychoViTHdf5Pair


def _extract_positions(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract y/x scan vectors from common coords_nominal layouts."""
    if coords.ndim == 2 and coords.shape[1] >= 2:
        y = coords[:, 0]
        x = coords[:, 1]
        return y, x
    if coords.ndim == 4 and coords.shape[1] >= 2:
        y = coords[:, 0, 0, 0]
        x = coords[:, 1, 0, 0]
        return y, x
    if coords.ndim == 4 and coords.shape[1] == 1 and coords.shape[2] >= 2:
        y = coords[:, 0, 0, 0]
        x = coords[:, 0, 1, 0]
        return y, x
    raise ValueError(f"Unsupported coords_nominal shape: {coords.shape}")


def _select_position_array(data: dict) -> np.ndarray:
    """Select absolute scan positions for interop (never local relative coords)."""
    for key in ("coords_offsets", "coords_start_offsets"):
        if key in data:
            return np.asarray(data[key])
    raise KeyError(
        "NPZ must contain absolute scan positions in coords_offsets (or coords_start_offsets); "
        "local coords_true/coords_nominal are not valid for PtychoViT probe positions"
    )


def _to_probe_dataset(probe_guess: np.ndarray) -> np.ndarray:
    probe = np.asarray(probe_guess)
    if np.iscomplexobj(probe):
        probe = probe.astype(np.complex64)
    if probe.ndim == 2:
        return probe[np.newaxis, np.newaxis, ...].astype(np.complex64)
    if probe.ndim == 3:
        return probe[np.newaxis, ...].astype(np.complex64)
    if probe.ndim == 4:
        return probe.astype(np.complex64)
    raise ValueError(f"Unsupported probeGuess shape: {probe.shape}")


def _to_object_dataset(data: dict) -> np.ndarray:
    if "YY_full" in data:
        obj = np.asarray(data["YY_full"])
    elif "YY_ground_truth" in data:
        obj = np.asarray(data["YY_ground_truth"])
    else:
        raise KeyError("NPZ must contain YY_ground_truth or YY_full")
    obj = np.squeeze(obj.astype(np.complex64))
    if obj.ndim == 2:
        selected = obj
    elif obj.ndim == 3:
        # PtychoViT para contract expects a single object map; choose a stable first sample.
        selected = obj[0]
    else:
        raise ValueError(f"Unsupported object shape for PtychoViT conversion: {obj.shape}")

    if selected.ndim != 2:
        raise ValueError(f"Expected selected object to be 2D, got {selected.shape}")
    return selected[np.newaxis, ...]


def _position_origin(object_shape_hw: tuple[int, int]) -> tuple[float, float]:
    h, w = object_shape_hw
    return (float(np.round(h / 2.0) + 0.5), float(np.round(w / 2.0) + 0.5))


def _overlap_score(
    y_centered: np.ndarray,
    x_centered: np.ndarray,
    *,
    object_shape_hw: tuple[int, int],
    patch_shape_hw: tuple[int, int],
) -> float:
    """Return mean patch overlap fraction after PtychoViT loader origin-shift."""
    h, w = object_shape_hw
    ph, pw = patch_shape_hw
    oy, ox = _position_origin(object_shape_hw)
    cy = y_centered + oy
    cx = x_centered + ox
    half_h = (ph - 1.0) / 2.0
    half_w = (pw - 1.0) / 2.0

    y0 = cy - half_h
    y1 = cy + half_h
    x0 = cx - half_w
    x1 = cx + half_w

    overlap_h = np.maximum(0.0, np.minimum(y1, h - 1.0) - np.maximum(y0, 0.0))
    overlap_w = np.maximum(0.0, np.minimum(x1, w - 1.0) - np.maximum(x0, 0.0))
    overlap = (overlap_h * overlap_w) / float(ph * pw)
    return float(np.mean(overlap))


def _normalize_positions_for_ptychovit(
    y_raw: np.ndarray,
    x_raw: np.ndarray,
    *,
    object_shape_hw: tuple[int, int],
    patch_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Choose centered-vs-absolute interpretation of incoming scan positions.

    PtychoViT loader expects centered coords and later adds object-origin internally.
    Our upstream NPZs may carry either:
    - centered coords already
    - absolute top-left-origin pixel coords
    """
    y_raw = np.asarray(y_raw, dtype=np.float64)
    x_raw = np.asarray(x_raw, dtype=np.float64)
    oy, ox = _position_origin(object_shape_hw)

    # Candidate A: already-centered.
    ya, xa = y_raw, x_raw
    # Candidate B: absolute pixels -> centered by subtracting origin.
    yb, xb = y_raw - oy, x_raw - ox

    score_a = _overlap_score(ya, xa, object_shape_hw=object_shape_hw, patch_shape_hw=patch_shape_hw)
    score_b = _overlap_score(yb, xb, object_shape_hw=object_shape_hw, patch_shape_hw=patch_shape_hw)
    if score_b > score_a:
        return yb, xb
    return ya, xa


def convert_npz_split_to_hdf5_pair(
    npz_path: Path,
    out_dir: Path,
    object_name: str,
    pixel_size_m: float = 1.0,
) -> PtychoViTHdf5Pair:
    """Convert a grid-lines NPZ split into PtychoViT paired HDF5 files."""
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=True) as npz:
        data = {key: npz[key] for key in npz.files}

    if "diffraction" not in data:
        raise KeyError("NPZ must contain diffraction")
    if "probeGuess" not in data:
        raise KeyError("NPZ must contain probeGuess")
    diffraction_amp = np.asarray(data["diffraction"], dtype=np.float32)
    if diffraction_amp.ndim == 4:
        diffraction_amp = diffraction_amp[..., 0]
    if diffraction_amp.ndim != 3:
        raise ValueError(f"Expected diffraction rank 3/4, got {diffraction_amp.shape}")
    dp = np.square(np.clip(diffraction_amp, a_min=0.0, a_max=None)).astype(np.float32)

    object_ds = _to_object_dataset(data)
    probe_ds = _to_probe_dataset(data["probeGuess"])
    y_raw, x_raw = _extract_positions(_select_position_array(data))
    y, x = _normalize_positions_for_ptychovit(
        y_raw,
        x_raw,
        object_shape_hw=(int(object_ds.shape[1]), int(object_ds.shape[2])),
        patch_shape_hw=(int(dp.shape[1]), int(dp.shape[2])),
    )

    dp_path = out_dir / f"{object_name}_dp.hdf5"
    para_path = out_dir / f"{object_name}_para.hdf5"

    with h5py.File(dp_path, "w") as dp_file:
        dp_file.create_dataset("dp", data=dp)

    with h5py.File(para_path, "w") as para_file:
        object_dset = para_file.create_dataset("object", data=object_ds)
        object_dset.attrs["pixel_height_m"] = float(pixel_size_m)
        object_dset.attrs["pixel_width_m"] = float(pixel_size_m)
        object_dset.attrs["center_x_m"] = 0.0
        object_dset.attrs["center_y_m"] = 0.0

        probe_dset = para_file.create_dataset("probe", data=probe_ds)
        probe_dset.attrs["pixel_height_m"] = float(pixel_size_m)
        probe_dset.attrs["pixel_width_m"] = float(pixel_size_m)

        para_file.create_dataset(
            "probe_position_x_m",
            data=np.asarray(x, dtype=np.float64) * float(pixel_size_m),
        )
        para_file.create_dataset(
            "probe_position_y_m",
            data=np.asarray(y, dtype=np.float64) * float(pixel_size_m),
        )

    return PtychoViTHdf5Pair(dp_hdf5=dp_path, para_hdf5=para_path, object_name=object_name)

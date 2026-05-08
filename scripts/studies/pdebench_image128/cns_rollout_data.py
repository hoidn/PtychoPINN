"""Trajectory-window loading for PDEBench CNS rollout videos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import torch

from scripts.studies.pdebench_image128.data import CFD_CNS_FIELD_ORDER, _channel_first
from scripts.studies.pdebench_image128.normalization import normalize_batch
from scripts.studies.pdebench_image128.splits import axis_index


@dataclass(frozen=True)
class CnsTrajectoryWindow:
    trajectory_id: int
    sample_id: int
    start_time: int
    history_len: int
    field_order: tuple[str, ...]
    initial_history_norm: torch.Tensor
    initial_history_phys: torch.Tensor
    true_future_norm: torch.Tensor
    true_future_phys: torch.Tensor
    dt: float | None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_first_existing_json(run_root: Path, names: tuple[str, ...]) -> dict[str, Any]:
    for name in names:
        path = run_root / name
        if path.exists():
            return _read_json(path)
    raise FileNotFoundError(f"none of these metadata files exist under {run_root}: {', '.join(names)}")


def _resolve_data_file(run_root: Path, metadata: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return Path(override)
    raw_path = metadata.get("data_file") or metadata.get("dataset_file")
    if raw_path is None:
        raise KeyError("CNS metadata must include data_file or dataset_file")
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (run_root / path).resolve()


def _split_ids(split_manifest: dict[str, Any], split: str) -> list[int]:
    if "splits" in split_manifest and split in split_manifest["splits"]:
        return [int(item) for item in split_manifest["splits"][split]]
    if split in split_manifest:
        value = split_manifest[split]
        if isinstance(value, dict) and "indices" in value:
            return [int(item) for item in value["indices"]]
        if isinstance(value, list):
            return [int(item) for item in value]
    raise KeyError(f"split_manifest does not contain split {split!r}")


def _read_multifield_state(
    handle: h5py.File,
    *,
    field_order: tuple[str, ...],
    axis_order: str,
    trajectory_id: int,
    time_index: int,
) -> torch.Tensor:
    state_channels = []
    for field in field_order:
        dataset = handle[field]
        selection: list[Any] = [slice(None)] * len(axis_order)
        selection[axis_index(axis_order, "N")] = int(trajectory_id)
        selection[axis_index(axis_order, "T")] = int(time_index)
        array = dataset[tuple(selection)]
        remaining_axes = [axis for axis in axis_order if axis not in {"N", "T"}]
        state_channels.append(torch.from_numpy(_channel_first(array, "".join(remaining_axes))).float())
    return torch.cat(state_channels, dim=0)


def _metadata_time_steps(metadata: dict[str, Any]) -> int:
    if "time_steps" in metadata:
        return int(metadata["time_steps"])
    dimensions = metadata.get("dimensions")
    if isinstance(dimensions, dict) and "time_steps" in dimensions:
        return int(dimensions["time_steps"])
    raise KeyError("CNS metadata must include time_steps or dimensions.time_steps")


def load_cns_trajectory_window(
    *,
    run_root: Path,
    split: str = "test",
    sample_id: int = 0,
    start_time: int | None = None,
    steps: int = 20,
    data_file: Path | None = None,
) -> CnsTrajectoryWindow:
    """Load one CNS trajectory window for autoregressive rollout."""
    run_root = Path(run_root)
    metadata = _load_first_existing_json(run_root, ("hdf5_metadata.json", "dataset_manifest.json"))
    split_manifest = _read_json(run_root / "split_manifest.json")
    state_stats = _read_json(run_root / "normalization_stats_state.json")
    field_order = tuple(str(item) for item in metadata.get("field_order", state_stats.get("field_order", CFD_CNS_FIELD_ORDER)))
    axis_order = str(metadata.get("field_axis_order", metadata.get("axis_order", state_stats.get("axis_order", "NTHW")))).upper()
    history_len = int(metadata.get("history_len", state_stats.get("history_len", 1)))
    if start_time is None:
        start_time = history_len
    start_time = int(start_time)
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if start_time < history_len:
        raise ValueError(f"start_time must be >= history_len ({history_len})")
    time_steps = _metadata_time_steps(metadata)
    if start_time + steps > time_steps:
        raise ValueError(f"requested rollout exceeds trajectory length: start_time={start_time}, steps={steps}, time_steps={time_steps}")
    split_ids = _split_ids(split_manifest, split)
    sample_id = int(sample_id)
    if sample_id < 0 or sample_id >= len(split_ids):
        raise IndexError(f"sample_id {sample_id} outside split {split!r} with {len(split_ids)} trajectories")
    trajectory_id = int(split_ids[sample_id])
    resolved_data_file = _resolve_data_file(run_root, metadata, data_file)

    with h5py.File(resolved_data_file, "r") as handle:
        history_frames = [
            _read_multifield_state(
                handle,
                field_order=field_order,
                axis_order=axis_order,
                trajectory_id=trajectory_id,
                time_index=time_index,
            )
            for time_index in range(start_time - history_len, start_time)
        ]
        future_frames = [
            _read_multifield_state(
                handle,
                field_order=field_order,
                axis_order=axis_order,
                trajectory_id=trajectory_id,
                time_index=time_index,
            )
            for time_index in range(start_time, start_time + steps)
        ]

    initial_history_phys = torch.stack(history_frames).float()
    true_future_phys = torch.stack(future_frames).float()
    initial_history_norm = normalize_batch(initial_history_phys, state_stats)
    true_future_norm = normalize_batch(true_future_phys, state_stats)
    return CnsTrajectoryWindow(
        trajectory_id=trajectory_id,
        sample_id=sample_id,
        start_time=start_time,
        history_len=history_len,
        field_order=field_order,
        initial_history_norm=initial_history_norm,
        initial_history_phys=initial_history_phys,
        true_future_norm=true_future_norm,
        true_future_phys=true_future_phys,
        dt=None if metadata.get("dt") is None else float(metadata["dt"]),
    )

"""Deterministic split helpers for PDEBench 128x128 image-suite tasks."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


def axis_index(axis_order: str, token: str) -> int:
    try:
        return axis_order.index(token)
    except ValueError as exc:
        raise ValueError(f"axis_order {axis_order!r} is missing required token {token!r}") from exc


def infer_dynamic_dimensions(shape: list[int], axis_order: str) -> dict[str, int]:
    dims = {
        "num_trajectories": int(shape[axis_index(axis_order, "N")]),
        "time_steps": int(shape[axis_index(axis_order, "T")]),
        "height": int(shape[axis_index(axis_order, "H")]),
        "width": int(shape[axis_index(axis_order, "W")]),
        "channels": int(shape[axis_index(axis_order, "C")]) if "C" in axis_order else 1,
    }
    if dims["time_steps"] < 2:
        raise ValueError("dynamic one-step prediction requires at least two time steps")
    return dims


def build_trajectory_split(
    num_trajectories: int,
    *,
    seed: int = 20260420,
    counts: tuple[int, int, int] | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, Any]:
    """Build deterministic train/val/test trajectory-id splits."""
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")
    ids = list(range(int(num_trajectories)))
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    if counts is None:
        if len(ratios) != 3 or not math.isclose(sum(ratios), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("ratios must contain three values that sum to 1.0")
        train_count = int(num_trajectories * ratios[0])
        val_count = int(num_trajectories * ratios[1])
        test_count = num_trajectories - train_count - val_count
        if num_trajectories >= 3:
            if val_count == 0:
                val_count = 1
                train_count -= 1
            if test_count == 0:
                test_count = 1
                train_count -= 1
        if train_count <= 0:
            train_count = 1
            if val_count > test_count and val_count > 0:
                val_count -= 1
            elif test_count > 0:
                test_count -= 1
    else:
        train_count, val_count, test_count = (int(item) for item in counts)
        if train_count + val_count + test_count != num_trajectories:
            raise ValueError("split counts must sum to num_trajectories")
        if train_count <= 0 or val_count < 0 or test_count < 0:
            raise ValueError("invalid split counts")
    return {
        "seed": int(seed),
        "ratios": [float(item) for item in ratios],
        "train": ids[:train_count],
        "val": ids[train_count:train_count + val_count],
        "test": ids[train_count + val_count:train_count + val_count + test_count],
    }


def capped_trajectory_split(
    split: dict[str, Any],
    *,
    max_train_trajectories: int | None = None,
    max_val_trajectories: int | None = None,
    max_test_trajectories: int | None = None,
) -> dict[str, Any]:
    def cap(values: list[int], limit: int | None) -> list[int]:
        return list(values) if limit is None else list(values)[: int(limit)]

    return {
        "seed": int(split["seed"]),
        "ratios": list(split.get("ratios", [0.8, 0.1, 0.1])),
        "train": cap(list(split["train"]), max_train_trajectories),
        "val": cap(list(split["val"]), max_val_trajectories),
        "test": cap(list(split["test"]), max_test_trajectories),
    }


def build_sample_split(
    num_samples: int,
    *,
    seed: int = 20260420,
    counts: tuple[int, int, int] | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, Any]:
    """Build deterministic train/val/test sample-index splits."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    ids = list(range(int(num_samples)))
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    if counts is None:
        train_count = int(num_samples * ratios[0])
        val_count = int(num_samples * ratios[1])
        test_count = num_samples - train_count - val_count
    else:
        train_count, val_count, test_count = (int(item) for item in counts)
        if train_count + val_count + test_count != num_samples:
            raise ValueError("split counts must sum to num_samples")
    if train_count <= 0 or val_count < 0 or test_count < 0:
        raise ValueError("invalid split counts")
    return {
        "seed": int(seed),
        "train": ids[:train_count],
        "val": ids[train_count:train_count + val_count],
        "test": ids[train_count + val_count:train_count + val_count + test_count],
    }


def capped_split(
    split: dict[str, Any],
    *,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, Any]:
    def cap(values: list[int], limit: int | None) -> list[int]:
        return list(values) if limit is None else list(values)[: int(limit)]

    return {
        "seed": int(split["seed"]),
        "train": cap(list(split["train"]), max_train_samples),
        "val": cap(list(split["val"]), max_val_samples),
        "test": cap(list(split["test"]), max_test_samples),
    }


def _sha256_short(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sample_split_manifest(
    *,
    output_root: Path,
    data_file: Path,
    split: dict[str, Any],
    beta: float | None,
    input_dataset: str,
    target_dataset: str,
    filename: str = "split_manifest.json",
    compute_sha256: bool = False,
    extra: dict[str, Any] | None = None,
) -> Path:
    data_file = Path(data_file)
    payload: dict[str, Any] = {
        "schema_version": "pdebench_image128_sample_split_manifest_v1",
        "seed": int(split["seed"]),
        "source_file": {
            "path": str(data_file),
            "size_bytes": int(data_file.stat().st_size),
            "sha256": _sha256_short(data_file) if compute_sha256 else None,
        },
        "beta": beta,
        "input_dataset": input_dataset,
        "target_dataset": target_dataset,
        "split_counts": {name: len(split[name]) for name in ("train", "val", "test")},
        "splits": {name: [int(item) for item in split[name]] for name in ("train", "val", "test")},
        "split_unit": "sample_index",
        "overlap_allowed": False,
    }
    if extra:
        payload.update(extra)
    path = Path(output_root) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _window_count_for_trajectories(
    trajectory_ids: list[int],
    *,
    time_steps: int,
    history_len: int,
    max_windows_per_trajectory: int | None = None,
) -> int:
    windows_per = max(0, int(time_steps) - int(history_len))
    if max_windows_per_trajectory is not None:
        windows_per = min(windows_per, int(max_windows_per_trajectory))
    return len(trajectory_ids) * windows_per


def write_trajectory_split_manifest(
    *,
    output_root: Path,
    data_file: Path,
    split: dict[str, Any],
    state_dataset: str,
    axis_order: str,
    shape: list[int],
    history_len: int,
    max_windows_per_trajectory: int | None = None,
    filename: str = "split_manifest.json",
    compute_sha256: bool = False,
    extra: dict[str, Any] | None = None,
) -> Path:
    data_file = Path(data_file)
    dims = infer_dynamic_dimensions([int(item) for item in shape], axis_order)
    payload: dict[str, Any] = {
        "schema_version": "pdebench_image128_trajectory_split_manifest_v1",
        "seed": int(split["seed"]),
        "ratios": list(split.get("ratios", [0.8, 0.1, 0.1])),
        "source_file": {
            "path": str(data_file),
            "size_bytes": int(data_file.stat().st_size),
            "sha256": _sha256_short(data_file) if compute_sha256 else None,
        },
        "state_dataset": state_dataset,
        "axis_order": axis_order,
        "shape": [int(item) for item in shape],
        "dimensions": dims,
        "history_len": int(history_len),
        "split_counts": {name: len(split[name]) for name in ("train", "val", "test")},
        "window_counts": {
            name: _window_count_for_trajectories(
                list(split[name]),
                time_steps=dims["time_steps"],
                history_len=int(history_len),
                max_windows_per_trajectory=max_windows_per_trajectory,
            )
            for name in ("train", "val", "test")
        },
        "splits": {name: [int(item) for item in split[name]] for name in ("train", "val", "test")},
        "max_windows_per_trajectory": max_windows_per_trajectory,
        "split_unit": "trajectory_id",
        "overlap_allowed": False,
        "horizon": "one_step",
    }
    if extra:
        payload.update(extra)
    path = Path(output_root) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

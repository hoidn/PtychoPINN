"""Deterministic split and one-step pair helpers for PDEBench SWE."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


def build_trajectory_split(
    num_trajectories: int,
    *,
    seed: int = 20260420,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, Any]:
    """Build a deterministic trajectory-level train/val/test split."""
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")
    if len(ratios) != 3 or not math.isclose(sum(ratios), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("ratios must contain three values that sum to 1.0")

    ids = list(range(num_trajectories))
    rng = random.Random(seed)
    rng.shuffle(ids)

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

    train = ids[:train_count]
    val = ids[train_count:train_count + val_count]
    test = ids[train_count + val_count:train_count + val_count + test_count]
    return {
        "seed": int(seed),
        "ratios": [float(item) for item in ratios],
        "train": train,
        "val": val,
        "test": test,
    }


def axis_index(axis_order: str, token: str) -> int:
    try:
        return axis_order.index(token)
    except ValueError as exc:
        raise ValueError(f"axis_order {axis_order!r} is missing required token {token!r}") from exc


def infer_dimensions(shape: list[int], axis_order: str) -> dict[str, int]:
    dims = {
        "num_trajectories": int(shape[axis_index(axis_order, "N")]),
        "time_steps": int(shape[axis_index(axis_order, "T")]),
        "height": int(shape[axis_index(axis_order, "H")]),
        "width": int(shape[axis_index(axis_order, "W")]),
        "channels": int(shape[axis_index(axis_order, "C")]) if "C" in axis_order else 1,
    }
    if dims["time_steps"] < 2:
        raise ValueError("one-step prediction requires at least two time steps")
    return dims


def pair_count_for_trajectories(
    trajectory_ids: list[int],
    *,
    time_steps: int,
    max_pairs_per_trajectory: int | None = None,
) -> int:
    pairs_per = max(0, time_steps - 1)
    if max_pairs_per_trajectory is not None:
        pairs_per = min(pairs_per, int(max_pairs_per_trajectory))
    return len(trajectory_ids) * pairs_per


def build_run_subset_split(
    full_split: dict[str, Any],
    *,
    max_train_trajectories: int | None = None,
    max_val_trajectories: int | None = None,
    max_test_trajectories: int | None = None,
) -> dict[str, Any]:
    """Build a capped split that preserves prefix order from the full split."""

    def limit(ids: list[int], cap: int | None) -> list[int]:
        return list(ids) if cap is None else list(ids)[: int(cap)]

    return {
        "seed": int(full_split["seed"]),
        "ratios": list(full_split["ratios"]),
        "train": limit(list(full_split["train"]), max_train_trajectories),
        "val": limit(list(full_split["val"]), max_val_trajectories),
        "test": limit(list(full_split["test"]), max_test_trajectories),
    }


def _is_subset_prefix(full_split: dict[str, Any], run_split: dict[str, Any]) -> bool:
    return all(
        list(run_split[name]) == list(full_split[name])[: len(run_split[name])]
        for name in ("train", "val", "test")
    )


def write_split_manifest(
    *,
    output_root: Path,
    source_file_identity: dict[str, Any],
    state_dataset: str,
    axis_order: str,
    shape: list[int],
    split: dict[str, Any],
    max_pairs_per_trajectory: int | None = None,
    run_id: str | None = None,
    filename: str = "split_manifest.json",
    manifest_kind: str = "split",
    extra: dict[str, Any] | None = None,
) -> Path:
    dims = infer_dimensions(shape, axis_order)
    pair_counts = {
        name: pair_count_for_trajectories(
            list(split[name]),
            time_steps=dims["time_steps"],
            max_pairs_per_trajectory=max_pairs_per_trajectory,
        )
        for name in ("train", "val", "test")
    }
    payload = {
        "schema_version": "pdebench_swe_split_manifest_v1",
        "manifest_kind": manifest_kind,
        "run_id": run_id,
        "seed": int(split["seed"]),
        "ratios": split["ratios"],
        "source_file_identity": source_file_identity,
        "state_dataset": state_dataset,
        "axis_order": axis_order,
        "shape": [int(item) for item in shape],
        "dimensions": dims,
        "splits": {
            "train": list(split["train"]),
            "val": list(split["val"]),
            "test": list(split["test"]),
        },
        "max_pairs_per_trajectory": max_pairs_per_trajectory,
        "pair_counts": pair_counts,
        "horizon": "one_step",
    }
    if extra:
        payload.update(extra)
    path = Path(output_root) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_longer_split_manifests(
    *,
    output_root: Path,
    source_file_identity: dict[str, Any],
    state_dataset: str,
    axis_order: str,
    shape: list[int],
    full_split: dict[str, Any],
    run_split: dict[str, Any],
    max_pairs_per_trajectory: int,
    run_id: str | None = None,
) -> dict[str, Path]:
    """Write full 1000-ID and budget-capped split manifests for longer runs."""

    full_path = write_split_manifest(
        output_root=output_root,
        source_file_identity=source_file_identity,
        state_dataset=state_dataset,
        axis_order=axis_order,
        shape=shape,
        split=full_split,
        max_pairs_per_trajectory=None,
        run_id=run_id,
        filename="split_manifest_full.json",
        manifest_kind="full_split",
    )
    run_path = write_split_manifest(
        output_root=output_root,
        source_file_identity=source_file_identity,
        state_dataset=state_dataset,
        axis_order=axis_order,
        shape=shape,
        split=run_split,
        max_pairs_per_trajectory=max_pairs_per_trajectory,
        run_id=run_id,
        filename="split_manifest_run.json",
        manifest_kind="run_subset",
        extra={
            "full_split_manifest": full_path.name,
            "subset_of_full_split": _is_subset_prefix(full_split, run_split),
        },
    )
    return {"full": full_path, "run": run_path}

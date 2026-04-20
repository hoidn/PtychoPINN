"""Deterministic split helpers for PDEBench 128x128 image-suite tasks."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


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

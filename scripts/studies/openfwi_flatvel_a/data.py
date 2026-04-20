"""Shape validation and lazy datasets for OpenFWI FlatVel-A smoke shards."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


EXPECTED_DATA_SHAPE_SUFFIX = (5, 1000, 70)
EXPECTED_MODEL_SHAPE_SUFFIX = (1, 70, 70)


def inspect_shard_pair(data_path: Path, model_path: Path) -> dict[str, Any]:
    """Inspect a seismic/velocity shard pair without eager loading."""
    data_path = Path(data_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    data = np.load(data_path, mmap_mode="r")
    model = np.load(model_path, mmap_mode="r")
    data_shape = [int(item) for item in data.shape]
    model_shape = [int(item) for item in model.shape]
    differences: list[str] = []
    if tuple(data_shape[1:]) != EXPECTED_DATA_SHAPE_SUFFIX:
        differences.append("data_shape_suffix")
    if tuple(model_shape[1:]) != EXPECTED_MODEL_SHAPE_SUFFIX:
        differences.append("model_shape_suffix")
    if data_shape[0] != model_shape[0]:
        differences.append("sample_count")
    return {
        "data_shard": data_path.name,
        "model_shard": model_path.name,
        "data_path": str(data_path),
        "model_path": str(model_path),
        "data_shape": data_shape,
        "model_shape": model_shape,
        "data_dtype": str(data.dtype),
        "model_dtype": str(model.dtype),
        "num_samples": int(min(data_shape[0], model_shape[0])),
        "expected_data_shape_suffix": list(EXPECTED_DATA_SHAPE_SUFFIX),
        "expected_model_shape_suffix": list(EXPECTED_MODEL_SHAPE_SUFFIX),
        "status": "valid" if not differences else "schema_difference",
        "differences": differences,
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_split_manifest(
    *,
    train_count: int,
    test_count: int,
    train_samples: int = 32,
    val_samples: int = 16,
    test_samples: int = 16,
    seed: int = 20260420,
) -> dict[str, Any]:
    """Build deterministic smoke splits from data1/model1 and data49/model49."""
    rng = np.random.default_rng(int(seed))
    train_indices = np.arange(int(train_count))
    test_indices = np.arange(int(test_count))
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    val_count = min(int(val_samples), len(test_indices))
    test_count_cap = min(int(test_samples), max(0, len(test_indices) - val_count))
    return {
        "schema_version": "openfwi_flatvel_a_split_manifest_v1",
        "seed": int(seed),
        "sample_caps": {
            "train_samples": int(train_samples),
            "val_samples": int(val_samples),
            "test_samples": int(test_samples),
        },
        "train": {
            "data_shard": "data1.npy",
            "model_shard": "model1.npy",
            "indices": [int(item) for item in train_indices[: int(train_samples)]],
        },
        "val": {
            "data_shard": "data49.npy",
            "model_shard": "model49.npy",
            "indices": [int(item) for item in test_indices[:val_count]],
        },
        "test": {
            "data_shard": "data49.npy",
            "model_shard": "model49.npy",
            "indices": [int(item) for item in test_indices[val_count : val_count + test_count_cap]],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


class OpenFWIShardDataset(Dataset):
    """Lazy mmap-backed FlatVel-A shard dataset for local 2D adapters."""

    def __init__(
        self,
        *,
        data_path: Path,
        model_path: Path,
        split_name: str,
        indices: list[int],
        normalization: dict[str, Any] | None = None,
        input_resize_mode: str = "bilinear",
    ):
        self.data_path = Path(data_path).expanduser().resolve()
        self.model_path = Path(model_path).expanduser().resolve()
        self.split_name = split_name
        self.indices = [int(item) for item in indices]
        self.normalization = normalization
        self.input_resize_mode = input_resize_mode
        self._data = None
        self._model = None

    def __len__(self) -> int:
        return len(self.indices)

    def _arrays(self):
        if self._data is None:
            self._data = np.load(self.data_path, mmap_mode="r")
        if self._model is None:
            self._model = np.load(self.model_path, mmap_mode="r")
        return self._data, self._model

    def _resize_input(self, raw: torch.Tensor) -> torch.Tensor:
        tensor = raw.unsqueeze(0)
        if self.input_resize_mode != "bilinear":
            raise ValueError(f"unsupported input_resize_mode: {self.input_resize_mode}")
        resized = F.interpolate(tensor, size=(70, 70), mode="bilinear", align_corners=False)
        return resized.squeeze(0)

    def _apply_normalization(self, input_tensor: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.normalization:
            return input_tensor, target
        input_stats = self.normalization["input"]
        target_stats = self.normalization["target"]
        input_tensor = (input_tensor - float(input_stats["mean"])) / max(float(input_stats["std"]), 1e-12)
        target = (target - float(target_stats["mean"])) / max(float(target_stats["std"]), 1e-12)
        return input_tensor, target

    def __getitem__(self, index: int) -> dict[str, Any]:
        data, model = self._arrays()
        sample_index = self.indices[index]
        input_raw = torch.from_numpy(np.array(data[sample_index], dtype=np.float32, copy=True)).float()
        target = torch.from_numpy(np.array(model[sample_index], dtype=np.float32, copy=True)).float()
        input_tensor = self._resize_input(input_raw)
        input_tensor, target = self._apply_normalization(input_tensor, target)
        return {
            "input_raw": input_raw,
            "input": input_tensor,
            "target": target,
            "sample_id": f"{self.data_path.name}:{sample_index}",
            "source_shard": self.data_path.name,
            "model_shard": self.model_path.name,
            "split": self.split_name,
            "preprocessing": {
                "input_resize_mode": self.input_resize_mode,
                "input_shape_before": list(input_raw.shape),
                "input_shape_after": list(input_tensor.shape),
            },
        }


def _mean_std_from_tensors(tensors: list[torch.Tensor]) -> tuple[float, float]:
    if not tensors:
        raise ValueError("cannot compute normalization stats from an empty dataset")
    flat = torch.cat([item.reshape(-1).double() for item in tensors])
    mean = flat.mean()
    std = flat.std(unbiased=False)
    if float(std) <= 1e-12:
        std = torch.tensor(1.0, dtype=torch.float64)
    return float(mean.item()), float(std.item())


def compute_normalization_stats(dataset: OpenFWIShardDataset, *, max_samples: int | None = None) -> dict[str, Any]:
    """Compute scalar input/target stats from train split samples only."""
    limit = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    original_normalization = dataset.normalization
    dataset.normalization = None
    try:
        for index in range(limit):
            item = dataset[index]
            inputs.append(item["input"].float())
            targets.append(item["target"].float())
    finally:
        dataset.normalization = original_normalization
    input_mean, input_std = _mean_std_from_tensors(inputs)
    target_mean, target_std = _mean_std_from_tensors(targets)
    return {
        "schema_version": "openfwi_flatvel_a_normalization_stats_v1",
        "source": "train_split_only",
        "num_samples": int(limit),
        "input": {"mean": input_mean, "std": input_std},
        "target": {"mean": target_mean, "std": target_std},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

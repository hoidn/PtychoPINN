from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
PROVISIONING_DECISION_PATH = (
    REPO_ROOT
    / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/provisioning_decision.json"
)
DATASET_MANIFEST_PATH = (
    REPO_ROOT
    / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/dataset_manifest.json"
)
PRELIGHT_METADATA_PATH = (
    REPO_ROOT
    / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/preflight_metadata.json"
)
LOCKED_ROWS = (
    "cnn",
    "hybrid_resnet",
    "spectral_resnet_bottleneck_net",
    "fno",
    "ffno",
)
LOCKED_LATENT_CHANNELS = (32, 64)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.exists(), f"missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_locked_contract(repo_root: Path | str = ".") -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    provisioning = load_json(repo_root / PROVISIONING_DECISION_PATH.relative_to(REPO_ROOT))
    dataset_manifest = load_json(repo_root / DATASET_MANIFEST_PATH.relative_to(REPO_ROOT))
    preflight = load_json(repo_root / PRELIGHT_METADATA_PATH.relative_to(REPO_ROOT))

    return {
        "selected_variant": provisioning["selected_variant"],
        "selected_dataset_member": provisioning["selected_dataset_member"],
        "stable_dataset_target": {
            "repo_relative": provisioning["stable_dataset_target"]["repo_relative"],
            "description": provisioning["stable_dataset_target"]["description"],
        },
        "observed_wavebench_checkout": dataset_manifest["observed_wavebench_checkout"],
        "split": {"train": 9000, "val": 500, "test": 500, "seed": 42},
        "tensor_contract": {
            "input_shape": preflight["tensor_contracts"]["observed_y"]["archive_shape_per_sample"],
            "target_shape": preflight["tensor_contracts"]["target_q0"]["archive_shape_per_sample"],
            "dtype": "float32",
        },
        "latent_channels": list(LOCKED_LATENT_CHANNELS),
        "row_roster": list(LOCKED_ROWS),
    }


def build_split_indices(
    *,
    train: int = 9000,
    val: int = 500,
    test: int = 500,
    seed: int = 42,
    total_samples: int = 10000,
) -> dict[str, list[int]]:
    require(train + val + test <= total_samples, "requested split exceeds total_samples")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_samples, generator=generator).tolist()
    return {
        "train": indices[:train],
        "val": indices[train : train + val],
        "test": indices[train + val : train + val + test],
    }


def trim_split_indices(
    split_indices: dict[str, list[int]],
    *,
    train: int | None = None,
    val: int | None = None,
    test: int | None = None,
) -> dict[str, list[int]]:
    return {
        "train": split_indices["train"] if train is None else split_indices["train"][:train],
        "val": split_indices["val"] if val is None else split_indices["val"][:val],
        "test": split_indices["test"] if test is None else split_indices["test"][:test],
    }


def summarize_tensor_batch(inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    return {
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "input_dtype": str(inputs.dtype),
        "target_dtype": str(targets.dtype),
        "input_min": float(inputs.min().item()),
        "input_max": float(inputs.max().item()),
        "target_min": float(targets.min().item()),
        "target_max": float(targets.max().item()),
    }


def ensure_wavebench_on_path(wavebench_root: Path) -> None:
    root_text = str(wavebench_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def build_dataloaders(
    *,
    repo_root: Path | str = ".",
    wavebench_root: Path | str = "tmp/wavebench_repo",
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 0,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    wavebench_root = (
        Path(wavebench_root).resolve()
        if Path(wavebench_root).is_absolute()
        else (repo_root / wavebench_root).resolve()
    )
    contract = load_locked_contract(repo_root)
    dataset_path = wavebench_root / contract["selected_dataset_member"]
    require(dataset_path.exists(), f"missing staged WaveBench dataset member: {dataset_path}")

    ensure_wavebench_on_path(wavebench_root)
    from ffcv.fields.decoders import NDArrayDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import ToTensor

    split_indices = trim_split_indices(
        build_split_indices(**contract["split"]),
        train=max_train_samples,
        val=max_val_samples,
        test=max_test_samples,
    )

    batch_sizes = {"train": train_batch_size, "val": eval_batch_size, "test": eval_batch_size}
    return {
        split: Loader(
            str(dataset_path),
            batch_size=batch_sizes[split],
            num_workers=num_workers,
            order=OrderOption.RANDOM if split == "train" else OrderOption.SEQUENTIAL,
            indices=split_indices[split],
            pipelines={"input": [NDArrayDecoder(), ToTensor()], "target": [NDArrayDecoder(), ToTensor()]},
        )
        for split in ("train", "val", "test")
    }

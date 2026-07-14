"""Regression coverage for CUDA collation of expanded CI probe fields."""

from __future__ import annotations

from pathlib import Path

import torch
from tensordict import MemoryMappedTensor, TensorDict

from ptycho_torch.dataloader import Collate


def _expanded_probe(
    *,
    batch: int = 2,
    channels: int = 4,
    modes: int = 3,
    height: int = 4,
    width: int = 5,
    offset: float = 0.0,
) -> torch.Tensor:
    base = torch.arange(batch * modes * height * width, dtype=torch.float32) + offset
    return base.reshape(batch, 1, modes, height, width).expand(
        batch, channels, modes, height, width
    )


def _has_expanded_overlap(value: torch.Tensor) -> bool:
    return any(size > 1 and stride == 0 for size, stride in zip(value.shape, value.stride()))


def test_collate_materializes_expanded_multimode_probe_fields_and_probe() -> None:
    probe_physical = _expanded_probe()
    probe_training = _expanded_probe(offset=1000.0)
    separate_probe = _expanded_probe(offset=2000.0)
    tensor_dict = TensorDict(
        {
            "probe_physical": probe_physical,
            "probe_training": probe_training,
        },
        batch_size=[2],
    )
    scaling = torch.ones(2, 1, 1, 1)

    output, output_probe, output_scaling = Collate(device="cpu")(
        (tensor_dict, separate_probe, scaling)
    )

    for original, materialized in (
        (probe_physical, output["probe_physical"]),
        (probe_training, output["probe_training"]),
        (separate_probe, output_probe),
    ):
        assert materialized.shape == (2, 4, 3, 4, 5)
        assert torch.equal(materialized, original)
        assert materialized.data_ptr() != original.data_ptr()
        assert not _has_expanded_overlap(materialized)
    assert output_scaling is scaling


def test_collate_materializes_expanded_fields_recursively() -> None:
    expanded = _expanded_probe(modes=2)
    nested = TensorDict({"probe_training": expanded}, batch_size=[2])
    tensor_dict = TensorDict({"nested": nested}, batch_size=[2])

    output, _, _ = Collate(device="cpu")(
        (tensor_dict, torch.ones(2, 1), torch.ones(2, 1))
    )

    materialized = output["nested", "probe_training"]
    assert torch.equal(materialized, expanded)
    assert materialized.data_ptr() != expanded.data_ptr()
    assert not _has_expanded_overlap(materialized)


def test_collate_reuses_ordinary_contiguous_mmap_and_scaling_storage(
    tmp_path: Path,
) -> None:
    contiguous = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    mmap_backed = MemoryMappedTensor.from_tensor(
        torch.arange(16, dtype=torch.float32).reshape(2, 2, 4),
        filename=tmp_path / "ordinary.memmap",
    )
    tensor_dict = TensorDict(
        {"contiguous": contiguous, "mmap_backed": mmap_backed},
        batch_size=[2],
    )
    probe = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    scaling = torch.arange(2, dtype=torch.float32).reshape(2, 1, 1, 1)

    output, output_probe, output_scaling = Collate(device="cpu")(
        (tensor_dict, probe, scaling)
    )

    assert output["contiguous"].data_ptr() == contiguous.data_ptr()
    assert output["mmap_backed"].data_ptr() == mmap_backed.data_ptr()
    assert output_probe.data_ptr() == probe.data_ptr()
    assert output_scaling.data_ptr() == scaling.data_ptr()
    assert output["contiguous"].shape == contiguous.shape
    assert output["mmap_backed"].shape == mmap_backed.shape
    assert output_probe.shape == probe.shape
    assert output_scaling.shape == scaling.shape
    assert torch.equal(output["contiguous"], contiguous)
    assert torch.equal(output["mmap_backed"], mmap_backed)
    assert torch.equal(output_probe, probe)
    assert torch.equal(output_scaling, scaling)

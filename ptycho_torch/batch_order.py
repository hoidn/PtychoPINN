"""Explicit, versioned recipes for Torch training-batch order."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sized
from typing import Any

import torch
from torch.utils.data import Sampler


DEFAULT_BATCH_ORDER_RECIPE = "torch-generator-v1"
JULY2026_BATCH_ORDER_RECIPE = "torch-implicit-july2026-v1"
SUPPORTED_BATCH_ORDER_RECIPES = frozenset(
    {
        DEFAULT_BATCH_ORDER_RECIPE,
        JULY2026_BATCH_ORDER_RECIPE,
    }
)

_RECIPE_SPECIFICATIONS: dict[str, dict[str, Any]] = {
    DEFAULT_BATCH_ORDER_RECIPE: {
        "recipe": DEFAULT_BATCH_ORDER_RECIPE,
        "index_order": "torch.utils.data.RandomSampler",
        "generator": "private torch.Generator seeded from torch_training_seed",
        "loader_generator_coupling": True,
    },
    JULY2026_BATCH_ORDER_RECIPE: {
        "recipe": JULY2026_BATCH_ORDER_RECIPE,
        "reference_runtime": "torch-2.9.1+cu128-cpu",
        "schedule_generator": "private torch.Generator(device=cpu)",
        "schedule_draw": "torch.int64 random_",
        "permutation_seed_offset": 1,
        "permutation_seed_stride": 3,
        "permutation": "torch.randperm(device=cpu,dtype=int64)",
    },
}
_JULY2026_REFERENCE_RUNTIME = "torch-2.9.1+cu128-cpu"
_JULY2026_RUNTIME_CONFORMANCE = (
    (
        3839649608876091011,
        "40f5f7ff7e47d426d52d43919b650b249d3b2498090765e89b26c9871a0e3f5b",
    ),
    (
        7248591967756700307,
        "8104bdad0539032f6abc198e6eb8451dad6cb19ea320a26cc3fd000a3553f770",
    ),
    (
        3822956344058944405,
        "91317cbad6c8f67c4a23c2d3142521c3e7ec4d0e2776e85fc0166055dcfb27c2",
    ),
)


def validate_batch_order_recipe(value: Any) -> str:
    """Return one exact recipe identifier or reject ambiguous input."""

    if not isinstance(value, str):
        raise TypeError("batch_order_recipe must be a string")
    if value not in SUPPORTED_BATCH_ORDER_RECIPES:
        expected = ", ".join(repr(item) for item in sorted(SUPPORTED_BATCH_ORDER_RECIPES))
        raise ValueError(
            f"batch_order_recipe must be one of {expected}; got {value!r}"
        )
    return value


def validate_batch_order_distribution(
    recipe: str,
    *,
    is_distributed: bool,
) -> str:
    """Reject a historical order after Lightning resolves its true strategy."""

    recipe = validate_batch_order_recipe(recipe)
    if not isinstance(is_distributed, bool):
        raise TypeError("is_distributed must be a bool")
    if recipe == JULY2026_BATCH_ORDER_RECIPE and is_distributed:
        raise ValueError(
            f"batch_order_recipe={JULY2026_BATCH_ORDER_RECIPE!r} is a "
            "single-device historical reproduction contract; Lightning "
            "distributed strategies replace its sampler"
        )
    return recipe


def validate_batch_order_loader_schedule(
    recipe: str,
    *,
    has_validation_loader: bool,
) -> str:
    """Reject loader schedules that cannot realize a named historical recipe."""

    recipe = validate_batch_order_recipe(recipe)
    if not isinstance(has_validation_loader, bool):
        raise TypeError("has_validation_loader must be a bool")
    if recipe == JULY2026_BATCH_ORDER_RECIPE and not has_validation_loader:
        raise ValueError(
            f"batch_order_recipe={JULY2026_BATCH_ORDER_RECIPE!r} requires a "
            "validation loader evaluated every epoch; its sealed stride-3 "
            "seed schedule is undefined for validation-free training"
        )
    return recipe


def _validate_nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _implementation_sha256(recipe: str) -> str:
    encoded = json.dumps(
        _RECIPE_SPECIFICATIONS[recipe],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _order_sha256(order: torch.Tensor) -> str:
    contiguous = order.to(device="cpu", dtype=torch.int64).contiguous()
    payload = contiguous.numpy().astype("<i8", copy=False).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


class July2026BatchOrderSampler(Sampler[int]):
    """RNG-isolated reproduction of the July 2026 implicit shuffle schedule.

    The historical run relied on three ambient CPU-generator draws per epoch:
    the train loader base seed, the train ``RandomSampler`` child seed, and the
    validation loader base seed.  This sampler encodes that schedule directly,
    so unrelated global RNG consumers can no longer change the order.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        seed: int,
        recipe: str = JULY2026_BATCH_ORDER_RECIPE,
    ) -> None:
        if validate_batch_order_recipe(recipe) != JULY2026_BATCH_ORDER_RECIPE:
            raise ValueError(
                "July2026BatchOrderSampler requires "
                f"batch_order_recipe={JULY2026_BATCH_ORDER_RECIPE!r}"
            )
        self.data_source = data_source
        self.seed = _validate_nonnegative_integer(seed, name="seed")
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _validate_nonnegative_integer(epoch, name="epoch")

    def permutation_seed(self, epoch: int | None = None) -> int:
        """Return the historical child seed for one epoch without global RNG."""

        resolved_epoch = self.epoch if epoch is None else epoch
        resolved_epoch = _validate_nonnegative_integer(
            resolved_epoch,
            name="epoch",
        )
        schedule = torch.Generator(device="cpu")
        schedule.manual_seed(self.seed)
        child_seed = 0
        for _ in range(3 * resolved_epoch + 2):
            child_seed = int(
                torch.empty((), dtype=torch.int64).random_(
                    generator=schedule,
                ).item()
            )
        return child_seed

    def order(self, epoch: int | None = None) -> torch.Tensor:
        """Materialize one CPU int64 permutation under the frozen recipe."""

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.permutation_seed(epoch))
        return torch.randperm(
            len(self),
            generator=generator,
            dtype=torch.int64,
            device="cpu",
        )

    def __iter__(self):
        return iter(self.order().tolist())


def validate_july2026_runtime_conformance() -> dict[str, Any]:
    """Fail if this Torch CPU RNG no longer realizes the sealed July orders."""

    sampler = July2026BatchOrderSampler(range(8978), seed=3)
    epochs: dict[str, dict[str, Any]] = {}
    for epoch, (expected_seed, expected_hash) in enumerate(
        _JULY2026_RUNTIME_CONFORMANCE
    ):
        actual_seed = sampler.permutation_seed(epoch)
        actual_hash = _order_sha256(sampler.order(epoch))
        if actual_seed != expected_seed or actual_hash != expected_hash:
            raise RuntimeError(
                "historical July 2026 batch-order runtime conformance failed: "
                "canonical CPU permutation fingerprint mismatch for epoch "
                f"{epoch}; reference {_JULY2026_REFERENCE_RUNTIME}; "
                f"expected seed/hash {expected_seed}/{expected_hash}, got "
                f"{actual_seed}/{actual_hash}; refusing to train"
            )
        epochs[str(epoch)] = {
            "permutation_seed": actual_seed,
            "order_sha256": actual_hash,
        }
    return {
        "status": "passed",
        "reference_runtime": _JULY2026_REFERENCE_RUNTIME,
        "seed": 3,
        "dataset_size": 8978,
        "epochs": epochs,
    }


def batch_order_provenance(
    *,
    recipe: str,
    seed: int,
    dataset_size: int,
    reference_epochs: int = 0,
) -> dict[str, Any]:
    """Describe the effective order recipe without consuming runtime RNG."""

    recipe = validate_batch_order_recipe(recipe)
    seed = _validate_nonnegative_integer(seed, name="seed")
    dataset_size = _validate_nonnegative_integer(
        dataset_size,
        name="dataset_size",
    )
    reference_epochs = _validate_nonnegative_integer(
        reference_epochs,
        name="reference_epochs",
    )
    record: dict[str, Any] = {
        "schema_version": "torch-batch-order-v1",
        "recipe": recipe,
        "seed": seed,
        "dataset_size": dataset_size,
        "implementation_sha256": _implementation_sha256(recipe),
        "torch_version": str(torch.__version__),
    }
    if recipe == JULY2026_BATCH_ORDER_RECIPE:
        record["runtime_conformance"] = (
            validate_july2026_runtime_conformance()
        )
    hashes: dict[str, str] = {}
    if reference_epochs:
        if recipe != JULY2026_BATCH_ORDER_RECIPE:
            raise ValueError(
                "reference epoch hashes are only defined for "
                f"batch_order_recipe={JULY2026_BATCH_ORDER_RECIPE!r}"
            )
        sampler = July2026BatchOrderSampler(range(dataset_size), seed=seed)
        for epoch in range(reference_epochs):
            hashes[str(epoch)] = _order_sha256(sampler.order(epoch))
    record["epoch_sha256"] = hashes
    return record


__all__ = [
    "DEFAULT_BATCH_ORDER_RECIPE",
    "JULY2026_BATCH_ORDER_RECIPE",
    "July2026BatchOrderSampler",
    "SUPPORTED_BATCH_ORDER_RECIPES",
    "batch_order_provenance",
    "validate_batch_order_distribution",
    "validate_batch_order_loader_schedule",
    "validate_batch_order_recipe",
    "validate_july2026_runtime_conformance",
]

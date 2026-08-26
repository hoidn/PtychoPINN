"""Versioned training-batch order contracts."""

import hashlib

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def _order_digest(order) -> str:
    tensor = torch.as_tensor(order, dtype=torch.int64).contiguous()
    return hashlib.sha256(tensor.numpy().astype("<i8", copy=False).tobytes()).hexdigest()


def test_july2026_recipe_reproduces_the_recorded_epoch_orders():
    from ptycho_torch.batch_order import (
        JULY2026_BATCH_ORDER_RECIPE,
        July2026BatchOrderSampler,
    )

    sampler = July2026BatchOrderSampler(
        range(8978),
        seed=3,
        recipe=JULY2026_BATCH_ORDER_RECIPE,
    )
    expected = (
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

    for epoch, (child_seed, digest) in enumerate(expected):
        sampler.set_epoch(epoch)
        assert sampler.permutation_seed(epoch) == child_seed
        assert _order_digest(list(iter(sampler))) == digest


def test_july2026_recipe_is_isolated_from_global_rng_and_loader_worker_rng():
    from ptycho_torch.batch_order import July2026BatchOrderSampler

    dataset = TensorDataset(torch.arange(64))

    def actual_order(*, global_seed: int, worker_seed: int):
        torch.manual_seed(global_seed)
        state_before = torch.random.get_rng_state().clone()
        sampler = July2026BatchOrderSampler(dataset, seed=3)
        sampler.set_epoch(2)
        worker_generator = torch.Generator().manual_seed(worker_seed)
        loader = DataLoader(
            dataset,
            batch_size=7,
            sampler=sampler,
            generator=worker_generator,
        )
        order = [int(value) for batch in loader for value in batch[0]]
        assert torch.equal(torch.random.get_rng_state(), state_before)
        return order

    assert actual_order(global_seed=11, worker_seed=19) == actual_order(
        global_seed=97,
        worker_seed=101,
    )


def test_july2026_recipe_set_epoch_is_repeatable_and_resume_safe():
    from ptycho_torch.batch_order import July2026BatchOrderSampler

    sampler = July2026BatchOrderSampler(range(31), seed=73)
    sampler.set_epoch(4)
    first = list(iter(sampler))
    sampler.set_epoch(4)
    assert list(iter(sampler)) == first
    sampler.set_epoch(5)
    assert list(iter(sampler)) != first


def test_batch_order_provenance_identifies_recipe_implementation_and_orders():
    from ptycho_torch.batch_order import (
        JULY2026_BATCH_ORDER_RECIPE,
        batch_order_provenance,
    )

    record = batch_order_provenance(
        recipe=JULY2026_BATCH_ORDER_RECIPE,
        seed=3,
        dataset_size=8978,
        reference_epochs=3,
    )

    assert record["schema_version"] == "torch-batch-order-v1"
    assert record["recipe"] == JULY2026_BATCH_ORDER_RECIPE
    assert record["seed"] == 3
    assert record["dataset_size"] == 8978
    assert len(record["implementation_sha256"]) == 64
    assert record["epoch_sha256"]["0"] == (
        "40f5f7ff7e47d426d52d43919b650b249d3b2498090765e89b26c9871a0e3f5b"
    )
    assert record["runtime_conformance"]["status"] == "passed"
    assert record["runtime_conformance"]["reference_runtime"] == (
        "torch-2.9.1+cu128-cpu"
    )


def test_historical_recipe_fails_if_runtime_no_longer_matches_reference(
    monkeypatch,
):
    from ptycho_torch import batch_order

    monkeypatch.setattr(batch_order, "_order_sha256", lambda _order: "0" * 64)

    with pytest.raises(RuntimeError, match="runtime conformance"):
        batch_order.batch_order_provenance(
            recipe=batch_order.JULY2026_BATCH_ORDER_RECIPE,
            seed=3,
            dataset_size=8978,
            reference_epochs=3,
        )


def test_historical_recipe_requires_the_validation_loader_schedule():
    from ptycho_torch.batch_order import validate_batch_order_loader_schedule

    with pytest.raises(ValueError, match="validation loader"):
        validate_batch_order_loader_schedule(
            "torch-implicit-july2026-v1",
            has_validation_loader=False,
        )

    assert validate_batch_order_loader_schedule(
        "torch-implicit-july2026-v1",
        has_validation_loader=True,
    ) == "torch-implicit-july2026-v1"
    assert validate_batch_order_loader_schedule(
        "torch-generator-v1",
        has_validation_loader=False,
    ) == "torch-generator-v1"


@pytest.mark.parametrize("value", [None, "implicit", "torch-generator-v2", 3])
def test_batch_order_recipe_rejects_unknown_values(value):
    from ptycho_torch.batch_order import validate_batch_order_recipe

    with pytest.raises((TypeError, ValueError), match="batch_order_recipe"):
        validate_batch_order_recipe(value)


def test_lightning_advances_the_historical_sampler_epoch():
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset

    from ptycho_torch.batch_order import July2026BatchOrderSampler

    dataset = TensorDataset(torch.arange(12, dtype=torch.float32).reshape(-1, 1))
    sampler = July2026BatchOrderSampler(dataset, seed=3)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    observed_epochs = []

    class Module(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def training_step(self, batch, _batch_idx):
            return (batch[0].mean() * self.weight) ** 2

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-4)

    class Capture(L.Callback):
        def on_train_epoch_start(self, _trainer, _module):
            observed_epochs.append(sampler.epoch)

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        callbacks=[Capture()],
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        limit_train_batches=1,
    )
    trainer.fit(Module(), train_dataloaders=loader)

    assert observed_epochs == [0, 1, 2]


def test_historical_recipe_fails_after_auto_strategy_resolves_distributed():
    from ptycho_torch.batch_order import validate_batch_order_distribution

    with pytest.raises(ValueError, match="single-device.*distributed"):
        validate_batch_order_distribution(
            "torch-implicit-july2026-v1",
            is_distributed=True,
        )

    assert validate_batch_order_distribution(
        "torch-generator-v1",
        is_distributed=True,
    ) == "torch-generator-v1"

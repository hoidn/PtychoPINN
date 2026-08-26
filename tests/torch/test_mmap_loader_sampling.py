"""Sampling contracts shared by the in-memory and mmap training rails."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from ptycho_torch.dataloader import PtychoDataset


class _IndexableDataset(PtychoDataset):
    """Minimal batched-indexing carrier; sampler tests never collate its rows."""

    data_dir_path = "unused-map"

    def __init__(self):
        pass

    def __len__(self):
        return 24

    def __getitem__(self, index):
        return index


def _payload(*, sequential_sampling: bool, strategy: str = "auto"):
    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig

    return SimpleNamespace(
        tf_training_config=SimpleNamespace(
            sequential_sampling=sequential_sampling,
        ),
        pt_data_config=DataConfig(),
        pt_model_config=ModelConfig(),
        pt_training_config=TrainingConfig(
            batch_size=4,
            device="cpu",
            framework="Lightning",
        ),
        execution_config=PyTorchExecutionConfig(
            accelerator="cpu",
            devices=1,
            strategy=strategy,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        ),
    )


def _sampler_orders(loader, epochs=2):
    return [list(iter(loader.sampler)) for _ in range(epochs)]


def _build_mmap(dataset, *, sequential_sampling, strategy="auto", **kwargs):
    from ptycho_torch.workflows.components import _build_lightning_dataloaders

    payload = _payload(
        sequential_sampling=sequential_sampling,
        strategy=strategy,
    )
    return _build_lightning_dataloaders(
        dataset,
        dataset,
        payload.tf_training_config,
        payload=payload,
        **kwargs,
    )


def _inline_container(length=24, N=64):
    sample_ids = (
        torch.arange(length, dtype=torch.float32)
        .view(length, 1, 1, 1)
        .expand(length, N, N, 1)
        .clone()
    )
    return {
        "X": sample_ids,
        "coords_relative": torch.zeros(length, 1, 1, 2),
        "rms_scaling_constant": torch.ones(length, 1, 1, 1),
        "physics_scaling_constant": torch.ones(length, 1, 1, 1),
        "probe": torch.ones(N, N, dtype=torch.complex64),
        "scaling_constant": torch.ones(1),
    }


def _inline_orders(loader, epochs=2):
    orders = []
    for _ in range(epochs):
        order = []
        for batch in loader:
            order.extend(
                int(value)
                for value in batch[0]["images"][:, 0, 0, 0]
            )
        orders.append(order)
    return orders


def _inline_epoch_order(loader, epoch):
    loader.sampler.set_epoch(epoch)
    return _inline_orders(loader, epochs=1)[0]


def test_in_memory_shuffle_owns_a_seeded_epoch_varying_stream():
    from ptycho.config.config import ModelConfig, TrainingConfig
    from ptycho_torch.workflows.components import _build_lightning_dataloaders

    def build(seed, *, sequential_sampling=False):
        config = TrainingConfig(
            model=ModelConfig(N=64, gridsize=1),
            batch_size=4,
            sequential_sampling=sequential_sampling,
            subsample_seed=seed,
        )
        return _build_lightning_dataloaders(
            _inline_container(),
            _inline_container(),
            config,
            payload=None,
            torch_training_seed=seed,
        )

    first_train, first_val = build(73)
    second_train, second_val = build(73)
    other_train, _ = build(74)

    assert isinstance(first_train.sampler, RandomSampler)
    assert isinstance(first_val.sampler, SequentialSampler)
    first_orders = _inline_orders(first_train)
    assert first_orders == _inline_orders(second_train)
    assert first_orders[0] != first_orders[1]
    assert first_orders[0] != _inline_orders(other_train, epochs=1)[0]
    expected = list(range(24))
    assert _inline_orders(first_val, epochs=1)[0] == expected
    assert _inline_orders(second_val, epochs=1)[0] == expected

    sequential_train, sequential_val = build(73, sequential_sampling=True)
    assert _inline_orders(sequential_train) == [expected, expected]
    assert _inline_orders(sequential_val) == [expected, expected]


def test_single_device_mmap_shuffle_is_seeded_and_epoch_varying():
    dataset = _IndexableDataset()
    first_train, first_val = _build_mmap(
        dataset,
        sequential_sampling=False,
        torch_training_seed=73,
    )
    second_train, second_val = _build_mmap(
        dataset,
        sequential_sampling=False,
        torch_training_seed=73,
    )

    assert isinstance(first_train.sampler, RandomSampler)
    assert isinstance(first_val.sampler, SequentialSampler)
    assert isinstance(second_val.sampler, SequentialSampler)
    first_orders = _sampler_orders(first_train)
    second_orders = _sampler_orders(second_train)
    assert first_orders == second_orders
    assert first_orders[0] != first_orders[1]


def test_single_device_mmap_honors_explicit_sequential_sampling():
    train, val = _build_mmap(
        _IndexableDataset(),
        sequential_sampling=True,
        torch_training_seed=73,
    )

    assert isinstance(train.sampler, SequentialSampler)
    assert isinstance(val.sampler, SequentialSampler)


def test_historical_batch_order_recipe_matches_in_memory_and_mmap_rails():
    from ptycho.config.config import ModelConfig, TrainingConfig
    from ptycho_torch.batch_order import (
        JULY2026_BATCH_ORDER_RECIPE,
        July2026BatchOrderSampler,
    )
    from ptycho_torch.workflows.components import _build_lightning_dataloaders

    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=1),
        batch_size=4,
        sequential_sampling=False,
        subsample_seed=3,
    )
    in_memory, _ = _build_lightning_dataloaders(
        _inline_container(),
        _inline_container(),
        config,
        payload=None,
        torch_training_seed=3,
        batch_order_recipe=JULY2026_BATCH_ORDER_RECIPE,
    )
    mmap, _ = _build_mmap(
        _IndexableDataset(),
        sequential_sampling=False,
        torch_training_seed=3,
        batch_order_recipe=JULY2026_BATCH_ORDER_RECIPE,
    )

    assert isinstance(in_memory.sampler, July2026BatchOrderSampler)
    assert isinstance(mmap.sampler, July2026BatchOrderSampler)
    for epoch in range(3):
        expected = list(iter(mmap.sampler.set_epoch(epoch) or mmap.sampler))
        assert _inline_epoch_order(in_memory, epoch) == expected


def test_historical_batch_order_recipe_fails_closed_for_ddp():
    from ptycho_torch.batch_order import JULY2026_BATCH_ORDER_RECIPE

    with pytest.raises(ValueError, match="single-device"):
        _build_mmap(
            _IndexableDataset(),
            sequential_sampling=False,
            strategy="ddp",
            torch_training_seed=3,
            batch_order_recipe=JULY2026_BATCH_ORDER_RECIPE,
        )


def test_historical_batch_order_recipe_requires_validation_data():
    from ptycho_torch.batch_order import JULY2026_BATCH_ORDER_RECIPE
    from ptycho_torch.workflows.components import _build_lightning_dataloaders

    payload = _payload(sequential_sampling=False)

    with pytest.raises(ValueError, match="validation loader"):
        _build_lightning_dataloaders(
            _IndexableDataset(),
            None,
            payload.tf_training_config,
            payload=payload,
            torch_training_seed=3,
            batch_order_recipe=JULY2026_BATCH_ORDER_RECIPE,
        )


def test_ddp_mmap_datamodule_uses_the_same_sampling_policy_and_seed():
    from ptycho_torch.train_utils import PrebuiltPtychoDataModule

    dataset = _IndexableDataset()
    module = _build_mmap(
        dataset,
        sequential_sampling=False,
        strategy="ddp",
        torch_training_seed=91,
    )
    assert isinstance(module, PrebuiltPtychoDataModule)
    module.train_dataset = dataset
    module.val_dataset = dataset

    first_orders = _sampler_orders(module.train_dataloader())
    second_orders = _sampler_orders(module.train_dataloader())
    assert first_orders == second_orders
    assert first_orders[0] != first_orders[1]
    assert isinstance(module.val_dataloader().sampler, SequentialSampler)

    sequential = _build_mmap(
        dataset,
        sequential_sampling=True,
        strategy="ddp",
        torch_training_seed=91,
    )
    sequential.train_dataset = dataset
    sequential.val_dataset = dataset
    assert isinstance(sequential.train_dataloader().sampler, SequentialSampler)
    assert isinstance(sequential.val_dataloader().sampler, SequentialSampler)


def test_ddp_runtime_provenance_reads_the_prebuilt_loader_settings():
    from ptycho_torch.workflows import components

    payload = _payload(sequential_sampling=False, strategy="ddp")
    module = _build_mmap(
        _IndexableDataset(),
        sequential_sampling=False,
        strategy="ddp",
    )

    assert components._effective_dataloader_settings(
        module,
        None,
        payload.execution_config,
    ) == {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
    }


def test_direct_prebuilt_spawn_loader_disables_nested_workers():
    from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
    from ptycho_torch.train_utils import PrebuiltPtychoDataModule

    module = PrebuiltPtychoDataModule(
        "unused-map",
        ModelConfig(),
        DataConfig(),
        TrainingConfig(strategy="ddp_spawn", num_workers=3),
    )

    assert module._loader_settings()["num_workers"] == 0


@pytest.mark.parametrize(
    ("accelerator", "backend"),
    [("cpu", None), ("cuda", "nccl")],
)
def test_spawn_strategy_preserves_controls_for_accelerator(accelerator, backend):
    from lightning.pytorch.strategies import DDPStrategy
    from ptycho_torch.train_utils import get_training_strategy

    strategy = get_training_strategy(
        "ddp_spawn",
        2,
        accelerator=accelerator,
    )

    assert isinstance(strategy, DDPStrategy)
    assert strategy._start_method == "spawn"
    assert strategy._ddp_kwargs["find_unused_parameters"] is False
    assert strategy._process_group_backend == backend


def test_prebuilt_datamodule_owns_validation_split_and_training_drop_last(
    monkeypatch,
):
    from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
    from ptycho_torch.train_utils import PrebuiltPtychoDataModule

    dataset = _IndexableDataset()
    dataset.ci_contract_active = False
    monkeypatch.setattr(
        PtychoDataset,
        "from_existing_map",
        classmethod(lambda cls, *_args, **_kwargs: dataset),
    )

    module = PrebuiltPtychoDataModule(
        "unused-map",
        ModelConfig(),
        DataConfig(),
        TrainingConfig(batch_size=5, num_workers=0),
        validation_fraction=0.25,
        validation_seed=7,
        drop_last_training=True,
    )
    module.setup("fit")
    expected_train, expected_val = torch.utils.data.random_split(
        dataset,
        [18, 6],
        generator=torch.Generator().manual_seed(7),
    )

    assert module.train_dataset.indices == expected_train.indices
    assert module.val_dataset.indices == expected_val.indices
    assert module.train_dataloader().drop_last is True
    assert module.val_dataloader().drop_last is False


@pytest.mark.parametrize("validation_fraction", [-0.1, 0.0, 1.0, 1.1])
def test_prebuilt_datamodule_rejects_invalid_validation_fraction(
    validation_fraction,
):
    from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
    from ptycho_torch.train_utils import PrebuiltPtychoDataModule

    with pytest.raises(ValueError, match="validation_fraction"):
        PrebuiltPtychoDataModule(
            "unused-map",
            ModelConfig(),
            DataConfig(),
            TrainingConfig(),
            validation_fraction=validation_fraction,
        )


def test_build_prebuilt_datamodule_defers_one_map_write_to_prepare_data(
    monkeypatch,
    tmp_path,
):
    from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
    from ptycho_torch.train_utils import build_prebuilt_ptycho_datamodule

    calls = []

    class StubDataset:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr("ptycho_torch.train_utils.PtychoDataset", StubDataset)
    module = build_prebuilt_ptycho_datamodule(
        tmp_path / "source",
        tmp_path / "map" / "memmap",
        ModelConfig(),
        DataConfig(),
        TrainingConfig(orchestrator="Lightning"),
        validation_fraction=0.05,
        validation_seed=7,
        drop_last_training=True,
    )

    assert calls == []
    module.prepare_data()
    assert len(calls) == 1
    assert calls[0][1]["remake_map"] is True
    assert calls[0][1]["data_dir"] == str(tmp_path / "map" / "memmap")
    assert module.map_path == tmp_path / "map" / "memmap"
    assert module.validation_fraction == 0.05
    assert module.validation_seed == 7
    assert module.drop_last_training is True
    assert module.prepare_data_per_node is False


@pytest.mark.parametrize(
    "model_config",
    [
        pytest.param(
            SimpleNamespace(mode="Supervised", rect_s1s2_init="ones"),
            id="supervised",
        ),
        pytest.param(
            SimpleNamespace(mode="Unsupervised", rect_s1s2_init="dose_closure"),
            id="dose-closure",
        ),
    ],
)
def test_prebuilt_mmap_materialization_rejects_pre_fit_data_consumers(
    model_config,
    tmp_path,
):
    from ptycho_torch.config_params import DataConfig, TrainingConfig
    from ptycho_torch.train_utils import build_prebuilt_ptycho_datamodule

    with pytest.raises(ValueError):
        build_prebuilt_ptycho_datamodule(
            tmp_path / "source",
            tmp_path / "map" / "memmap",
            model_config,
            DataConfig(),
            TrainingConfig(orchestrator="Lightning"),
        )


def test_prebuilt_mmap_materialization_requires_lightning_orchestrator(tmp_path):
    from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
    from ptycho_torch.train_utils import build_prebuilt_ptycho_datamodule

    with pytest.raises(ValueError, match="orchestrator='Lightning'"):
        build_prebuilt_ptycho_datamodule(
            tmp_path / "source",
            tmp_path / "map" / "memmap",
            ModelConfig(),
            DataConfig(),
            TrainingConfig(orchestrator="Mlflow"),
        )

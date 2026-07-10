import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ptycho_torch.helper as hh
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.dataloader import PtychoDataset
from ptycho_torch.scaling_contract import derive_ci_experiment_statistics


N_PIX = 16
CI_FIELDS = {
    "measured_intensity",
    "rms_input_scale",
    "mean_measured_intensity",
    "probe_training",
    "probe_physical",
    "probe_normalization",
}


def _count_intensity_arrays(n_images=10):
    intensity = np.arange(
        1,
        n_images * N_PIX * N_PIX + 1,
        dtype=np.float32,
    ).reshape(n_images, N_PIX, N_PIX)
    xcoords = np.linspace(0.0, 9.0, n_images, dtype=np.float64)
    ycoords = np.linspace(1.0, 10.0, n_images, dtype=np.float64)
    grid = np.arange(1, N_PIX * N_PIX + 1, dtype=np.float32).reshape(
        N_PIX,
        N_PIX,
    )
    probe = np.stack(
        [grid + 1j * (grid + 3), (grid + 7) + 1j * (2 * grid + 1)],
    ).astype(np.complex64)
    obj = np.ones((N_PIX, N_PIX), dtype=np.complex64)
    return intensity, xcoords, ycoords, probe, obj


def _write_npz(path, payload):
    intensity, xcoords, ycoords, probe, obj = payload
    np.savez(
        path,
        diff3d=intensity,
        xcoords=xcoords,
        ycoords=ycoords,
        probeGuess=probe,
        objectGuess=obj,
    )


def _ci_configs(**data_overrides):
    data_config = DataConfig(
        N=N_PIX,
        C=1,
        grid_size=(1, 1),
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
        normalize="Batch",
        probe_normalize=True,
        **data_overrides,
    )
    model_config = ModelConfig(
        mode="Unsupervised",
        C_model=1,
        C_forward=1,
        object_big=False,
        physics_forward_mode="rectangular_scaled",
        cnn_output_mode="real_imag",
    )
    training_config = TrainingConfig(
        batch_size=4,
        torch_loss_mode="poisson",
        orchestrator="Mlflow",
        num_workers=0,
    )
    return data_config, model_config, training_config


def _build_file_dataset(tmp_path, payload, data_config, model_config, training_config):
    ptycho_dir = tmp_path / "npz"
    ptycho_dir.mkdir(parents=True)
    _write_npz(ptycho_dir / "counts.npz", payload)
    return PtychoDataset(
        ptycho_dir=str(ptycho_dir),
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        data_dir=str(tmp_path / "memmap"),
        remake_map=True,
    )


def _build_memory_dataset(payload, data_config, model_config):
    intensity, xcoords, ycoords, probe, _ = payload
    positions = np.stack([ycoords, xcoords], axis=1)
    return PtychoDataset.from_np(
        intensity,
        probe,
        positions,
        model_config,
        data_config,
    )


def _ci_lightning_configs():
    data_config, model_config, training_config = _ci_configs()
    return replace(data_config, N=64), model_config, training_config


def test_ci_mmap_and_from_np_emit_identical_physical_multimode_batches(tmp_path):
    payload = _count_intensity_arrays()
    data_config, model_config, training_config = _ci_configs()
    file_dataset = _build_file_dataset(
        tmp_path,
        payload,
        data_config,
        model_config,
        training_config,
    )
    memory_dataset = _build_memory_dataset(payload, data_config, model_config)
    indices = torch.arange(len(file_dataset))

    file_batch = file_dataset[indices]
    memory_batch = memory_dataset[indices]
    file_td, file_probe_alias, file_q_alias = file_batch
    memory_td, memory_probe_alias, memory_q_alias = memory_batch

    assert CI_FIELDS <= set(file_td.keys())
    assert CI_FIELDS <= set(memory_td.keys())
    assert "physics_scaling_constant" not in file_td.keys()
    assert "physics_scaling_constant" not in memory_td.keys()
    for dataset in (file_dataset, memory_dataset):
        assert "rms_input_scale" not in dataset.mmap_ptycho.keys()
        assert "mean_measured_intensity" not in dataset.mmap_ptycho.keys()
    for field in CI_FIELDS | {"images", "experiment_id", "nn_indices"}:
        torch.testing.assert_close(file_td[field], memory_td[field])

    physical_probe = torch.from_numpy(payload[3]).to(torch.complex64)
    _, expected_q = hh.normalize_probe_like_tf(
        payload[3],
        probe_scale=data_config.probe_scale,
        probe_mask=model_config.probe_mask,
        probe_mask_tensor=model_config.probe_mask_tensor,
        probe_mask_sigma=model_config.probe_mask_sigma,
        probe_mask_diameter=model_config.probe_mask_diameter,
    )
    expected_physical = physical_probe[None, None].expand(
        len(file_dataset),
        1,
        2,
        N_PIX,
        N_PIX,
    )

    assert file_td["probe_physical"].shape == (
        len(file_dataset),
        1,
        2,
        N_PIX,
        N_PIX,
    )
    assert file_td["probe_training"].shape == expected_physical.shape
    assert file_td["probe_normalization"].shape == (
        len(file_dataset),
        1,
        1,
        1,
        1,
    )
    assert file_probe_alias.shape == expected_physical.shape
    assert file_q_alias.shape == (len(file_dataset), 1, 1, 1)
    torch.testing.assert_close(file_td["probe_physical"], expected_physical)
    torch.testing.assert_close(
        file_td["probe_training"],
        file_td["probe_normalization"] * file_td["probe_physical"],
    )
    torch.testing.assert_close(
        file_td["probe_normalization"],
        torch.full_like(file_td["probe_normalization"], expected_q),
    )
    torch.testing.assert_close(file_probe_alias, file_td["probe_training"])
    torch.testing.assert_close(
        file_q_alias,
        file_td["probe_normalization"].squeeze(-1),
    )
    torch.testing.assert_close(memory_probe_alias, file_probe_alias)
    torch.testing.assert_close(memory_q_alias, file_q_alias)
    torch.testing.assert_close(
        file_dataset.data_dict["probes_physical"][0],
        physical_probe,
    )

    expected_statistics = derive_ci_experiment_statistics(
        torch.from_numpy(payload[0])[:, None],
        N_PIX,
    )
    torch.testing.assert_close(
        file_td["rms_input_scale"],
        expected_statistics.rms_input_scale.expand(len(file_dataset), 1, 1, 1),
    )
    torch.testing.assert_close(
        file_td["mean_measured_intensity"],
        expected_statistics.mean_measured_intensity.expand(
            len(file_dataset), 1, 1, 1
        ),
    )


def test_ci_named_probe_normalization_is_five_dimensional_for_all_indexing(tmp_path):
    payload = _count_intensity_arrays()
    data_config, model_config, training_config = _ci_configs()
    dataset = _build_file_dataset(
        tmp_path,
        payload,
        data_config,
        model_config,
        training_config,
    )

    scalar_td, _, scalar_alias = dataset[0]
    batch_td, _, batch_alias = dataset[torch.tensor([0, 1, 2])]

    assert scalar_td["probe_normalization"].shape == (1, 1, 1, 1, 1)
    assert batch_td["probe_normalization"].shape == (3, 1, 1, 1, 1)
    assert scalar_alias.shape == (1, 1, 1)
    assert batch_alias.shape == (3, 1, 1, 1)


@pytest.mark.parametrize("source", ["mmap", "from_np"])
def test_explicit_legacy_loader_fields_and_tuple_aliases_are_byte_identical(
    tmp_path,
    source,
):
    payload = _count_intensity_arrays()
    baseline_data = DataConfig(
        N=N_PIX,
        C=1,
        grid_size=(1, 1),
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
        normalize="Batch",
    )
    legacy_data = DataConfig(
        N=N_PIX,
        C=1,
        grid_size=(1, 1),
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
        normalize="Batch",
        scale_contract_version="legacy_v1",
        measurement_domain="normalized_amplitude",
    )
    baseline_model = ModelConfig(C_model=1, C_forward=1, object_big=False)
    legacy_model = ModelConfig(
        mode="Unsupervised",
        C_model=1,
        C_forward=1,
        object_big=False,
        physics_forward_mode="rectangular_scaled",
        cnn_output_mode="real_imag",
    )
    training_config = TrainingConfig(batch_size=4, orchestrator="Mlflow")

    if source == "mmap":
        baseline = _build_file_dataset(
            tmp_path / "baseline",
            payload,
            baseline_data,
            baseline_model,
            training_config,
        )
        explicit = _build_file_dataset(
            tmp_path / "explicit",
            payload,
            legacy_data,
            legacy_model,
            training_config,
        )
    else:
        baseline = _build_memory_dataset(payload, baseline_data, baseline_model)
        explicit = _build_memory_dataset(payload, legacy_data, legacy_model)

    indices = torch.arange(len(baseline))
    baseline_td, baseline_probe, baseline_q = baseline[indices]
    explicit_td, explicit_probe, explicit_q = explicit[indices]
    for field in (
        "images",
        "rms_scaling_constant",
        "physics_scaling_constant",
    ):
        assert torch.equal(baseline_td[field], explicit_td[field])
    assert torch.equal(baseline_probe, explicit_probe)
    assert torch.equal(baseline_q, explicit_q)


def test_data_module_freezes_ci_statistics_from_final_training_subset(tmp_path):
    from ptycho_torch.train_utils import PtychoDataModule

    payload = list(_count_intensity_arrays(n_images=10))
    split_generator = torch.Generator().manual_seed(19)
    _, expected_validation = torch.utils.data.random_split(
        range(10),
        [8, 2],
        generator=split_generator,
    )
    payload[0][:] = 1.0
    payload[0][expected_validation.indices] = 1000.0
    payload = tuple(payload)
    ptycho_dir = tmp_path / "npz"
    ptycho_dir.mkdir()
    _write_npz(ptycho_dir / "counts.npz", payload)
    data_config, model_config, training_config = _ci_configs()
    module = PtychoDataModule(
        str(ptycho_dir),
        model_config,
        data_config,
        training_config,
        initial_remake_map=True,
        val_split=0.2,
        val_seed=19,
        memory_map_dir=str(tmp_path / "memmap"),
    )

    module.setup("fit")

    full_dataset = module.train_dataset.dataset
    train_indices = torch.as_tensor(module.train_dataset.indices)
    train_images = torch.as_tensor(full_dataset.mmap_ptycho["images"])[train_indices]
    expected = derive_ci_experiment_statistics(train_images, N_PIX)
    full = derive_ci_experiment_statistics(
        torch.as_tensor(full_dataset.mmap_ptycho["images"]),
        N_PIX,
    )
    assert not torch.equal(
        expected.mean_measured_intensity,
        full.mean_measured_intensity,
    )
    torch.testing.assert_close(
        module.ci_statistics["rms_input_scale"],
        expected.rms_input_scale.reshape(1),
    )
    torch.testing.assert_close(
        module.ci_statistics["mean_measured_intensity"],
        expected.mean_measured_intensity.reshape(1),
    )

    train_td, _, _ = full_dataset[train_indices]
    val_td, _, _ = full_dataset[torch.as_tensor(module.val_dataset.indices)]
    for td in (train_td, val_td):
        assert torch.all(td["rms_input_scale"] == expected.rms_input_scale)
        assert torch.all(
            td["mean_measured_intensity"] == expected.mean_measured_intensity
        )


def test_ci_statistics_read_finalized_indices_in_bounded_chunks(tmp_path, monkeypatch):
    import ptycho_torch.dataloader as dataloader_module

    payload = _count_intensity_arrays(n_images=10)
    data_config, model_config, training_config = _ci_configs()
    dataset = _build_file_dataset(
        tmp_path,
        payload,
        data_config,
        model_config,
        training_config,
    )
    measured_storage = dataset.mmap_ptycho["measured_intensity"]
    experiment_storage = dataset.mmap_ptycho["experiment_id"]
    requests = []
    chunk_bound = 3

    class IndexedStorageSpy:
        def __init__(self, storage):
            self.storage = storage

        def __getitem__(self, index):
            if isinstance(index, torch.Tensor):
                request_size = index.numel()
            elif isinstance(index, (list, tuple)):
                request_size = len(index)
            else:
                raise AssertionError(f"unbounded indexing request: {index!r}")
            requests.append(request_size)
            assert request_size <= chunk_bound
            return self.storage[index]

    dataset.mmap_ptycho = {
        "measured_intensity": IndexedStorageSpy(measured_storage),
        "experiment_id": IndexedStorageSpy(experiment_storage),
    }
    monkeypatch.setattr(
        dataloader_module,
        "_CI_STATISTICS_CHUNK_SIZE",
        chunk_bound,
        raising=False,
    )
    finalized_indices = [9, 1, 7, 3, 5, 0, 8]
    expected = derive_ci_experiment_statistics(
        torch.as_tensor(measured_storage)[torch.tensor(finalized_indices)],
        N_PIX,
    )

    statistics = dataset.set_ci_statistics_from_indices(finalized_indices)

    assert requests
    assert max(requests) <= chunk_bound
    torch.testing.assert_close(
        statistics["rms_input_scale"],
        expected.rms_input_scale.reshape(1),
    )
    torch.testing.assert_close(
        statistics["mean_measured_intensity"],
        expected.mean_measured_intensity.reshape(1),
    )


def test_ci_mmap_manifest_allows_compatible_reuse(tmp_path):
    payload = _count_intensity_arrays()
    data_config, model_config, training_config = _ci_configs()
    dataset = _build_file_dataset(
        tmp_path,
        payload,
        data_config,
        model_config,
        training_config,
    )

    manifest = json.loads(dataset.manifest_path.read_text())
    assert manifest["schema_version"] >= 1
    assert manifest["scale_contract_version"] == "ci_intensity_v2"
    assert manifest["measurement_domain"] == "count_intensity"
    assert "measured_intensity" in manifest["required_fields"]
    assert "rms_input_scale" not in manifest["required_fields"]

    reused = PtychoDataset(
        ptycho_dir=str(tmp_path / "npz"),
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        data_dir=str(tmp_path / "memmap"),
        remake_map=False,
    )

    assert len(reused) == len(dataset)
    for name, expected in dataset.get_ci_statistics().items():
        torch.testing.assert_close(reused.get_ci_statistics()[name], expected)


@pytest.mark.parametrize("incompatibility", ["missing", "pre_task3", "opposite"])
def test_ci_mmap_reuse_rejects_incompatible_manifest(tmp_path, incompatibility):
    payload = _count_intensity_arrays()
    data_config, model_config, training_config = _ci_configs()
    dataset = _build_file_dataset(
        tmp_path,
        payload,
        data_config,
        model_config,
        training_config,
    )

    reuse_data_config = data_config
    reuse_model_config = model_config
    if incompatibility == "missing":
        dataset.manifest_path.unlink()
    elif incompatibility == "pre_task3":
        manifest = json.loads(dataset.manifest_path.read_text())
        manifest["schema_version"] = 0
        dataset.manifest_path.write_text(json.dumps(manifest))
    else:
        reuse_data_config = replace(
            data_config,
            scale_contract_version="legacy_v1",
            measurement_domain="normalized_amplitude",
        )

    with pytest.raises(ValueError, match=r"[Rr]ebuild.*remake_map=True"):
        PtychoDataset(
            ptycho_dir=str(tmp_path / "npz"),
            model_config=reuse_model_config,
            data_config=reuse_data_config,
            training_config=training_config,
            data_dir=str(tmp_path / "memmap"),
            remake_map=False,
        )


def _assert_finalized_module_statistics(module, provisional_statistics):
    from ptycho_torch.lightning_utils import CIStatisticsCallback

    module.setup("fit")
    dataset = module.train_dataset.dataset
    train_indices = torch.as_tensor(module.train_dataset.indices)
    validation_indices = torch.as_tensor(module.val_dataset.indices)
    train_images = torch.as_tensor(dataset.mmap_ptycho["images"])[train_indices]
    expected = derive_ci_experiment_statistics(train_images, N_PIX)

    assert not torch.equal(
        provisional_statistics["mean_measured_intensity"],
        expected.mean_measured_intensity.reshape(1),
    )
    torch.testing.assert_close(
        module.ci_statistics["rms_input_scale"],
        expected.rms_input_scale.reshape(1),
    )
    torch.testing.assert_close(
        module.ci_statistics["mean_measured_intensity"],
        expected.mean_measured_intensity.reshape(1),
    )

    for indices in (train_indices, validation_indices):
        tensor_dict, _, _ = dataset[indices]
        assert torch.all(
            tensor_dict["rms_input_scale"] == expected.rms_input_scale
        )
        assert torch.all(
            tensor_dict["mean_measured_intensity"]
            == expected.mean_measured_intensity
        )

    registered = {}

    class Model:
        model_config = dataset.model_config
        data_config = dataset.data_config

        def register_ci_statistics(self, statistics):
            registered.update(statistics)

    trainer = SimpleNamespace(
        datamodule=module,
        logger=None,
        is_global_zero=True,
    )
    CIStatisticsCallback().on_fit_start(trainer, Model())
    for name, expected_value in module.ci_statistics.items():
        torch.testing.assert_close(registered[name], expected_value)


def _skew_validation_payload(val_split, val_seed):
    payload = list(_count_intensity_arrays(n_images=10))
    train_size = 10 - int(val_split * 10)
    train_subset, validation_subset = torch.utils.data.random_split(
        range(10),
        [train_size, 10 - train_size],
        generator=torch.Generator().manual_seed(val_seed),
    )
    payload[0][:] = 2.0
    payload[0][validation_subset.indices] = 2000.0
    return tuple(payload), train_subset.indices, validation_subset.indices


def test_in_memory_data_module_replaces_provisional_ci_statistics_from_train_split():
    from ptycho_torch.train_utils import InMemoryPtychoDataModule

    val_split = 0.2
    val_seed = 17
    payload, _, _ = _skew_validation_payload(val_split, val_seed)
    data_config, model_config, training_config = _ci_configs()
    dataset = _build_memory_dataset(payload, data_config, model_config)
    provisional_statistics = dataset.get_ci_statistics()
    module = InMemoryPtychoDataModule(
        dataset,
        training_config,
        val_split=val_split,
        val_seed=val_seed,
    )

    _assert_finalized_module_statistics(module, provisional_statistics)


def test_prebuilt_data_module_replaces_provisional_ci_statistics_from_train_split(
    tmp_path,
):
    from ptycho_torch.train_utils import PrebuiltPtychoDataModule

    val_split = 0.1
    val_seed = 42
    payload, _, _ = _skew_validation_payload(val_split, val_seed)
    data_config, model_config, training_config = _ci_configs()
    source_dataset = _build_file_dataset(
        tmp_path,
        payload,
        data_config,
        model_config,
        training_config,
    )
    provisional_statistics = source_dataset.get_ci_statistics()
    module = PrebuiltPtychoDataModule(
        str(tmp_path / "memmap"),
        model_config,
        data_config,
        training_config,
    )

    _assert_finalized_module_statistics(module, provisional_statistics)


def test_ci_statistics_callback_registers_before_batches_and_logs_metadata(tmp_path):
    from ptycho_torch.lightning_utils import CIStatisticsCallback, MetadataLogger

    statistics = {
        "rms_input_scale": torch.tensor([0.25, 0.5]),
        "mean_measured_intensity": torch.tensor([10.0, 20.0]),
    }
    registered = {}

    class Model:
        def register_ci_statistics(self, value):
            registered.update(value)

        def get_ci_statistics(self):
            return registered

    class Logger:
        def __init__(self):
            self.payloads = []

        def log_hyperparams(self, payload):
            self.payloads.append(payload)

    logger = Logger()
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(ci_statistics=statistics),
        logger=logger,
        callbacks=[],
        current_epoch=0,
        is_global_zero=True,
    )
    model = Model()

    CIStatisticsCallback().on_fit_start(trainer, model)

    assert registered == statistics
    assert logger.payloads[-1]["ci_statistics"] == {
        "rms_input_scale": [0.25, 0.5],
        "mean_measured_intensity": [10.0, 20.0],
    }

    metadata_logger = MetadataLogger(run_dir=str(tmp_path))
    metadata_logger.on_train_start(trainer, model)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["ci_statistics"] == logger.payloads[-1]["ci_statistics"]


def test_ci_statistics_checkpoint_round_trip():
    from ptycho_torch.model import PtychoPINN_Lightning

    data_config, model_config, training_config = _ci_lightning_configs()
    statistics = {
        "rms_input_scale": torch.tensor([0.25, 0.5]),
        "mean_measured_intensity": torch.tensor([10.0, 20.0]),
    }
    source = PtychoPINN_Lightning(
        model_config,
        data_config,
        training_config,
        InferenceConfig(),
    )
    source.register_ci_statistics(statistics)
    checkpoint = {}

    source.on_save_checkpoint(checkpoint)

    restored = PtychoPINN_Lightning(
        model_config,
        data_config,
        training_config,
        InferenceConfig(),
    )
    restored.on_load_checkpoint(checkpoint)
    restored_statistics = restored.get_ci_statistics()
    for name, expected in statistics.items():
        torch.testing.assert_close(restored_statistics[name], expected)


def test_ci_statistics_actual_lightning_checkpoint_round_trip(tmp_path):
    import lightning as L
    from ptycho_torch.model import PtychoPINN_Lightning

    data_config, model_config, training_config = _ci_lightning_configs()
    statistics = {
        "rms_input_scale": torch.tensor([0.25, 0.5]),
        "mean_measured_intensity": torch.tensor([10.0, 20.0]),
    }
    source = PtychoPINN_Lightning(
        model_config,
        data_config,
        training_config,
        InferenceConfig(),
    )
    source.register_ci_statistics(statistics)
    checkpoint_path = tmp_path / "ci-statistics.ckpt"
    trainer = L.Trainer(
        max_epochs=0,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.strategy._lightning_module = source

    trainer.save_checkpoint(checkpoint_path)
    restored = PtychoPINN_Lightning.load_from_checkpoint(checkpoint_path)

    for name, expected in statistics.items():
        torch.testing.assert_close(restored.get_ci_statistics()[name], expected)


def test_training_entry_points_construct_rank_safe_ci_statistics_callbacks():
    import ptycho_torch.train as train
    import ptycho_torch.train_lightning_only as train_lightning_only
    from ptycho_torch.lightning_utils import CIStatisticsCallback

    tracked = train._build_ci_statistics_callback(disable_mlflow=False)
    untracked = train._build_ci_statistics_callback(disable_mlflow=True)
    lightning_only = train_lightning_only._build_ci_statistics_callback()

    assert isinstance(tracked, CIStatisticsCallback)
    assert tracked.metadata_sink is train._persist_finalized_ci_statistics_to_mlflow
    assert isinstance(untracked, CIStatisticsCallback)
    assert untracked.metadata_sink is None
    assert isinstance(lightning_only, CIStatisticsCallback)
    assert lightning_only.metadata_sink is None


@pytest.mark.parametrize("is_global_zero", [True, False])
def test_ci_statistics_callback_registers_every_rank_and_gates_side_effects(
    is_global_zero,
):
    from ptycho_torch.lightning_utils import CIStatisticsCallback

    statistics = {
        "rms_input_scale": torch.tensor([0.25]),
        "mean_measured_intensity": torch.tensor([2.0]),
    }
    registered = []
    logger_payloads = []
    metadata_payloads = []

    model = SimpleNamespace(
        register_ci_statistics=lambda value: registered.append(value)
    )
    logger = SimpleNamespace(
        log_hyperparams=lambda value: logger_payloads.append(value)
    )
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(ci_statistics=statistics),
        logger=logger,
        is_global_zero=is_global_zero,
    )
    callback = CIStatisticsCallback(metadata_sink=metadata_payloads.append)

    callback.on_fit_start(trainer, model)

    assert registered == [statistics]
    if is_global_zero:
        assert len(logger_payloads) == 1
        assert len(metadata_payloads) == 1
    else:
        assert logger_payloads == []
        assert metadata_payloads == []


def test_mlflow_callback_persists_finalized_statistics_after_data_setup(monkeypatch):
    import ptycho_torch.train as train
    from ptycho_torch.lightning_utils import CIStatisticsCallback

    provisional = {
        "rms_input_scale": torch.tensor([9.0]),
        "mean_measured_intensity": torch.tensor([9000.0]),
    }
    finalized = {
        "rms_input_scale": torch.tensor([0.25]),
        "mean_measured_intensity": torch.tensor([2.0]),
    }
    persisted = []
    logger_payloads = []

    monkeypatch.setattr(train.mlflow, "active_run", lambda: object())
    monkeypatch.setattr(
        train.mlflow,
        "log_dict",
        lambda payload, path: persisted.append((payload, path)),
    )

    class Model:
        def __init__(self):
            self.statistics = provisional

        def register_ci_statistics(self, statistics):
            self.statistics = statistics

    class Logger:
        def log_hyperparams(self, payload):
            logger_payloads.append(payload)

    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(ci_statistics=finalized),
        logger=Logger(),
        is_global_zero=True,
    )
    model = Model()
    callback = CIStatisticsCallback(
        metadata_sink=train._persist_finalized_ci_statistics_to_mlflow
    )

    callback.on_fit_start(trainer, model)

    expected = {
        "rms_input_scale": [0.25],
        "mean_measured_intensity": [2.0],
    }
    assert model.statistics is finalized
    assert logger_payloads == [{"ci_statistics": expected}]
    assert persisted == [(expected, "ci_statistics.json")]


def test_ci_compute_loss_does_not_require_physics_scaling_constant():
    from ptycho_torch.model import PtychoPINN_Lightning

    data_config, model_config, training_config = _ci_lightning_configs()
    module = PtychoPINN_Lightning(
        model_config,
        data_config,
        training_config,
        InferenceConfig(),
    )

    def fake_forward(
        self,
        x,
        positions,
        probe,
        input_scale_factor,
        output_scale_factor,
        experiment_ids=None,
    ):
        return x.clone(), x.sqrt(), torch.zeros_like(x)

    module.forward = fake_forward.__get__(module, PtychoPINN_Lightning)
    batch_size = 2
    measured = torch.ones(batch_size, 1, N_PIX, N_PIX)
    batch = (
        {
            "images": measured,
            "measured_intensity": measured,
            "coords_relative": torch.zeros(batch_size, 1, 1, 2),
            "rms_input_scale": torch.ones(batch_size, 1, 1, 1),
            "mean_measured_intensity": torch.ones(batch_size, 1, 1, 1),
            "experiment_id": torch.zeros(batch_size, dtype=torch.int32),
        },
        torch.ones(batch_size, 1, 1, N_PIX, N_PIX, dtype=torch.complex64),
        torch.ones(batch_size, 1, 1, 1),
    )

    loss = module.compute_loss(batch)

    assert torch.isfinite(loss)

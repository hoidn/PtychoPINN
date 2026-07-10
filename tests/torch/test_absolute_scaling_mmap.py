import inspect
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


def test_data_module_freezes_ci_statistics_from_final_training_subset(
    tmp_path,
    monkeypatch,
):
    import ptycho_torch.dataloader as dataloader_module
    from ptycho_torch.train_utils import PtychoDataModule

    statistic_inputs = []

    def recording_derivation(measured_intensity, N):
        statistic_inputs.append(measured_intensity.detach().clone())
        return derive_ci_experiment_statistics(measured_intensity, N)

    monkeypatch.setattr(
        dataloader_module,
        "derive_ci_experiment_statistics",
        recording_derivation,
    )

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
    assert len(statistic_inputs) == 1
    torch.testing.assert_close(statistic_inputs[0], train_images)
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


def test_both_training_entry_points_install_ci_statistics_callback():
    import ptycho_torch.train as train
    import ptycho_torch.train_lightning_only as train_lightning_only

    assert "CIStatisticsCallback()" in inspect.getsource(train.main)
    assert "CIStatisticsCallback()" in inspect.getsource(train_lightning_only.main)


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

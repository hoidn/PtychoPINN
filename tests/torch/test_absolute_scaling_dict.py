from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ptycho import params
from ptycho.config.config import (
    ModelConfig as TFModelConfig,
    PyTorchExecutionConfig,
    TrainingConfig as TFTrainingConfig,
    update_legacy_dict,
)
from ptycho_torch.config_params import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.model import RectangularScaledDiffraction
from ptycho_torch.scaling_contract import derive_ci_experiment_statistics
from ptycho_torch.workflows import components as torch_components
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, run_torch_training


@pytest.fixture
def params_cfg_snapshot():
    original = params.cfg.copy()
    yield
    params.cfg.clear()
    params.cfg.update(original)


def _amplitude_fixture(
    n_samples: int = 8,
    channels: int = 1,
    modes: int = 1,
    N: int = 8,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    values = torch.linspace(
        0.1,
        1.3,
        n_samples * N * N * channels,
        dtype=torch.float32,
    )
    amplitude = values.reshape(n_samples, N, N, channels)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, N),
        torch.linspace(-1.0, 1.0, N),
        indexing="ij",
    )
    probe_modes = [
        torch.complex(
            (1.0 + 0.2 * mode) * torch.exp(-(xx.square() + yy.square())),
            (0.05 + 0.03 * mode) * xx,
        )
        for mode in range(modes)
    ]
    probe = torch.stack(probe_modes)
    if modes == 1:
        probe = probe[0]
    container = {
        "X": amplitude.clone(),
        "observed_images": amplitude.clone(),
        "coords_nominal": torch.zeros(n_samples, 1, 2, channels),
        "coords_relative": torch.zeros(n_samples, 1, 2, channels),
        "probe": probe.clone(),
    }
    return container, amplitude, probe


def _ci_payload(N: int) -> SimpleNamespace:
    return SimpleNamespace(
        pt_data_config=DataConfig(N=N, grid_size=(1, 1)),
        pt_model_config=ModelConfig(
            mode="Unsupervised",
            physics_forward_mode="rectangular_scaled",
        ),
        pt_training_config=TrainingConfig(torch_loss_mode="poisson"),
    )


def _tf_training_config(tmp_path, N: int, batch_size: int) -> TFTrainingConfig:
    config = TFTrainingConfig(
        model=TFModelConfig(N=N, gridsize=1, object_big=False),
        train_data_file=None,
        output_dir=tmp_path,
        batch_size=batch_size,
        n_groups=8,
        torch_loss_mode="poisson",
    )
    update_legacy_dict(params.cfg, config)
    return config


@pytest.mark.torch
def test_normalized_amplitude_dict_adapter_emits_physical_ci_fields():
    N = 8
    scale = 7.0
    container, amplitude, probe = _amplitude_fixture(
        n_samples=5,
        channels=2,
        modes=2,
        N=N,
    )

    adapter = torch_components.NormalizedAmplitudeCIDictAdapter(
        count_amplitude_scale=scale,
        N=N,
        probe_scale=1.0,
        probe_mask=False,
    )
    statistics = adapter.adapt(container)

    expected_intensity = (scale * amplitude).square()
    expected_probe_physical = scale * probe
    expected_statistics = derive_ci_experiment_statistics(
        expected_intensity.permute(0, 3, 1, 2),
        N,
    )

    assert torch.equal(container["measured_intensity"], expected_intensity)
    assert torch.equal(container["observed_images"], expected_intensity)
    assert torch.equal(container["probe_physical"], expected_probe_physical)
    assert torch.equal(statistics.rms_input_scale, expected_statistics.rms_input_scale)
    assert torch.equal(
        statistics.mean_measured_intensity,
        expected_statistics.mean_measured_intensity,
    )
    assert torch.equal(container["rms_input_scale"], statistics.rms_input_scale)
    assert torch.equal(
        container["mean_measured_intensity"],
        statistics.mean_measured_intensity,
    )
    assert torch.allclose(
        container["probe_training"],
        container["probe_normalization"] * container["probe_physical"],
    )
    assert "physics_scaling_constant" not in container


@pytest.mark.torch
def test_ci_trailing_singleton_probe_is_canonical_before_normalization_and_collation(
    tmp_path,
    params_cfg_snapshot,
):
    N = 8
    batch_size = 2
    count_amplitude_scale = 3.0
    probe_scale = 2.0
    container, _, original_probe_2d = _amplitude_fixture(
        n_samples=4,
        channels=1,
        modes=1,
        N=N,
    )
    container["probe"] = original_probe_2d.unsqueeze(-1)

    torch_components.NormalizedAmplitudeCIDictAdapter(
        count_amplitude_scale=count_amplitude_scale,
        N=N,
        probe_scale=probe_scale,
        probe_mask=False,
    ).adapt(container)

    expected_probe_physical = (
        count_amplitude_scale * original_probe_2d
    ).unsqueeze(0)
    expected_normalization = 1.0 / (
        probe_scale * expected_probe_physical.abs().mean()
    )
    assert container["probe_physical"].shape == (1, N, N)
    assert torch.equal(container["probe_physical"], expected_probe_physical)
    assert container["probe_training"].shape == (1, N, N)
    assert torch.allclose(
        container["probe_training"],
        expected_normalization * expected_probe_physical,
    )
    assert torch.allclose(
        container["probe_normalization"],
        expected_normalization,
    )

    container["X"] = container["measured_intensity"]
    loader, _ = torch_components._build_lightning_dataloaders(
        container,
        None,
        _tf_training_config(tmp_path, N, batch_size),
        payload=_ci_payload(N),
    )
    tensor_dict, probe_training, probe_normalization = next(iter(loader))
    assert probe_training.shape == (batch_size, 1, 1, N, N)
    assert tensor_dict["probe_physical"].shape == (batch_size, 1, 1, N, N)

    object_field = torch.complex(
        torch.full((batch_size, 1, N, N), 0.7),
        torch.full((batch_size, 1, N, N), -0.3),
    )
    prediction = RectangularScaledDiffraction(ModelConfig(num_datasets=1))(
        object_field,
        tensor_dict["measured_intensity"],
        probe_training,
        probe_normalization.reciprocal(),
        tensor_dict["experiment_id"],
    )
    original_physical_exit_wave = (
        count_amplitude_scale
        * original_probe_2d.view(1, 1, 1, N, N)
        * object_field.unsqueeze(2)
    )
    detector_field = torch.fft.fftshift(
        torch.fft.fft2(original_physical_exit_wave, norm="ortho"),
        dim=(-2, -1),
    )
    expected_intensity = detector_field.abs().square().sum(dim=2)
    assert torch.allclose(prediction, expected_intensity, rtol=1e-5, atol=1e-6)


@pytest.mark.torch
@pytest.mark.parametrize(
    "invalid_shape",
    [
        (8, 7),
        (8, 8, 2),
        (2, 8, 8, 1),
    ],
)
def test_ci_dict_adapter_rejects_ambiguous_or_invalid_probe_layouts(invalid_shape):
    container, _, _ = _amplitude_fixture(n_samples=2, N=8)
    container["probe"] = torch.ones(invalid_shape, dtype=torch.complex64)

    with pytest.raises(ValueError, match="probe.*shape"):
        torch_components.NormalizedAmplitudeCIDictAdapter(
            count_amplitude_scale=2.0,
            N=8,
        ).adapt(container)


@pytest.mark.torch
def test_validation_dict_reuses_exact_training_statistics():
    N = 8
    scale = 5.0
    train_container, train_amplitude, _ = _amplitude_fixture(n_samples=6, N=N)
    test_container, _, _ = _amplitude_fixture(n_samples=3, N=N)
    test_amplitude = torch.full_like(test_container["observed_images"], 0.15)
    test_amplitude[0, :2, :2, :] = 3.0
    test_container["X"] = test_amplitude.clone()
    test_container["observed_images"] = test_amplitude.clone()

    train_adapter = torch_components.NormalizedAmplitudeCIDictAdapter(
        count_amplitude_scale=scale,
        N=N,
        probe_scale=1.0,
        probe_mask=False,
    )
    training_statistics = train_adapter.adapt(train_container)

    test_local_statistics = derive_ci_experiment_statistics(
        (scale * test_amplitude).square().permute(0, 3, 1, 2),
        N,
    )
    assert not torch.equal(
        training_statistics.rms_input_scale,
        test_local_statistics.rms_input_scale,
    )

    validation_adapter = torch_components.NormalizedAmplitudeCIDictAdapter(
        count_amplitude_scale=scale,
        N=N,
        statistics=training_statistics,
        probe_scale=1.0,
        probe_mask=False,
    )
    attached_statistics = validation_adapter.adapt(test_container)

    assert torch.equal(
        attached_statistics.rms_input_scale,
        training_statistics.rms_input_scale,
    )
    assert torch.equal(
        attached_statistics.mean_measured_intensity,
        training_statistics.mean_measured_intensity,
    )
    assert torch.equal(
        test_container["rms_input_scale"],
        train_container["rms_input_scale"],
    )
    assert torch.equal(
        test_container["mean_measured_intensity"],
        train_container["mean_measured_intensity"],
    )
    assert not torch.equal(train_amplitude[:3], test_amplitude)


@pytest.mark.torch
def test_grid_lines_ci_runner_freezes_training_statistics_for_validation(
    tmp_path,
    monkeypatch,
):
    N = 32
    n_samples = 4
    train_amplitude = torch.linspace(
        0.05,
        1.2,
        n_samples * N * N,
        dtype=torch.float32,
    ).reshape(n_samples, N, N, 1)
    test_amplitude = torch.full_like(train_amplitude, 0.1)
    test_amplitude[0, :4, :4, :] = 4.0
    probe = torch.complex(
        torch.ones(N, N),
        torch.linspace(-0.2, 0.2, N).view(1, N).expand(N, N),
    )
    train_data = {
        "diffraction": train_amplitude.numpy(),
        "probeGuess": probe.numpy(),
        "coords_nominal": torch.zeros(n_samples, 2).numpy(),
    }
    test_data = {
        "diffraction": test_amplitude.numpy(),
        "probeGuess": probe.numpy(),
        "coords_nominal": torch.zeros(n_samples, 2).numpy(),
    }
    captured = {}

    def fake_train(
        train_container,
        test_container,
        config,
        execution_config=None,
        overrides=None,
    ):
        captured["train"] = train_container
        captured["test"] = test_container
        return {"history": {}, "models": {}}

    monkeypatch.setattr(torch_components, "_train_with_lightning", fake_train)
    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path / "output",
        architecture="cnn",
        physics_forward_mode="rectangular_scaled",
        torch_loss_mode="poisson",
        cnn_output_mode="real_imag",
        N=N,
    )

    run_torch_training(cfg, train_data, test_data)

    train_container = captured["train"]
    test_container = captured["test"]
    assert torch.equal(
        train_container["rms_input_scale"],
        test_container["rms_input_scale"],
    )
    assert torch.equal(
        train_container["mean_measured_intensity"],
        test_container["mean_measured_intensity"],
    )
    test_local_statistics = derive_ci_experiment_statistics(
        test_container["measured_intensity"].permute(0, 3, 1, 2),
        N,
    )
    assert not torch.equal(
        test_local_statistics.rms_input_scale,
        test_container["rms_input_scale"],
    )
    assert "physics_scaling_constant" not in train_container
    assert "physics_scaling_constant" not in test_container


@pytest.mark.torch
def test_train_with_lightning_registers_dict_training_statistics_before_fit(
    tmp_path,
    monkeypatch,
    params_cfg_snapshot,
):
    import lightning.pytorch as L

    N = 64
    batch_size = 2
    train_path = tmp_path / "train.npz"
    np.savez(train_path, probeGuess=np.ones((N, N), dtype=np.complex64))
    tf_training_config = TFTrainingConfig(
        model=TFModelConfig(
            N=N,
            gridsize=1,
            object_big=False,
            probe_big=False,
        ),
        train_data_file=train_path,
        output_dir=tmp_path / "output",
        batch_size=batch_size,
        n_groups=batch_size,
        nepochs=0,
        torch_loss_mode="poisson",
    )
    update_legacy_dict(params.cfg, tf_training_config)

    container, _, _ = _amplitude_fixture(
        n_samples=batch_size,
        channels=1,
        modes=1,
        N=N,
    )
    training_statistics = torch_components.NormalizedAmplitudeCIDictAdapter(
        count_amplitude_scale=4.0,
        N=N,
        probe_mask=False,
    ).adapt(container)
    container["X"] = container["measured_intensity"]
    captured = {}

    def capture_checkpoint_at_fit(
        self,
        model,
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=None,
        **kwargs,
    ):
        checkpoint = {}
        model.on_save_checkpoint(checkpoint)
        captured["checkpoint"] = checkpoint
        captured["model"] = model

    monkeypatch.setattr(L.Trainer, "fit", capture_checkpoint_at_fit)
    results = torch_components._train_with_lightning(
        train_container=container,
        test_container=None,
        config=tf_training_config,
        execution_config=PyTorchExecutionConfig(
            accelerator="cpu",
            strategy="auto",
            enable_checkpointing=False,
            logger_backend=None,
        ),
        overrides={
            "physics_forward_mode": "rectangular_scaled",
            "cnn_output_mode": "real_imag",
            "scale_contract_version": "ci_intensity_v2",
            "measurement_domain": "count_intensity",
        },
    )

    assert captured["model"] is results["models"]["diffraction_to_obj"]
    checkpoint_statistics = captured["checkpoint"]["ci_statistics"]
    assert torch.equal(
        checkpoint_statistics["rms_input_scale"],
        training_statistics.rms_input_scale.reshape(1),
    )
    assert torch.equal(
        checkpoint_statistics["mean_measured_intensity"],
        training_statistics.mean_measured_intensity.reshape(1),
    )


@pytest.mark.torch
def test_train_with_lightning_registers_native_dataset_statistics_before_fit(
    tmp_path,
    monkeypatch,
    params_cfg_snapshot,
):
    import lightning.pytorch as L

    from ptycho_torch.dataloader import PtychoDataset

    N = 64
    batch_size = 2
    train_path = tmp_path / "train.npz"
    physical_probe = np.ones((N, N), dtype=np.complex64)
    np.savez(train_path, probeGuess=physical_probe)
    tf_training_config = TFTrainingConfig(
        model=TFModelConfig(
            N=N,
            gridsize=1,
            object_big=False,
            probe_big=False,
        ),
        train_data_file=train_path,
        output_dir=tmp_path / "output",
        batch_size=batch_size,
        n_groups=batch_size,
        nepochs=0,
        torch_loss_mode="poisson",
    )
    update_legacy_dict(params.cfg, tf_training_config)

    native_dataset = PtychoDataset.from_np(
        np.linspace(
            1.0,
            10.0,
            batch_size * N * N,
            dtype=np.float32,
        ).reshape(batch_size, N, N),
        physical_probe,
        np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        ModelConfig(
            mode="Unsupervised",
            C_model=1,
            C_forward=1,
            object_big=False,
            physics_forward_mode="rectangular_scaled",
            cnn_output_mode="real_imag",
        ),
        DataConfig(
            N=N,
            C=1,
            grid_size=(1, 1),
            x_bounds=(0.0, 1.0),
            y_bounds=(0.0, 1.0),
            normalize="Batch",
            probe_normalize=True,
        ),
    )
    training_statistics = native_dataset.get_ci_statistics()

    def fail_rederivation(*args, **kwargs):
        raise AssertionError("standalone training must reuse finalized CI statistics")

    monkeypatch.setattr(
        native_dataset,
        "set_ci_statistics_from_indices",
        fail_rederivation,
    )
    captured = {}

    def capture_checkpoint_at_fit(
        self,
        model,
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=None,
        **kwargs,
    ):
        checkpoint = {}
        model.on_save_checkpoint(checkpoint)
        captured["checkpoint"] = checkpoint
        captured["model"] = model

    monkeypatch.setattr(L.Trainer, "fit", capture_checkpoint_at_fit)
    results = torch_components._train_with_lightning(
        train_container=native_dataset,
        test_container=native_dataset,
        config=tf_training_config,
        execution_config=PyTorchExecutionConfig(
            accelerator="cpu",
            strategy="auto",
            enable_checkpointing=False,
            logger_backend=None,
        ),
        overrides={
            "physics_forward_mode": "rectangular_scaled",
            "cnn_output_mode": "real_imag",
            "scale_contract_version": "ci_intensity_v2",
            "measurement_domain": "count_intensity",
            "strategy": None,
            "device": "cpu",
            "num_workers": 1,
        },
    )

    assert captured["model"] is results["models"]["diffraction_to_obj"]
    checkpoint_statistics = captured["checkpoint"]["ci_statistics"]
    for name, expected in training_statistics.items():
        torch.testing.assert_close(checkpoint_statistics[name], expected)


@pytest.mark.torch
@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_ci_shared_probe_is_batch_invariant(
    batch_size,
    tmp_path,
    params_cfg_snapshot,
):
    N = 8
    count_amplitude_scale = 3.0
    container, _, original_probe_2d = _amplitude_fixture(n_samples=8, N=N)
    torch_components.NormalizedAmplitudeCIDictAdapter(
        count_amplitude_scale=count_amplitude_scale,
        N=N,
        probe_scale=1.0,
        probe_mask=False,
    ).adapt(container)
    assert container["probe_physical"].shape == (N, N)
    container["X"] = container["measured_intensity"]

    loader, _ = torch_components._build_lightning_dataloaders(
        container,
        None,
        _tf_training_config(tmp_path, N, batch_size),
        payload=_ci_payload(N),
    )
    tensor_dict, probe_training, probe_normalization = next(iter(loader))

    assert probe_training.shape == (batch_size, 1, 1, N, N)
    assert tensor_dict["probe_physical"].shape == (batch_size, 1, 1, N, N)
    assert tensor_dict["probe_normalization"].shape == (batch_size, 1, 1, 1, 1)

    object_field = torch.complex(
        torch.full((batch_size, 1, N, N), 0.75),
        torch.full((batch_size, 1, N, N), 0.25),
    )
    physics = RectangularScaledDiffraction(ModelConfig(num_datasets=1))
    prediction = physics(
        object_field,
        tensor_dict["measured_intensity"],
        probe_training,
        probe_normalization.reciprocal(),
        tensor_dict["experiment_id"],
    )
    physical_exit_wave = (
        count_amplitude_scale
        * original_probe_2d.view(1, 1, 1, N, N)
        * object_field.unsqueeze(2)
    )
    physical_detector_field = torch.fft.fftshift(
        torch.fft.fft2(physical_exit_wave, norm="ortho"),
        dim=(-2, -1),
    )
    expected_intensity = physical_detector_field.abs().square().sum(dim=2)

    assert prediction.shape == (batch_size, 1, N, N)
    assert torch.allclose(
        prediction,
        expected_intensity,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.torch
def test_ci_two_probe_modes_are_retained_and_summed_incoherently(
    tmp_path,
    params_cfg_snapshot,
):
    N = 8
    batch_size = 2
    container, _, _ = _amplitude_fixture(
        n_samples=4,
        channels=1,
        modes=2,
        N=N,
    )
    torch_components.NormalizedAmplitudeCIDictAdapter(
        count_amplitude_scale=4.0,
        N=N,
        probe_scale=1.0,
        probe_mask=False,
    ).adapt(container)
    container["X"] = container["measured_intensity"]

    loader, _ = torch_components._build_lightning_dataloaders(
        container,
        None,
        _tf_training_config(tmp_path, N, batch_size),
        payload=_ci_payload(N),
    )
    tensor_dict, probe_training, probe_normalization = next(iter(loader))

    assert probe_training.shape == (batch_size, 1, 2, N, N)
    assert tensor_dict["probe_physical"].shape == (batch_size, 1, 2, N, N)

    object_field = torch.complex(
        torch.full((batch_size, 1, N, N), 0.6),
        torch.full((batch_size, 1, N, N), -0.2),
    )
    physics = RectangularScaledDiffraction(ModelConfig(num_datasets=1))
    prediction = physics(
        object_field,
        tensor_dict["measured_intensity"],
        probe_training,
        probe_normalization.reciprocal(),
        tensor_dict["experiment_id"],
    )

    exit_waves = tensor_dict["probe_physical"] * object_field.unsqueeze(2)
    detector_fields = torch.fft.fftshift(
        torch.fft.fft2(exit_waves, norm="ortho"),
        dim=(-2, -1),
    )
    expected = detector_fields.abs().square().sum(dim=2)

    assert torch.allclose(prediction, expected, rtol=1e-5, atol=1e-6)

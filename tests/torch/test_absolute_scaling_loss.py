import math

import pytest
import torch
import torch.nn as nn

import ptycho_torch.model as model_module
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.scaling_contract import (
    LEGACY_SCALE_CONTRACT,
    NORMALIZED_AMPLITUDE,
)


class _FixedRealImagGenerator(nn.Module):
    def __init__(self, object_field: torch.Tensor):
        super().__init__()
        self.register_buffer("object_field", object_field)

    def forward(self, x):
        object_field = self.object_field.expand(x.shape[0], -1, -1, -1)
        return object_field.real, object_field.imag


def _ci_module(
    object_field,
    *,
    probe_mask=False,
    amp_loss=None,
    amp_loss_coeff=1.0,
):
    N = object_field.shape[-1]
    model_config = ModelConfig(
        mode="Unsupervised",
        loss_function="MAE",
        object_big=False,
        C_model=1,
        C_forward=1,
        num_datasets=2,
        physics_forward_mode="rectangular_scaled",
        probe_mask=probe_mask,
        amp_loss=amp_loss,
        amp_loss_coeff=amp_loss_coeff,
    )
    return model_module.PtychoPINN_Lightning(
        model_config,
        DataConfig(N=N, C=1, grid_size=(1, 1)),
        TrainingConfig(torch_loss_mode="poisson"),
        InferenceConfig(),
        generator_module=_FixedRealImagGenerator(object_field),
        generator_output="real_imag",
    )


def _physical_oracle(object_field, probe_physical, mask):
    exit_waves = (
        mask.view(1, 1, 1, *mask.shape)
        * probe_physical
        * object_field.unsqueeze(2)
    )
    detector_fields = torch.fft.fftshift(
        torch.fft.fft2(exit_waves, norm="ortho"),
        dim=(-2, -1),
    )
    return detector_fields.abs().square().sum(dim=2)


def _manual_count_nll(rate, measured_intensity):
    per_pixel = (
        rate
        - measured_intensity * torch.log(rate)
        + torch.lgamma(measured_intensity + 1.0)
    )
    return per_pixel.sum(dim=(1, 2, 3))


def _ci_batch(
    measured_intensity,
    probe_physical,
    probe_normalization,
    mean_measured_intensity,
):
    B, C, H, W = measured_intensity.shape
    probe_training = probe_normalization * probe_physical
    fields = {
        "images": torch.linspace(
            0.25,
            1.25,
            steps=B * C * H * W,
            dtype=measured_intensity.dtype,
        ).reshape(B, C, H, W),
        "measured_intensity": measured_intensity,
        "coords_relative": torch.zeros((B, C, 1, 2)),
        "experiment_id": torch.arange(B, dtype=torch.long),
        "rms_input_scale": torch.full((B, 1, 1, 1), 0.75),
        "mean_measured_intensity": mean_measured_intensity,
        "probe_training": probe_training,
        "probe_physical": probe_physical,
        "probe_normalization": probe_normalization,
    }
    # CI must not use these deprecated tuple aliases. Their values are
    # deliberately incompatible with the named physical-forward contract.
    tuple_probe = torch.full_like(probe_training, 37.0 + 11.0j)
    tuple_probe_normalization = object()
    return fields, tuple_probe, tuple_probe_normalization


def _ci_metric_fixture():
    B, C, P, N = 2, 1, 1, 4
    real = torch.linspace(0.2, 0.9, steps=N * N).reshape(1, C, N, N)
    imag = torch.linspace(-0.15, 0.25, steps=N * N).reshape(1, C, N, N)
    object_field = torch.complex(real, imag)
    probe_real = torch.linspace(
        0.3,
        1.1,
        steps=B * C * P * N * N,
    ).reshape(B, C, P, N, N)
    probe_imag = torch.linspace(
        -0.2,
        0.35,
        steps=B * C * P * N * N,
    ).reshape(B, C, P, N, N)
    probe_physical = torch.complex(probe_real, probe_imag)
    probe_normalization = torch.tensor([0.2, 0.45]).view(B, 1, 1, 1, 1)
    predicted_intensity = _physical_oracle(
        object_field,
        probe_physical,
        torch.ones((N, N)),
    )
    measured_intensity = (
        0.7 * predicted_intensity
        + torch.tensor([0.4, 1.2]).view(B, 1, 1, 1)
    )
    mean_measured_intensity = torch.tensor([2.5, 6.0]).view(B, 1, 1, 1)
    raw_per_sample = _manual_count_nll(
        predicted_intensity.clamp_min(1e-8),
        measured_intensity,
    )
    optimized_loss = (
        raw_per_sample / mean_measured_intensity.reshape(B)
    ).mean()
    batch = _ci_batch(
        measured_intensity,
        probe_physical,
        probe_normalization,
        mean_measured_intensity,
    )
    return (
        _ci_module(object_field),
        batch,
        raw_per_sample.mean(),
        optimized_loss,
    )


def _capture_module_logs(monkeypatch, module):
    captured = {}

    def capture(name, value, **kwargs):
        captured[name] = (torch.as_tensor(value).detach().clone(), kwargs)

    monkeypatch.setattr(module, "log", capture)
    return captured


@pytest.mark.torch
def test_ci_compute_loss_matches_independent_physical_forward_and_reduction():
    torch.manual_seed(20260709)
    B, C, P, N = 2, 1, 2, 8
    object_field = torch.complex(
        0.3 + 0.5 * torch.rand(1, C, N, N),
        -0.2 + 0.4 * torch.rand(1, C, N, N),
    )
    probe_physical = torch.complex(
        0.2 + torch.rand(B, C, P, N, N),
        -0.3 + 0.6 * torch.rand(B, C, P, N, N),
    )
    mask = ((torch.arange(N).view(-1, 1) + torch.arange(N)) % 3 != 0).float()
    q = torch.tensor([0.125, 0.4]).view(B, 1, 1, 1, 1)

    expected_intensity = _physical_oracle(object_field, probe_physical, mask)
    measured_intensity = expected_intensity * 0.8 + 0.35
    physical_mean = torch.tensor([2.0, 7.0]).view(B, 1, 1, 1)
    batch = _ci_batch(
        measured_intensity,
        probe_physical,
        q,
        physical_mean,
    )
    module = _ci_module(object_field, probe_mask=mask)

    captured = {}

    def capture_loss_inputs(_loss, args):
        captured["prediction"] = args[0].detach().clone()
        captured["measurement"] = args[1].detach().clone()

    handle = module.Loss.register_forward_pre_hook(capture_loss_inputs)
    try:
        loss = module.compute_loss(batch)
    finally:
        handle.remove()

    torch.testing.assert_close(captured["prediction"], expected_intensity)
    assert torch.equal(captured["measurement"], measured_intensity)
    expected_raw_per_sample = _manual_count_nll(
        expected_intensity.clamp_min(1e-8),
        measured_intensity,
    )
    expected_loss = (expected_raw_per_sample / physical_mean.reshape(B)).mean()
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(
        module._last_ci_raw_count_nll,
        expected_raw_per_sample.mean(),
    )


@pytest.mark.torch
def test_ci_training_step_logs_raw_pre_normalization_batch_mean(monkeypatch):
    module, batch, expected_raw_mean, expected_optimized_loss = _ci_metric_fixture()
    captured = _capture_module_logs(monkeypatch, module)

    class _NoOpOptimizer:
        def step(self):
            pass

        def zero_grad(self):
            pass

    monkeypatch.setattr(module, "optimizers", lambda: _NoOpOptimizer())
    monkeypatch.setattr(module, "manual_backward", lambda _loss: None)

    loss = module.training_step(batch, batch_idx=0)

    logged_raw, log_kwargs = captured["raw_count_nll_train"]
    torch.testing.assert_close(logged_raw, expected_raw_mean)
    torch.testing.assert_close(loss, expected_optimized_loss)
    assert not torch.isclose(logged_raw, loss)
    assert log_kwargs["on_epoch"] is True
    assert log_kwargs["sync_dist"] is True


@pytest.mark.torch
def test_ci_validation_step_logs_raw_pre_normalization_batch_mean(monkeypatch):
    module, batch, expected_raw_mean, expected_optimized_loss = _ci_metric_fixture()
    captured = _capture_module_logs(monkeypatch, module)

    loss = module.validation_step(batch, batch_idx=0)

    logged_raw, log_kwargs = captured["raw_count_nll_val"]
    torch.testing.assert_close(logged_raw, expected_raw_mean)
    torch.testing.assert_close(loss, expected_optimized_loss)
    assert not torch.isclose(logged_raw, loss)
    assert log_kwargs["on_epoch"] is True
    assert log_kwargs["sync_dist"] is True


@pytest.mark.torch
def test_ci_auxiliary_regularizer_is_added_after_nll_normalization():
    N = 4
    real = torch.linspace(0.2, 1.0, steps=N * N).reshape(1, 1, N, N)
    imag = torch.linspace(-0.3, 0.2, steps=N * N).reshape(1, 1, N, N)
    object_field = torch.complex(real, imag)
    probe_physical = torch.complex(
        torch.linspace(0.4, 1.2, steps=N * N).reshape(1, 1, 1, N, N),
        torch.linspace(-0.1, 0.25, steps=N * N).reshape(1, 1, 1, N, N),
    )
    prediction = _physical_oracle(
        object_field,
        probe_physical,
        torch.ones((N, N)),
    )
    measured_intensity = 0.8 * prediction + 0.6
    mean_measured_intensity = torch.full((1, 1, 1, 1), 5.0)
    coefficient = 1.75
    module = _ci_module(
        object_field,
        amp_loss="Mean_Deviation",
        amp_loss_coeff=coefficient,
    )
    batch = _ci_batch(
        measured_intensity,
        probe_physical,
        torch.full((1, 1, 1, 1, 1), 0.25),
        mean_measured_intensity,
    )

    loss = module.compute_loss(batch)

    raw_nll = _manual_count_nll(
        prediction.clamp_min(1e-8),
        measured_intensity,
    ).mean()
    amplitude = object_field.abs()
    amplitude_mean = amplitude.mean(dim=(2, 3), keepdim=True)
    regularizer = (
        (amplitude - amplitude_mean).abs().sum(dim=(1, 2, 3)).mean()
        * coefficient
    )
    expected = raw_nll / mean_measured_intensity.reshape(()) + regularizer
    incorrectly_normalized_regularizer = (
        raw_nll + regularizer
    ) / mean_measured_intensity.reshape(())
    torch.testing.assert_close(loss, expected)
    assert not torch.isclose(loss, incorrectly_normalized_regularizer)


@pytest.mark.torch
def test_ci_poisson_clamps_zero_rate_at_one_e_minus_eight():
    B, C, P, N = 1, 1, 1, 8
    object_field = torch.zeros((1, C, N, N), dtype=torch.complex64)
    probe_physical = torch.ones((B, C, P, N, N), dtype=torch.complex64)
    measured_intensity = torch.full((B, C, N, N), 2.5)
    physical_mean = torch.full((B, 1, 1, 1), 3.0)
    q = torch.full((B, 1, 1, 1, 1), 0.25)
    module = _ci_module(object_field)

    loss = module.compute_loss(
        _ci_batch(measured_intensity, probe_physical, q, physical_mean)
    )

    expected_raw = _manual_count_nll(
        torch.full_like(measured_intensity, 1e-8),
        measured_intensity,
    ).mean()
    assert torch.isfinite(loss)
    torch.testing.assert_close(module._last_ci_raw_count_nll, expected_raw)
    torch.testing.assert_close(loss, expected_raw / physical_mean.reshape(()))


@pytest.mark.torch
def test_ci_poisson_is_stable_for_high_float32_counts_and_gradients():
    count_values = (1e7, 1e9)
    pred = torch.tensor(
        count_values,
        dtype=torch.float32,
    ).reshape(2, 1, 1, 1).requires_grad_()
    raw = pred.detach().clone()

    loss = model_module.CIIntensityPoissonLoss()(pred, raw)

    reference = torch.tensor(
        [
            value
            - value * math.log(value)
            + math.lgamma(value + 1.0)
            for value in count_values
        ],
        dtype=torch.float64,
    )
    assert loss.dtype == torch.float32
    torch.testing.assert_close(
        loss.to(torch.float64),
        reference,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        loss,
        torch.tensor([8.977986, 11.280571], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.all(torch.isfinite(loss))
    assert torch.all(loss > 0)

    loss.sum().backward()

    assert pred.grad is not None
    assert pred.grad.dtype == torch.float32
    assert torch.all(torch.isfinite(pred.grad))
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred))


@pytest.mark.torch
@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (-1.0, "non-negative"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (float("-inf"), "finite"),
    ],
)
def test_ci_poisson_rejects_invalid_observations(observation, message):
    pred = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    raw = torch.full_like(pred, observation)

    with pytest.raises(RuntimeError, match=message):
        model_module.CIIntensityPoissonLoss()(pred, raw)


@pytest.mark.torch
def test_ci_poisson_does_not_extract_tensor_values_in_python(monkeypatch):
    pred = torch.tensor(
        [[[[2.0, 4.0]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    raw = torch.tensor([[[[3.0, 5.0]]]], dtype=torch.float32)

    def fail_python_extraction(_tensor):
        raise AssertionError("CI Poisson loss must not extract tensor values")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", fail_python_extraction)
        patch.setattr(torch.Tensor, "item", fail_python_extraction)
        loss = model_module.CIIntensityPoissonLoss()(pred, raw)
        loss.sum().backward()

    assert loss.shape == (1,)
    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad))


@pytest.mark.torch
@pytest.mark.parametrize(
    "missing_field",
    [
        "measured_intensity",
        "probe_training",
        "probe_physical",
        "probe_normalization",
        "rms_input_scale",
        "mean_measured_intensity",
    ],
)
def test_ci_compute_loss_requires_every_named_physical_field(missing_field):
    N = 8
    object_field = torch.ones((1, 1, N, N), dtype=torch.complex64)
    probe_physical = torch.ones((1, 1, 1, N, N), dtype=torch.complex64)
    measured_intensity = torch.ones((1, 1, N, N))
    q = torch.ones((1, 1, 1, 1, 1))
    physical_mean = torch.ones((1, 1, 1, 1))
    batch = _ci_batch(measured_intensity, probe_physical, q, physical_mean)
    del batch[0][missing_field]

    with pytest.raises(KeyError, match=missing_field):
        _ci_module(object_field).compute_loss(batch)


@pytest.mark.torch
def test_ci_never_constructs_or_calls_rectangular_mae(monkeypatch):
    events = {"constructed": 0, "called": 0}

    class _RectangularMAESpy(nn.Module):
        def __init__(self):
            super().__init__()
            events["constructed"] += 1

        def forward(self, pred, raw):
            events["called"] += 1
            raise AssertionError("CI must never call RectangularMAELoss")

    monkeypatch.setattr(model_module, "RectangularMAELoss", _RectangularMAESpy)
    N = 8
    object_field = torch.ones((1, 1, N, N), dtype=torch.complex64)
    probe_physical = torch.ones((1, 1, 1, N, N), dtype=torch.complex64)
    measured_intensity = torch.ones((1, 1, N, N))
    q = torch.ones((1, 1, 1, 1, 1))
    physical_mean = torch.ones((1, 1, 1, 1))

    module = _ci_module(object_field)
    module.compute_loss(
        _ci_batch(measured_intensity, probe_physical, q, physical_mean)
    )

    assert events == {"constructed": 0, "called": 0}


@pytest.mark.torch
def test_explicit_legacy_rectangular_mae_keeps_double_square_loss():
    N = 8
    object_field = torch.ones((1, 1, N, N), dtype=torch.complex64)
    module = model_module.PtychoPINN_Lightning(
        ModelConfig(
            mode="Unsupervised",
            object_big=False,
            C_model=1,
            C_forward=1,
            physics_forward_mode="rectangular_scaled",
        ),
        DataConfig(
            N=N,
            C=1,
            grid_size=(1, 1),
            scale_contract_version=LEGACY_SCALE_CONTRACT,
            measurement_domain=NORMALIZED_AMPLITUDE,
        ),
        TrainingConfig(torch_loss_mode="mae"),
        InferenceConfig(),
        generator_module=_FixedRealImagGenerator(object_field),
        generator_output="real_imag",
    )

    assert isinstance(module.Loss, model_module.RectangularMAELoss)
    pred = torch.tensor([[[[2.0]]]])
    raw = torch.tensor([[[[3.0]]]])
    torch.testing.assert_close(module.Loss(pred, raw), torch.tensor([[[[1.0]]]]))


@pytest.mark.torch
def test_explicit_legacy_compute_loss_matches_frozen_pre_ci_oracle():
    B, C, P, N = 2, 1, 2, 4
    real = torch.linspace(0.25, 0.85, steps=N * N).reshape(1, C, N, N)
    imag = torch.linspace(-0.2, 0.3, steps=N * N).reshape(1, C, N, N)
    object_field = torch.complex(real, imag)
    module = model_module.PtychoPINN_Lightning(
        ModelConfig(
            mode="Unsupervised",
            loss_function="MAE",
            object_big=False,
            C_model=1,
            C_forward=1,
            num_datasets=B,
            physics_forward_mode="rectangular_scaled",
        ),
        DataConfig(
            N=N,
            C=C,
            grid_size=(1, 1),
            scale_contract_version=LEGACY_SCALE_CONTRACT,
            measurement_domain=NORMALIZED_AMPLITUDE,
        ),
        TrainingConfig(torch_loss_mode="poisson"),
        InferenceConfig(),
        generator_module=_FixedRealImagGenerator(object_field),
        generator_output="real_imag",
    )
    probe = torch.complex(
        torch.linspace(
            0.2,
            1.1,
            steps=B * C * P * N * N,
        ).reshape(B, C, P, N, N),
        torch.linspace(
            -0.3,
            0.25,
            steps=B * C * P * N * N,
        ).reshape(B, C, P, N, N),
    )
    probe_scaling = torch.tensor([0.2, 0.5]).view(B, 1, 1, 1)
    physics_scale = torch.tensor([3.0, 7.0]).view(B, 1, 1, 1)
    observed = (
        torch.arange(B * C * N * N).reshape(B, C, N, N) % 4 + 1
    ).float()
    fields = {
        "images": torch.linspace(
            0.1,
            0.9,
            steps=B * C * N * N,
        ).reshape(B, C, N, N),
        "observed_images": observed,
        "coords_relative": torch.zeros((B, C, 1, 2)),
        "experiment_id": torch.arange(B, dtype=torch.long),
        "rms_scaling_constant": torch.tensor([0.75, 1.25]).view(B, 1, 1, 1),
        "physics_scaling_constant": physics_scale,
    }
    captured = {}

    def capture_forward_scale(_module, _args, kwargs):
        captured["output_scale"] = kwargs["output_scale_factor"].detach().clone()

    def capture_loss_inputs(_loss, args):
        captured["prediction"] = args[0].detach().clone()
        captured["observation"] = args[1].detach().clone()

    forward_handle = module.register_forward_pre_hook(
        capture_forward_scale,
        with_kwargs=True,
    )
    loss_handle = module.Loss.register_forward_pre_hook(capture_loss_inputs)
    try:
        loss = module.compute_loss((fields, probe, probe_scaling))
    finally:
        forward_handle.remove()
        loss_handle.remove()

    # Frozen legacy_v1 oracle from the pre-CI compute_loss contract.
    expected_output_scale = torch.sqrt(
        1.0 / (probe_scaling.square() * physics_scale + 1e-9)
    )
    exit_waves = (
        expected_output_scale.unsqueeze(2)
        * probe
        * object_field.unsqueeze(2)
    )
    detector_fields = torch.fft.fftshift(
        torch.fft.fft2(exit_waves, norm="ortho"),
        dim=(-2, -1),
    )
    expected_prediction = detector_fields.abs().square().sum(dim=2)
    expected_raw_nll = _manual_count_nll(expected_prediction, observed).mean()
    expected_loss = expected_raw_nll / (observed.mean() + 1e-8)

    assert isinstance(module.Loss, model_module.RectangularPoissonLoss)
    torch.testing.assert_close(captured["output_scale"], expected_output_scale)
    torch.testing.assert_close(captured["prediction"], expected_prediction)
    assert torch.equal(captured["observation"], observed)
    torch.testing.assert_close(loss, expected_loss)
    assert module._last_ci_raw_count_nll is None

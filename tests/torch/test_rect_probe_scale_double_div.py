"""RECT-PROBE-SCALE-DOUBLE-DIV-001: probe_scale double-division pins (Task P1).

The dataloader already folds ``DataConfig.probe_scale`` into the normalized
probe (``helper.normalize_probe_like_tf``, ``dataloader.py:675-682``), yet
``PtychoPINN.forward`` divided the probe by ``probe_scale`` AGAIN before
handing it to the forward model. For the ``rectangular_scaled`` path this
suppressed the predicted intensity by ``probe_scale**2`` (~16x), so Poisson
training converged to objects ~``probe_scale`` (~4.16x) brighter than truth
(washed reconstructions, imag-rail lottery; see
``.superpowers/sdd/ext/task-b4-report.md`` Sec 1a/2 and
``docs/findings.md#RECT-PROBE-SCALE-DOUBLE-DIV-001``).

Two pins:

1. ``test_rectangular_scaled_intensity_matches_count_reference`` -- with a
   dataloader-normalized probe, ``object == truth`` and unit input/output
   scales, the rectangular_scaled predicted intensity must equal the
   count-unit reference ``sum_p |F_ortho{P_norm * O}|^2`` -- the SAME
   convention the inference VarPro basis uses
   (``reassembly.compute_varpro_basis``), keeping train and inference
   consistent. RED on the double-division (off by exactly probe_scale**2).

2. ``test_amplitude_forward_probe_division_byte_identical`` -- the default
   'amplitude' forward must keep dividing the probe by ``probe_scale``
   byte-for-byte (paired-module reference through the model's own
   ``forward_model`` with an explicitly divided probe).
"""
import math

import pytest
import torch
import torch.nn as nn

import ptycho_torch.helper as hh
from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
from ptycho_torch.model import PtychoPINN


class _FixedRealImagGenerator(nn.Module):
    """Stub generator emitting a fixed (real, imag) object, ignoring its input."""

    def __init__(self, real: torch.Tensor, imag: torch.Tensor):
        super().__init__()
        self.register_buffer("real", real)
        self.register_buffer("imag", imag)

    def forward(self, x):
        batch = x.shape[0]
        return (
            self.real.expand(batch, -1, -1, -1),
            self.imag.expand(batch, -1, -1, -1),
        )


class _FixedAmpPhaseGenerator(nn.Module):
    """Stub generator emitting a fixed (amp, phase) object, ignoring its input."""

    def __init__(self, amp: torch.Tensor, phase: torch.Tensor):
        super().__init__()
        self.register_buffer("amp", amp)
        self.register_buffer("phase", phase)

    def forward(self, x):
        batch = x.shape[0]
        return (
            self.amp.expand(batch, -1, -1, -1),
            self.phase.expand(batch, -1, -1, -1),
        )


def _normalized_probe(N: int, probe_scale: float) -> torch.Tensor:
    """Gaussian probe normalized exactly as the dataloader does (probe_scale folded in)."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, N), torch.linspace(-1.0, 1.0, N), indexing="ij"
    )
    raw = torch.exp(-(xx**2 + yy**2) / 0.18) * torch.exp(1j * 0.3 * xx)
    probe_np, _ = hh.normalize_probe_like_tf(
        raw.numpy().astype("complex64"), probe_scale=probe_scale
    )
    return torch.from_numpy(probe_np).to(torch.complex64)


def test_rectangular_scaled_intensity_matches_count_reference():
    torch.manual_seed(0)
    N = 32
    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1))
    model_cfg = ModelConfig(
        object_big=False, C_model=1, C_forward=1,
        physics_forward_mode="rectangular_scaled",
    )
    train_cfg = TrainingConfig()

    real = 0.2 + 0.6 * torch.rand(1, 1, N, N)
    imag = 0.5 * torch.rand(1, 1, N, N) - 0.25
    generator = _FixedRealImagGenerator(real, imag)
    model = PtychoPINN(
        model_cfg, data_cfg, train_cfg,
        generator=generator, generator_output="real_imag",
    )
    model.eval()

    probe = _normalized_probe(N, data_cfg.probe_scale).view(1, 1, 1, N, N)
    obj = torch.complex(real, imag)

    # Count-unit reference: the inference-side VarPro convention
    # (reassembly.compute_varpro_basis with s1=s2=1) and the intended physics
    # of the rect forward at output_scale=1: sum_p |F_ortho{P_norm * O}|^2.
    exit_wave = probe * obj.unsqueeze(2)
    psi = torch.fft.fftshift(torch.fft.fft2(exit_wave, norm="ortho"), dim=(-2, -1))
    intensity_ref = torch.sum(torch.abs(psi) ** 2, dim=2)

    ones = torch.ones(1, 1, 1, 1)
    x_dummy = torch.rand(1, 1, N, N)
    with torch.no_grad():
        intensity_pred, _, _ = model(
            x_dummy, None, probe,
            input_scale_factor=ones, output_scale_factor=ones,
        )

    ratio = (intensity_ref.sum() / intensity_pred.sum()).item()
    assert math.isclose(ratio, 1.0, rel_tol=1e-4), (
        f"rectangular_scaled predicted intensity is {ratio:.2f}x below the "
        f"count-unit reference (probe_scale**2 = {data_cfg.probe_scale**2:.1f}): "
        "probe_scale double-division (RECT-PROBE-SCALE-DOUBLE-DIV-001)"
    )
    torch.testing.assert_close(intensity_pred, intensity_ref, rtol=1e-4, atol=1e-6)


def test_ci_inverse_q_compensation_has_no_hidden_probe_or_output_scale():
    torch.manual_seed(9)
    N = 16
    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), probe_scale=6.25)
    model_cfg = ModelConfig(
        object_big=False,
        C_model=1,
        C_forward=1,
        physics_forward_mode="rectangular_scaled",
    )
    train_cfg = TrainingConfig()

    real = 0.25 + 0.5 * torch.rand(1, 1, N, N)
    imag = -0.2 + 0.4 * torch.rand(1, 1, N, N)
    model = PtychoPINN(
        model_cfg,
        data_cfg,
        train_cfg,
        generator=_FixedRealImagGenerator(real, imag),
        generator_output="real_imag",
    )
    model.eval()

    physical_probe = torch.complex(
        0.1 + torch.rand(1, 1, 2, N, N),
        -0.25 + 0.5 * torch.rand(1, 1, 2, N, N),
    )
    q = torch.tensor(0.16)
    training_probe = q * physical_probe
    object_field = torch.complex(real, imag)
    exit_waves = physical_probe * object_field.unsqueeze(2)
    detector_fields = torch.fft.fftshift(
        torch.fft.fft2(exit_waves, norm="ortho"),
        dim=(-2, -1),
    )
    expected = detector_fields.abs().square().sum(dim=2)

    with torch.no_grad():
        prediction, _, _ = model(
            torch.rand(1, 1, N, N),
            None,
            training_probe,
            input_scale_factor=torch.ones(1, 1, 1, 1),
            output_scale_factor=q.reciprocal().view(1, 1, 1, 1),
            experiment_ids=torch.zeros(1, dtype=torch.long),
        )

    torch.testing.assert_close(prediction, expected, rtol=1e-5, atol=1e-6)


def test_amplitude_forward_probe_division_byte_identical():
    torch.manual_seed(1)
    N = 32
    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1))
    model_cfg = ModelConfig(object_big=False, C_model=1, C_forward=1)
    train_cfg = TrainingConfig()
    assert model_cfg.physics_forward_mode == "amplitude"

    amp = torch.rand(1, 1, N, N)
    phase = math.pi * (2.0 * torch.rand(1, 1, N, N) - 1.0)
    generator = _FixedAmpPhaseGenerator(amp, phase)
    model = PtychoPINN(
        model_cfg, data_cfg, train_cfg,
        generator=generator, generator_output="amp_phase",
    )
    model.eval()

    # Documented (B, C, P, H, W) probe batch layout (PROBE-RANK-001): the
    # amplitude chain's ProbeIllumination now rejects the former rank-2 probe.
    probe = _normalized_probe(N, data_cfg.probe_scale).view(1, 1, 1, N, N)
    ones = torch.ones(1, 1, 1, 1)
    x = torch.rand(1, 1, N, N)
    eids = torch.zeros(1, dtype=torch.long)

    with torch.no_grad():
        out, _, _ = model(
            x, None, probe,
            input_scale_factor=ones, output_scale_factor=ones,
            experiment_ids=eids,
        )
        # Paired-module reference: the amplitude chain fed the probe divided
        # by probe_scale (the pre-P1 convention, which must NOT change).
        x_scaled = model.scaler.scale(x, ones)
        x_complex, _, _ = model._predict_complex(x_scaled)
        ref = model.forward_model.forward(
            x_complex, x_scaled, None,
            probe / data_cfg.probe_scale, ones, eids,
        )

    assert torch.equal(out, ref), (
        "amplitude forward no longer routes probe/probe_scale into "
        "ForwardModel -- the P1 fix must be rectangular_scaled-only"
    )

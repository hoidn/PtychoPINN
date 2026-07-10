"""Absolute-unit VarPro tests built from an independent detector oracle."""

import dataclasses
import math

import pytest
import torch
from tensordict import TensorDict

from ptycho_torch import reassembly
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)


_N = 32
_S1_TRUE = 1.35
_S2_TRUE = 0.72


def _soft_mask(n: int) -> torch.Tensor:
    axis = torch.arange(n, dtype=torch.float32) - n // 2 + 0.5
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    return torch.sigmoid((0.36 * n - radius) / 1.4)


def _physical_probe(n_modes: int) -> torch.Tensor:
    axis = torch.arange(_N, dtype=torch.float32) - _N // 2
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    modes = []
    for mode in range(n_modes):
        x_shift = 2.5 * mode
        envelope = torch.exp(
            -((xx - x_shift).square() + (yy + 1.5 * mode).square())
            / (2 * (_N / (5.5 + mode)) ** 2)
        )
        phase = torch.exp(1j * ((0.17 + 0.08 * mode) * xx / _N + 0.11 * yy / _N))
        modes.append((envelope * phase / math.sqrt(mode + 1)).to(torch.complex64))
    return torch.stack(modes).view(1, 1, n_modes, _N, _N)


def _textures() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260709)
    n_samples = 4
    real = 0.55 + 0.5 * torch.rand(
        n_samples, 1, _N, _N, generator=generator
    )
    imag = 0.35 * (
        torch.rand(n_samples, 1, _N, _N, generator=generator) - 0.5
    )
    axis = torch.linspace(-1.0, 1.0, _N)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    imag = imag + 0.25 * torch.sin(3.1 * xx)[None, None] * torch.cos(
        2.3 * yy
    )[None, None]
    return torch.complex(real, imag)


def _detector_oracle(
    probe_physical: torch.Tensor,
    textures: torch.Tensor,
    s1: float,
    s2: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Generate counts without using reassembly.compute_varpro_basis."""
    probe = probe_physical * mask.view(1, 1, 1, _N, _N)
    exit_wave = probe * (
        s1 * textures.real.unsqueeze(2) + 1j * s2 * textures.imag.unsqueeze(2)
    )
    detector_wave = torch.fft.fftshift(
        torch.fft.fft2(exit_wave, norm="ortho"), dim=(-2, -1)
    )
    return detector_wave.abs().square().sum(dim=2)


class _FixedTextureModel(torch.nn.Module):
    def __init__(self, textures: torch.Tensor):
        super().__init__()
        self.register_buffer("textures", textures)
        self.forward_inputs = []

    def forward_predict(self, intensity, positions, probe, input_scale):
        self.forward_inputs.append(
            (intensity.detach().clone(), probe.detach().clone(), input_scale.detach().clone())
        )
        return self.textures[: intensity.shape[0]]


class _CIDataset:
    def __init__(
        self,
        measured_intensity: torch.Tensor,
        probe_physical: torch.Tensor,
        textures: torch.Tensor,
    ):
        n_samples = textures.shape[0]
        coords = torch.tensor(
            [[64.0, 64.0], [68.0, 64.0], [64.0, 68.0], [68.0, 68.0]]
        ).view(n_samples, 1, 1, 2)
        physical_batch = probe_physical.expand(n_samples, -1, -1, -1, -1).clone()
        poison_training_probe = 0.07 * physical_batch
        self.mmap_ptycho = TensorDict(
            {
                "images": measured_intensity * 0.0 + 19.0,
                "measured_intensity": measured_intensity,
                "coords_relative": torch.zeros(n_samples, 1, 1, 2),
                "coords_global": coords,
                "experiment_id": torch.zeros(n_samples, dtype=torch.long),
                "rms_input_scale": torch.full((n_samples, 1, 1, 1), 0.25),
                "mean_measured_intensity": measured_intensity.mean().expand(
                    n_samples, 1, 1, 1
                ),
                "probe_physical": physical_batch,
                "probe_training": poison_training_probe,
                "probe_normalization": torch.full(
                    (n_samples, 1, 1, 1, 1), 0.07
                ),
            },
            batch_size=[n_samples],
        )
        self.data_dict = {}
        self.n_files = 1
        self._poison_training_probe = poison_training_probe[0]

    def __len__(self):
        return self.mmap_ptycho.batch_size[0]

    def __getitem__(self, index):
        batch = self.mmap_ptycho[index]
        batch_size = batch.batch_size[0]
        tuple_probe = self._poison_training_probe.unsqueeze(0).expand(
            batch_size, -1, -1, -1, -1
        )
        poison_output_scale_source = torch.full((batch_size, 1, 1, 1), 11.0)
        return batch, tuple_probe, poison_output_scale_source


def _run_ci_reconstruction(
    probe_physical: torch.Tensor,
    measured_intensity: torch.Tensor,
    textures: torch.Tensor,
    *,
    mask: torch.Tensor,
    num_workers: int = 1,
):
    model = _FixedTextureModel(textures)
    dataset = _CIDataset(measured_intensity, probe_physical, textures)
    data_config = DataConfig(
        N=_N,
        C=1,
        scale_contract_version="ci_intensity_v2",
        measurement_domain="count_intensity",
    )
    model_config = ModelConfig(
        physics_forward_mode="rectangular_scaled",
        probe_mask_tensor=mask,
    )
    training_config = dataclasses.replace(
        TrainingConfig(), device="cpu", num_workers=num_workers
    )
    inference_config = InferenceConfig(
        middle_trim=16,
        batch_size=16,
        patch_weighting="uniform",
        varpro_scaling=True,
    )
    result = reassembly.reconstruct_image_barycentric(
        model,
        dataset,
        training_config,
        data_config,
        model_config,
        inference_config,
        gpu_ids=None,
        verbose=False,
        swap_detection="None",
        return_diagnostics=True,
    )
    return result, model, dataset


def test_cpu_reconstruction_does_not_synchronize_cuda(monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("CPU reconstruction executed a torch.cuda operation")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_if_called)
    monkeypatch.setattr(torch.cuda, "empty_cache", fail_if_called)
    textures = _textures()
    probe_physical = _physical_probe(1)
    mask = torch.ones(_N, _N)
    measured_intensity = _detector_oracle(
        probe_physical, textures, _S1_TRUE, _S2_TRUE, mask
    )

    _run_ci_reconstruction(
        probe_physical, measured_intensity, textures, mask=mask
    )


def test_cpu_zero_worker_varpro_reconstruction_end_to_end():
    textures = _textures()
    probe_physical = _physical_probe(2)
    mask = _soft_mask(_N)
    measured_intensity = _detector_oracle(
        probe_physical, textures, _S1_TRUE, _S2_TRUE, mask
    )

    (canvas, _subset, stats, _prescale), model, _dataset = _run_ci_reconstruction(
        probe_physical,
        measured_intensity,
        textures,
        mask=mask,
        num_workers=0,
    )

    assert len(model.forward_inputs) == 1
    assert float(stats[4]) == pytest.approx(_S1_TRUE, rel=3e-2)
    assert float(stats[5]) == pytest.approx(_S2_TRUE, rel=3e-2)
    assert torch.isfinite(canvas.real).all()


def test_cuda_timing_sync_targets_requested_device(monkeypatch):
    calls = []
    device = torch.device("cuda:3")
    monkeypatch.setattr(torch.cuda, "synchronize", calls.append)

    reassembly._synchronize_cuda_for_timing(device)

    assert calls == [device]


def test_assembly_timing_syncs_at_both_boundaries(monkeypatch):
    events = []
    real_time = reassembly.time.time
    original_basis = reassembly.compute_varpro_basis
    original_accumulate = reassembly.VectorizedWeightedAccumulator.accumulate_batch

    def record_time():
        events.append("time")
        return real_time()

    def record_basis(*args, **kwargs):
        events.append("basis")
        return original_basis(*args, **kwargs)

    def record_accumulate(*args, **kwargs):
        events.append("stitch")
        return original_accumulate(*args, **kwargs)

    monkeypatch.setattr(
        reassembly,
        "_synchronize_cuda_for_timing",
        lambda device: events.append("sync"),
    )
    monkeypatch.setattr(reassembly.time, "time", record_time)
    monkeypatch.setattr(reassembly, "compute_varpro_basis", record_basis)
    monkeypatch.setattr(
        reassembly.VectorizedWeightedAccumulator,
        "accumulate_batch",
        record_accumulate,
    )

    textures = _textures()
    probe_physical = _physical_probe(1)
    mask = torch.ones(_N, _N)
    measured_intensity = _detector_oracle(
        probe_physical, textures, _S1_TRUE, _S2_TRUE, mask
    )
    _run_ci_reconstruction(
        probe_physical,
        measured_intensity,
        textures,
        mask=mask,
        num_workers=0,
    )

    basis_index = events.index("basis")
    stitch_index = events.index("stitch")
    assert events[basis_index - 2:basis_index] == ["sync", "time"]
    assert events[stitch_index + 1:stitch_index + 3] == ["sync", "time"]


@pytest.mark.parametrize(
    ("n_modes", "mask"),
    [(1, torch.ones(_N, _N)), (2, _soft_mask(_N))],
    ids=["full-frame-single-mode", "soft-mask-two-modes"],
)
def test_ci_varpro_recovers_known_scales_from_physical_probe(
    monkeypatch, n_modes, mask
):
    textures = _textures()
    probe_physical = _physical_probe(n_modes)
    measured_intensity = _detector_oracle(
        probe_physical, textures, _S1_TRUE, _S2_TRUE, mask
    )
    basis_scales = []
    original_compute = reassembly.compute_varpro_basis

    def capture_compute(probe, real, imag, scale=None):
        basis_scales.append(scale)
        return original_compute(probe, real, imag, scale=scale)

    monkeypatch.setattr(reassembly, "compute_varpro_basis", capture_compute)
    (canvas, _subset, stats, _prescale), model, dataset = _run_ci_reconstruction(
        probe_physical, measured_intensity, textures, mask=mask
    )

    assert float(stats[4]) == pytest.approx(_S1_TRUE, rel=3e-2)
    assert float(stats[5]) == pytest.approx(_S2_TRUE, rel=3e-2)
    assert stats[2].shape[-2:] == (_N, _N)
    assert stats[3].shape[2] == n_modes
    assert basis_scales == [None]
    seen_intensity, seen_probe, seen_input_scale = model.forward_inputs[0]
    torch.testing.assert_close(seen_intensity, measured_intensity)
    torch.testing.assert_close(
        seen_probe, dataset.mmap_ptycho["probe_physical"]
    )
    torch.testing.assert_close(
        seen_input_scale, dataset.mmap_ptycho["rms_input_scale"]
    )
    assert torch.isfinite(canvas.real).all()


def test_ci_varpro_object_scale_is_invariant_under_calibrated_dose():
    textures = _textures()
    base_probe = _physical_probe(2)
    mask = _soft_mask(_N)
    recovered = []

    for dose in (0.25, 1.0, 4.0):
        probe_physical = math.sqrt(dose) * base_probe
        measured_intensity = _detector_oracle(
            probe_physical, textures, _S1_TRUE, _S2_TRUE, mask
        )
        (canvas, _subset, stats, _prescale), _model, _dataset = (
            _run_ci_reconstruction(
                probe_physical, measured_intensity, textures, mask=mask
            )
        )
        recovered.append((canvas, float(stats[4]), float(stats[5])))

    for canvas, s1, s2 in recovered:
        assert s1 == pytest.approx(_S1_TRUE, rel=3e-2)
        assert s2 == pytest.approx(_S2_TRUE, rel=3e-2)
        torch.testing.assert_close(canvas, recovered[1][0], rtol=3e-2, atol=3e-3)


def test_ci_varpro_fixed_probe_negative_control_tracks_sqrt_dose():
    textures = _textures()
    probe_physical = _physical_probe(2)
    mask = _soft_mask(_N)
    base_intensity = _detector_oracle(
        probe_physical, textures, _S1_TRUE, _S2_TRUE, mask
    )

    for dose in (0.25, 1.0, 4.0):
        (_canvas, _subset, stats, _prescale), _model, _dataset = (
            _run_ci_reconstruction(
                probe_physical,
                dose * base_intensity,
                textures,
                mask=mask,
            )
        )
        expected = math.sqrt(dose)
        assert float(stats[4]) == pytest.approx(expected * _S1_TRUE, rel=3e-2)
        assert float(stats[5]) == pytest.approx(expected * _S2_TRUE, rel=3e-2)

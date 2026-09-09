"""Differentiable ptychographic forward and smooth-field refinement."""

from __future__ import annotations

import inspect

import numpy as np
import torch

from ptycho_torch.phase_field_refinement import (
    SmoothField,
    expected_counts,
    extract_canvas_patches,
    fit_gain_field,
    fit_tilt_field,
    frame_gains,
    frame_tilts,
    poisson_nll,
    refine_phase_field,
)
from scripts.studies.make_synthetic_truth_datasets import (
    extract_object_patches,
    noiseless_detector_intensity,
)

H = W = 64
M = 16


def _anchor():
    return {
        "scan_com": [W // 2, H // 2],
        "canvas_shape": [H, W],
        "canvas_origin_offset": [0.0, 0.0],
    }


def _scene(seed: int, frames: int = 80):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:H, :W]
    amplitude = 0.7 + 0.2 * np.cos(x / 5.0) * np.sin(y / 7.0)
    phase = 0.8 * np.sin(x / 8.0) + 0.5 * np.cos(y / 6.0)
    truth = np.asarray(amplitude * np.exp(1j * phase), dtype=np.complex64)
    grid = np.arange(M) - M / 2 + 0.5
    radius = grid[:, None] ** 2 + grid[None, :] ** 2
    probe = np.asarray(
        100.0 * np.exp(-radius / 18.0) * np.exp(1j * 0.02 * radius),
        dtype=np.complex64,
    )
    xcoords = rng.uniform(M / 2 + 1, W - M / 2 - 2, frames)
    ycoords = rng.uniform(M / 2 + 1, H - M / 2 - 2, frames)
    return truth, probe, xcoords, ycoords


def _smooth_canvas(height: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    frequency = (
        torch.fft.fftfreq(height)[:, None] ** 2
        + torch.fft.fftfreq(width)[None, :] ** 2
    )
    spectrum = torch.randn(
        height, width, generator=generator, dtype=torch.complex64
    ) * torch.exp(-frequency * 400)
    field = torch.fft.ifft2(spectrum)
    field /= field.abs().mean()
    return (1 + 0.3 * field.real) * torch.polar(
        torch.ones(height, width), 0.5 * field.imag
    )


def _gaussian_probe(size: int, sigma: float) -> torch.Tensor:
    grid = torch.arange(size) - size / 2
    return (
        torch.exp(
            -(grid[:, None].square() + grid[None, :].square()) / (2 * sigma**2)
        ).to(torch.complex64)
        * 30
    )


def _identity_anchor(positions: torch.Tensor, shape: tuple[int, int]):
    height, width = shape
    anchor = {
        "scan_com": [width // 2, height // 2],
        "canvas_shape": [height, width],
        "canvas_origin_offset": [0.0, 0.0],
    }
    return anchor, positions[:, 0].numpy(), positions[:, 1].numpy()


def test_frame_tilts_reads_a_global_ramp_with_sign_and_scale():
    height = width = 160
    size = 32
    canvas = _smooth_canvas(height, width, 0)
    probe = _gaussian_probe(size, 4.0)
    axis = torch.arange(40, 121, 20.0)
    positions = torch.stack(
        torch.meshgrid(axis, axis, indexing="xy"), -1
    ).reshape(-1, 2)
    tilt = torch.tensor([0.03, -0.02])
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    ramped = canvas * torch.polar(
        torch.ones(height, width), tilt[0] * x + tilt[1] * y
    )
    counts = expected_counts(ramped, positions, probe)

    estimate = frame_tilts(canvas, positions, probe, counts)

    assert estimate.shape == (len(positions), 2)
    assert torch.allclose(estimate, tilt.expand_as(estimate), atol=2e-3), estimate


def test_frame_gains_reads_a_smooth_log_amplitude_correction():
    height = width = 160
    size = 32
    canvas = _smooth_canvas(height, width, 1)
    probe = _gaussian_probe(size, 4.0)
    axis = torch.arange(40, 121, 20.0)
    positions = torch.stack(
        torch.meshgrid(axis, axis, indexing="xy"), -1
    ).reshape(-1, 2)
    _, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    log_gain = 0.03 + 0.02 * x / width
    brighter = canvas * torch.exp(log_gain)
    counts = expected_counts(brighter, positions, probe)

    estimate = frame_gains(canvas, positions, probe, counts)

    expected = 0.03 + 0.02 * positions[:, 0] / width
    assert estimate.shape == (len(positions),)
    assert torch.allclose(estimate, expected, atol=1e-3), estimate


def test_fit_tilt_field_integrates_a_linear_field():
    field = SmoothField((160, 160), pitch_px=16)
    axis = torch.arange(24, 137, 8.0)
    positions = torch.stack(
        torch.meshgrid(axis, axis, indexing="xy"), -1
    ).reshape(-1, 2)
    tilts = torch.tensor([0.01, 0.004]).expand(len(positions), 2)

    residual = fit_tilt_field(field, positions, tilts)

    assert residual < 5e-4
    phase = field().detach()
    gradient_x = (phase[80, 100] - phase[80, 60]) / 40
    gradient_y = (phase[100, 80] - phase[60, 80]) / 40
    assert abs(gradient_x - 0.01) < 5e-4
    assert abs(gradient_y - 0.004) < 5e-4


def test_fit_gain_field_integrates_constant_plus_ramp_values():
    field = SmoothField((160, 160), pitch_px=16)
    axis = torch.arange(24, 137, 8.0)
    positions = torch.stack(
        torch.meshgrid(axis, axis, indexing="xy"), -1
    ).reshape(-1, 2)
    gains = 0.04 + 0.02 * positions[:, 0] / 160 - 0.01 * positions[:, 1] / 160
    probe_intensity = _gaussian_probe(32, 4.0).abs().square()

    residual = fit_gain_field(
        field,
        positions,
        gains,
        probe_intensity=probe_intensity,
    )

    fitted = field().detach()
    assert residual < 5e-4
    assert abs(fitted[80, 64] - (0.04 + 0.02 * 64 / 160 - 0.01 * 80 / 160)) < 5e-4
    assert abs(fitted[96, 112] - (0.04 + 0.02 * 112 / 160 - 0.01 * 96 / 160)) < 5e-4


def test_seeded_refinement_removes_a_ramp_from_a_synthetic_record():
    height = width = 160
    size = 32
    truth = _smooth_canvas(height, width, 2)
    probe = _gaussian_probe(size, 4.0)
    axis = torch.arange(32, 129, 6.0)
    positions = torch.stack(
        torch.meshgrid(axis, axis, indexing="xy"), -1
    ).reshape(-1, 2)
    counts = torch.poisson(
        expected_counts(truth, positions, probe),
        generator=torch.Generator().manual_seed(3),
    )
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    wrong = truth * torch.polar(
        torch.ones(height, width),
        -(0.02 * x - 0.015 * y)
        + 0.0004 * ((x - 80).square() - (y - 80).square()) / 2,
    )
    anchor, xcoords, ycoords = _identity_anchor(positions, (height, width))
    support = torch.zeros(height, width, dtype=torch.bool)
    support[24:136, 24:136] = True

    result = refine_phase_field(
        wrong.numpy(),
        anchor,
        counts.numpy(),
        xcoords,
        ycoords,
        probe.numpy(),
        support_mask=support.numpy(),
        seed="tilt",
        max_iter=30,
        device="cpu",
    )

    assert result.seed == "tilt"
    assert result.nll_after_seed < result.nll_before
    assert result.nll_after <= result.nll_after_seed
    seed_residual = torch.angle(
        wrong
        * torch.polar(torch.ones(height, width), torch.as_tensor(result.seed_field))
        * torch.conj(truth)
    )[support]
    full_residual = torch.angle(torch.as_tensor(result.canvas) * torch.conj(truth))[
        support
    ]
    before = torch.angle(wrong * torch.conj(truth))[support]
    assert seed_residual.std() < 0.5 * before.std()
    assert full_residual.std() < seed_residual.std()


def test_extract_canvas_patches_matches_dataset_extractor():
    truth, _, xcoords, ycoords = _scene(0, frames=8)
    expected = extract_object_patches(truth, xcoords, ycoords, M)
    actual = extract_canvas_patches(
        torch.as_tensor(truth),
        torch.as_tensor(np.column_stack((xcoords, ycoords)), dtype=torch.float32),
        M,
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_expected_counts_matches_multimode_dataset_forward():
    truth, probe, xcoords, ycoords = _scene(1, frames=8)
    modes = np.stack((probe, 0.2 * probe))
    expected = noiseless_detector_intensity(
        extract_object_patches(truth, xcoords, ycoords, M), modes
    )
    actual = expected_counts(
        torch.as_tensor(truth),
        torch.as_tensor(np.column_stack((xcoords, ycoords)), dtype=torch.float32),
        torch.as_tensor(modes),
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-4, atol=1e-3)


def test_poisson_nll_is_minimized_at_the_data():
    counts = torch.tensor([[3.0, 0.0], [10.0, 1.0]])
    assert poisson_nll(counts, counts) <= poisson_nll(1.1 * counts, counts)
    assert poisson_nll(counts, counts) <= poisson_nll(0.9 * counts, counts)


def test_poisson_nll_uses_float64_accumulation():
    expected = torch.full((400_000,), 8192.0, dtype=torch.float32)
    counts = torch.zeros_like(expected)
    shifted = expected.clone()
    shifted[0] += 1

    baseline = poisson_nll(expected, counts)
    changed = poisson_nll(shifted, counts)

    assert baseline.dtype == torch.float64
    assert changed - baseline == 1


def test_refinement_defaults_to_10_iterations():
    assert inspect.signature(refine_phase_field).parameters["max_iter"].default == 10


def test_smooth_field_is_zero_and_canvas_shaped_at_initialization():
    field = SmoothField((H, W), 8)()
    assert field.shape == (H, W)
    assert torch.count_nonzero(field) == 0


def test_refinement_reduces_a_smooth_phase_error():
    truth, probe, xcoords, ycoords = _scene(2)
    counts = noiseless_detector_intensity(
        extract_object_patches(truth, xcoords, ycoords, M), probe
    ).astype(np.float32)
    y, x = np.mgrid[:H, :W]
    error = 0.25 * np.sin(x / 6.0) + 0.2 * np.cos(y / 5.0)
    corrupted = np.asarray(truth * np.exp(-1j * error), dtype=np.complex64)
    result = refine_phase_field(
        corrupted,
        _anchor(),
        counts,
        xcoords,
        ycoords,
        probe,
        support_mask=np.ones((H, W), dtype=bool),
        grid_pitch_px=6,
        max_iter=35,
        device="cpu",
    )
    before = np.angle(corrupted * np.conj(truth))[M:-M, M:-M]
    after = np.angle(result.canvas * np.conj(truth))[M:-M, M:-M]
    before -= before.mean()
    after -= after.mean()
    assert result.nll_after < result.nll_before
    assert np.sqrt(np.mean(after**2)) < 0.5 * np.sqrt(np.mean(before**2))


def test_joint_refinement_reduces_smooth_amplitude_and_phase_errors():
    truth, probe, xcoords, ycoords = _scene(4)
    counts = noiseless_detector_intensity(
        extract_object_patches(truth, xcoords, ycoords, M), probe
    ).astype(np.float32)
    y, x = np.mgrid[:H, :W]
    amplitude_error = 0.1 * np.sin(x / 10.0)
    phase_error = 0.25 * np.sin(x / 6.0) + 0.2 * np.cos(y / 5.0)
    corrupted = np.asarray(
        truth * np.exp(-amplitude_error - 1j * phase_error), dtype=np.complex64
    )

    result = refine_phase_field(
        corrupted,
        _anchor(),
        counts,
        xcoords,
        ycoords,
        probe,
        support_mask=np.ones((H, W), dtype=bool),
        grid_pitch_px=6,
        max_iter=20,
        fit_log_amplitude=True,
        device="cpu",
    )

    interior = np.s_[M:-M, M:-M]
    before = np.abs(corrupted)[interior] - np.abs(truth)[interior]
    after = np.abs(result.canvas)[interior] - np.abs(truth)[interior]
    assert np.sqrt(np.mean(after**2)) < 0.5 * np.sqrt(np.mean(before**2))
    assert result.log_amplitude_field is not None
    assert np.isfinite(result.canvas).all()
    assert np.isfinite(result.log_amplitude_field).all()
    assert result.nll_after <= result.nll_after_seed


def test_joint_refinement_is_a_noop_on_truth():
    truth, probe, xcoords, ycoords = _scene(5)
    expected = noiseless_detector_intensity(
        extract_object_patches(truth, xcoords, ycoords, M), probe
    )
    counts = np.random.default_rng(9).poisson(expected).astype(np.float32)

    result = refine_phase_field(
        truth,
        _anchor(),
        counts,
        xcoords,
        ycoords,
        probe,
        support_mask=np.ones((H, W), dtype=bool),
        grid_pitch_px=8,
        max_iter=20,
        fit_log_amplitude=True,
        device="cpu",
    )

    relative_nll_change = abs(result.nll_after - result.nll_before) / abs(
        result.nll_before
    )
    gain = np.exp(result.log_amplitude_field[M:-M, M:-M])
    assert relative_nll_change < 1e-3
    assert np.sqrt(np.mean((gain - 1) ** 2)) < 0.01


def test_gain_seed_rms_uses_every_chunk():
    truth, probe, xcoords, ycoords = _scene(6)
    counts = noiseless_detector_intensity(
        extract_object_patches(truth, xcoords, ycoords, M), probe
    ).astype(np.float32)
    _, x = np.mgrid[:H, :W]
    corrupted = np.asarray(truth * np.exp(-0.1 * np.sin(x / 7.0)), dtype=np.complex64)

    result = refine_phase_field(
        corrupted,
        _anchor(),
        counts,
        xcoords,
        ycoords,
        probe,
        support_mask=np.ones((H, W), dtype=bool),
        grid_pitch_px=8,
        seed="none",
        fit_log_amplitude=True,
        max_iter=0,
        chunk=20,
        device="cpu",
    )

    positions = torch.as_tensor(
        np.column_stack((xcoords, ycoords)), dtype=torch.float32
    )
    target = frame_gains(
        torch.as_tensor(corrupted),
        positions,
        torch.as_tensor(probe),
        torch.as_tensor(counts),
        chunk=20,
    )
    fitted = extract_canvas_patches(
        torch.as_tensor(result.log_amplitude_field).to(torch.complex64), positions, M
    )
    intensity = torch.as_tensor(np.abs(probe) ** 2)
    fitted = (fitted * intensity / intensity.sum()).sum((-2, -1))
    expected = float((fitted - target).square().mean().sqrt())
    assert abs(result.gain_seed_rms - expected) < 1e-6


def test_truth_refinement_stays_below_the_noise_floor():
    truth, probe, xcoords, ycoords = _scene(3)
    expected = noiseless_detector_intensity(
        extract_object_patches(truth, xcoords, ycoords, M), probe
    )
    counts = np.random.default_rng(8).poisson(expected).astype(np.float32)
    result = refine_phase_field(
        truth,
        _anchor(),
        counts,
        xcoords,
        ycoords,
        probe,
        support_mask=np.ones((H, W), dtype=bool),
        grid_pitch_px=8,
        max_iter=20,
        device="cpu",
    )
    relative_nll_change = abs(result.nll_after - result.nll_before) / abs(
        result.nll_before
    )
    assert relative_nll_change < 1e-3
    assert np.sqrt(np.mean(result.phase_field[M:-M, M:-M] ** 2)) < 0.02

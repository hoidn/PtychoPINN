import json
import subprocess
import warnings

import pytest
import torch

from ptycho_torch import reassembly


class DummyScaler:
    def __init__(self):
        self.calls = 0

    def solve_lbfgs(self, *args, **kwargs):
        self.calls += 1
        return torch.tensor(2.0), torch.tensor(0.5)


def test_apply_varpro_canvas_scaling_bypasses_solver_when_disabled():
    canvas = torch.tensor([[1 + 4j, 2 + 6j]], dtype=torch.complex64)
    scaler = DummyScaler()

    scaled, s1, s2 = reassembly.apply_varpro_canvas_scaling(
        canvas,
        scaler,
        enabled=False,
        verbose=False,
    )

    assert scaler.calls == 0
    torch.testing.assert_close(scaled, canvas)
    torch.testing.assert_close(s1, torch.tensor(1.0))
    torch.testing.assert_close(s2, torch.tensor(1.0))


def test_apply_varpro_canvas_scaling_uses_solver_when_enabled():
    canvas = torch.tensor([[1 + 4j, 2 + 6j]], dtype=torch.complex64)
    scaler = DummyScaler()

    scaled, s1, s2 = reassembly.apply_varpro_canvas_scaling(
        canvas,
        scaler,
        enabled=True,
        verbose=False,
    )

    assert scaler.calls == 1
    torch.testing.assert_close(
        scaled,
        torch.tensor([[2 + 2j, 4 + 3j]], dtype=torch.complex64),
    )
    torch.testing.assert_close(s1, torch.tensor(2.0))
    torch.testing.assert_close(s2, torch.tensor(0.5))


def test_varpro_scaler_recovers_known_channel_scales():
    device = torch.device("cpu")
    scaler = reassembly.VarProScaler(device)

    y = torch.linspace(-1.0, 1.0, 16, device=device)
    x = torch.linspace(-1.0, 1.0, 16, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    X1 = (1.0 + xx.square()).unsqueeze(0)
    X2 = (0.8 + yy.square()).unsqueeze(0)
    X3 = torch.zeros_like(X1)
    expected_s1 = torch.tensor(1.7)
    expected_s2 = torch.tensor(0.6)
    I_raw = expected_s1.square() * X1 + expected_s2.square() * X2

    scaler.accumulate_batch_from_basis(I_raw, X1, X2, X3)
    s1, s2 = scaler.solve_lbfgs(verbose=False)

    torch.testing.assert_close(s1, expected_s1, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(s2, expected_s2, rtol=2e-2, atol=2e-2)


def test_varpro_basis_ffts_preserve_parseval_energy():
    """``reassembly.compute_varpro_basis`` must use an energy-preserving
    (norm='ortho') 2D FFT, matching every other forward-physics FFT in this
    codebase (this file's own ``detect_swap_probe_reference``,
    ``ptycho_torch/model.py``). Parseval's theorem: for an orthonormal
    transform, sum(|FFT(x)|^2) == sum(|x|^2) exactly. An unnormalized fft2
    inflates this sum by exactly H*W, silently rescaling every downstream
    VarPro-fitted s1/s2 by sqrt(H*W).
    """
    torch.manual_seed(0)
    B, C, P, H, W = 1, 1, 3, 32, 32
    probe = torch.randn(B, C, P, H, W, dtype=torch.complex64)
    a_tilde = torch.randn(B, C, H, W)
    b_tilde = torch.randn(B, C, H, W)

    _Psi_a, _Psi_b, X1, X2, _X3 = reassembly.compute_varpro_basis(probe, a_tilde, b_tilde)

    exit_wave_a = probe * a_tilde.unsqueeze(2)
    exit_wave_b = 1j * probe * b_tilde.unsqueeze(2)
    energy_in_a = torch.sum(torch.abs(exit_wave_a) ** 2)
    energy_in_b = torch.sum(torch.abs(exit_wave_b) ** 2)

    torch.testing.assert_close(torch.sum(X1), energy_in_a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(torch.sum(X2), energy_in_b, rtol=1e-4, atol=1e-4)


def test_varpro_basis_matches_probe_reference_ortho_convention():
    """For a transparent object (a_tilde=1, b_tilde=0), Psi_a must match this
    same file's own reference FFT convention used by
    ``detect_swap_probe_reference``: fftshift(fft2(probe, norm='ortho')).
    Ties the VarPro basis FFT to the codebase's own already-correct
    convention instead of re-deriving it independently.
    """
    torch.manual_seed(0)
    B, C, P, H, W = 1, 1, 2, 32, 32
    probe = torch.randn(B, C, P, H, W, dtype=torch.complex64)
    a_tilde = torch.ones(B, C, H, W)
    b_tilde = torch.zeros(B, C, H, W)

    Psi_a, _Psi_b, _X1, _X2, _X3 = reassembly.compute_varpro_basis(probe, a_tilde, b_tilde)

    expected = torch.fft.fftshift(torch.fft.fft2(probe, norm='ortho'), dim=(-2, -1))
    torch.testing.assert_close(Psi_a, expected)


def _assemble_two_patch_overlap(*, uniform_weighting: bool) -> torch.Tensor:
    device = torch.device("cpu")
    patch_size = 8
    canvas_shape = (12, 16)
    canvas = torch.zeros(canvas_shape, dtype=torch.complex64, device=device)
    weights = torch.zeros(canvas_shape, dtype=torch.float32, device=device)
    accumulator = reassembly.VectorizedWeightedAccumulator(canvas_shape, device)

    left_patch = torch.ones((patch_size, patch_size), dtype=torch.complex64, device=device)
    right_patch = torch.ones((patch_size, patch_size), dtype=torch.complex64, device=device)
    left_patch[:, -2:] = 5 + 0j

    patches = torch.stack([left_patch, right_patch])
    positions = torch.tensor([[6.0, 6.0], [8.0, 6.0]], dtype=torch.float32, device=device)

    probe_weight = torch.ones((patch_size, patch_size), dtype=torch.float32, device=device)
    probe_weight[:, -2:] = 0.05

    accumulator.accumulate_batch(
        canvas,
        weights,
        patches,
        positions,
        probe_weight,
        patch_size=patch_size,
        uniform_weighting=uniform_weighting,
    )
    return canvas / (weights + 1e-12)


def test_probe_weighting_reduces_corrupted_overlap_seam_error():
    uniform = _assemble_two_patch_overlap(uniform_weighting=True)
    probe = _assemble_two_patch_overlap(uniform_weighting=False)

    seam = (slice(2, 10), slice(8, 10))
    expected = torch.ones((8, 2), dtype=torch.complex64)

    uniform_mae = torch.mean(torch.abs(uniform[seam] - expected))
    probe_mae = torch.mean(torch.abs(probe[seam] - expected))

    assert probe_mae < uniform_mae


# ---------------------------------------------------------------------------
# Task B4a -- barycentric hygiene: no silent patch drops at the canvas edge.
# ---------------------------------------------------------------------------

def test_accumulator_drops_patch_at_tight_canvas_bound():
    """Regression fixture for the drop this fix eliminates: at the TIGHT
    (unpadded) canvas bound, a patch centered at the true extreme offset
    fails the bounds check by exactly one pixel and is silently skipped (B4
    report Sec 4: 2/59 patches dropped on real data). Establishes the
    baseline defect before asserting the padded-canvas fix below."""
    device = torch.device("cpu")
    patch_size = 8
    middle_trim = patch_size
    max_offset = 10
    tight_canvas_shape = (middle_trim + 2 * max_offset, middle_trim + 2 * max_offset)
    canvas = torch.zeros(tight_canvas_shape, dtype=torch.complex64, device=device)
    weights = torch.zeros(tight_canvas_shape, dtype=torch.float32, device=device)
    accumulator = reassembly.VectorizedWeightedAccumulator(tight_canvas_shape, device)

    canvas_center = tight_canvas_shape[1] // 2
    extreme_position = torch.tensor([[canvas_center + max_offset, canvas_center]], dtype=torch.float32)
    patch = torch.ones((1, patch_size, patch_size), dtype=torch.complex64, device=device)
    probe_weight = torch.ones((patch_size, patch_size), dtype=torch.float32, device=device)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        accumulator.accumulate_batch(
            canvas, weights, patch, extreme_position, probe_weight,
            patch_size=patch_size, uniform_weighting=True,
        )

    assert weights.sum().item() == 0.0, "extreme-offset patch should be dropped at the tight bound"
    assert any("out-of-bounds" in str(w.message) for w in caught), "drop must warn loudly"


def test_padded_canvas_size_prevents_the_drop_at_the_same_extreme_offset():
    """The fix: reconstruct_image_barycentric now sizes its canvas via
    padded_canvas_size (one extra middle_trim of margin), which must make the
    identical extreme-offset patch from the test above land in-bounds --
    i.e. the drop is impossible on in-bounds coords, not merely less likely."""
    device = torch.device("cpu")
    patch_size = 8
    middle_trim = patch_size
    max_offset = 10
    padded_shape = reassembly.padded_canvas_size(middle_trim, max_offset, max_offset)
    canvas = torch.zeros(padded_shape, dtype=torch.complex64, device=device)
    weights = torch.zeros(padded_shape, dtype=torch.float32, device=device)
    accumulator = reassembly.VectorizedWeightedAccumulator(padded_shape, device)

    canvas_center = padded_shape[1] // 2
    extreme_position = torch.tensor([[canvas_center + max_offset, canvas_center]], dtype=torch.float32)
    patch = torch.full((1, patch_size, patch_size), 3 + 4j, dtype=torch.complex64, device=device)
    probe_weight = torch.ones((patch_size, patch_size), dtype=torch.float32, device=device)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        accumulator.accumulate_batch(
            canvas, weights, patch, extreme_position, probe_weight,
            patch_size=patch_size, uniform_weighting=True,
        )

    assert weights.sum().item() > 0.0, "extreme-offset patch must be stitched, not dropped, once padded"
    assert not any("out-of-bounds" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Task B4a -- barycentric hygiene: canvas anchor disclosure.
# ---------------------------------------------------------------------------

def test_build_canvas_anchor_reports_scan_com_and_correct_origin_offset():
    canvas_size = (204, 210)  # (H, W)
    scan_com = torch.tensor([160.0, 161.5])  # col0=x, col1=y

    anchor = reassembly.build_canvas_anchor(scan_com, canvas_size)

    torch.testing.assert_close(anchor["scan_com"], scan_com)
    assert anchor["canvas_shape"] == canvas_size
    expected_dx = canvas_size[1] // 2 - 160.0
    expected_dy = canvas_size[0] // 2 - 161.5
    dx, dy = anchor["canvas_origin_offset"]
    assert dx == pytest.approx(expected_dx)
    assert dy == pytest.approx(expected_dy)


def test_padded_canvas_size_adds_one_middle_trim_of_margin_per_dimension():
    h, w = reassembly.padded_canvas_size(middle_trim=32, max_offset_y=69, max_offset_x=73)

    assert h == 32 + 2 * 69 + 32
    assert w == 32 + 2 * 73 + 32


def test_varpro_probe_weighting_demo_outputs_expected_artifacts(tmp_path):
    output_root = tmp_path / "demo"
    subprocess.run(
        [
            "python",
            "scripts/studies/demo_varpro_probe_weighted_reassembly.py",
            "--output-root",
            str(output_root),
            "--seed",
            "0",
        ],
        check=True,
    )

    metrics = json.loads((output_root / "metrics.json").read_text())

    assert (output_root / "reconstruction_grid.png").exists()
    assert (output_root / "error_grid.png").exists()
    assert (output_root / "probe_weight_map.png").exists()
    assert metrics["probe_no_varpro"]["seam_mae"] < metrics["uniform_no_varpro"]["seam_mae"]
    assert metrics["uniform_varpro"]["complex_mae"] < metrics["uniform_no_varpro"]["complex_mae"]
    assert metrics["probe_varpro"]["complex_mae"] < metrics["probe_no_varpro"]["complex_mae"]
    assert metrics["probe_varpro"]["complex_mae"] <= metrics["uniform_varpro"]["complex_mae"]

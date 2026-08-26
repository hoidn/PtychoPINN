import torch
import pytest
from ptycho_torch.rect_scaling import (
    solve_rect_scales,
    accumulate_rect_basis,
    solve_from_state,
)


def _synthetic(s1, s2, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    psi_a = torch.randn(8, n, n, generator=g) + 1j * torch.randn(8, n, n, generator=g)
    psi_b = torch.randn(8, n, n, generator=g) + 1j * torch.randn(8, n, n, generator=g)
    I = (s1 * psi_a + s2 * psi_b).abs().square()
    A, B = psi_a.abs().square(), psi_b.abs().square()
    C = (psi_a.conj() * psi_b).real
    return A, C, B, I


def test_exact_recovery():
    A, C, B, I = _synthetic(3.2, 0.7)
    s1, s2 = solve_rect_scales(A, C, B, I)
    assert s1 == pytest.approx(3.2, rel=1e-4)
    assert s2 == pytest.approx(0.7, rel=1e-4)


def test_sign_convention_and_negative_s2():
    A, C, B, I = _synthetic(2.0, -1.5)
    s1, s2 = solve_rect_scales(A, C, B, I)
    assert s1 == pytest.approx(2.0, rel=1e-4)
    assert s2 == pytest.approx(-1.5, rel=1e-4)


def test_s1_zero_boundary_is_considered():
    A, C, B, I = _synthetic(0.0, 1.25, seed=29)
    s1, s2 = solve_rect_scales(A, C, B, I)
    assert s1 == pytest.approx(0.0, abs=1e-8)
    assert s2 == pytest.approx(1.25, rel=1e-4)


def test_streaming_accumulation_matches_batch_solve():
    # regenerate the complex fields so the streaming path sees genuine psi_a/psi_b
    g = torch.Generator().manual_seed(3)
    psi_a = torch.randn(8, 64, 64, generator=g) + 1j * torch.randn(8, 64, 64, generator=g)
    psi_b = torch.randn(8, 64, 64, generator=g) + 1j * torch.randn(8, 64, 64, generator=g)
    I = (9.6 * psi_a + 0.4 * psi_b).abs().square()
    state = None
    for k in range(0, 8, 2):
        sl = slice(k, k + 2)
        state = accumulate_rect_basis(psi_a[sl], psi_b[sl], I[sl], state)
    s1_stream, s2_stream = solve_from_state(state)
    A, B = psi_a.abs().square(), psi_b.abs().square()
    C = (psi_a.conj() * psi_b).real
    s1_batch, s2_batch = solve_rect_scales(A, C, B, I)
    assert s1_stream == pytest.approx(s1_batch, rel=1e-6)
    assert s2_stream == pytest.approx(s2_batch, rel=1e-6)


def test_rejects_nonfinite():
    A, C, B, I = _synthetic(1.0, 1.0)
    I[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        solve_rect_scales(A, C, B, I)


# ---------------------------------------------------------------------------
# Additional coverage (beyond the brief): weighting, dynamic range, degeneracy.
# ---------------------------------------------------------------------------


def test_weighted_recovery_matches_unweighted_on_noiseless_data():
    # On noiseless data the model is exact everywhere, so any positive
    # per-pixel weighting must recover the same (s1, s2).
    A, C, B, I = _synthetic(2.5, -0.9)
    g = torch.Generator().manual_seed(11)
    weights = torch.rand(I.shape, generator=g) + 0.1
    s1, s2 = solve_rect_scales(A, C, B, I, weights=weights)
    assert s1 == pytest.approx(2.5, rel=1e-4)
    assert s2 == pytest.approx(-0.9, rel=1e-4)


def test_large_dynamic_range_intensities():
    # Intensities spanning ~1e-3 .. 1e6: float64 accumulation must not lose
    # the fit. Scale psi so the summed intensity reaches count-scale.
    g = torch.Generator().manual_seed(7)
    psi_a = torch.randn(8, 64, 64, generator=g) + 1j * torch.randn(8, 64, 64, generator=g)
    psi_b = torch.randn(8, 64, 64, generator=g) + 1j * torch.randn(8, 64, 64, generator=g)
    psi_a = psi_a * 300.0
    psi_b = psi_b * 300.0
    s1_true, s2_true = 4.0, 1.7
    I = (s1_true * psi_a + s2_true * psi_b).abs().square()
    A, B = psi_a.abs().square(), psi_b.abs().square()
    C = (psi_a.conj() * psi_b).real
    s1, s2 = solve_rect_scales(A, C, B, I)
    assert s1 == pytest.approx(s1_true, rel=1e-4)
    assert s2 == pytest.approx(s2_true, rel=1e-4)


def test_noisy_correlated_basis_minimizes_original_residual():
    """The rank-one constraint must be solved in the residual metric.

    Euclidean projection of the unconstrained lifted solution is not the
    minimizer when the normal matrix is anisotropic.
    """
    g = torch.Generator().manual_seed(4019)
    psi_a = torch.randn(2, 4, 4, generator=g) + 1j * torch.randn(
        2, 4, 4, generator=g
    )
    independent = torch.randn(2, 4, 4, generator=g) + 1j * torch.randn(
        2, 4, 4, generator=g
    )
    psi_b = 0.99 * psi_a + (1.0 - 0.99**2) ** 0.5 * independent
    intensity = (
        (2.0 * psi_a + 0.7 * psi_b).abs().square()
        + torch.randn(2, 4, 4, generator=g)
    ).clamp_min(0)
    A = psi_a.abs().square()
    B = psi_b.abs().square()
    C = (psi_a.conj() * psi_b).real

    s1, s2 = solve_rect_scales(A, C, B, intensity)
    residual = intensity - (s1**2 * A + 2.0 * s1 * s2 * C + s2**2 * B)

    assert float(residual.square().sum()) < 15.22
    assert s1 == pytest.approx(2.215325, rel=2e-5)
    assert s2 == pytest.approx(0.487320, rel=2e-5)


def test_broadcast_shaped_inputs():
    # Batch-constant basis of shape (1, 8, 8) against measured intensities of
    # shape (4, 8, 8): the solver must broadcast, not require equal shapes.
    g = torch.Generator().manual_seed(13)
    psi_a = torch.randn(1, 8, 8, generator=g) + 1j * torch.randn(1, 8, 8, generator=g)
    psi_b = torch.randn(1, 8, 8, generator=g) + 1j * torch.randn(1, 8, 8, generator=g)
    s1_true, s2_true = 2.0, 1.0
    I = (s1_true * psi_a + s2_true * psi_b).abs().square().expand(4, 8, 8)
    A, B = psi_a.abs().square(), psi_b.abs().square()
    C = (psi_a.conj() * psi_b).real
    s1, s2 = solve_rect_scales(A, C, B, I)
    assert s1 == pytest.approx(s1_true, rel=1e-4)
    assert s2 == pytest.approx(s2_true, rel=1e-4)


def test_accumulate_rejects_nonfinite_at_ingest():
    # A bad batch must be identified when accumulated, not at solve time.
    g = torch.Generator().manual_seed(17)
    psi_a = torch.randn(2, 8, 8, generator=g) + 1j * torch.randn(2, 8, 8, generator=g)
    psi_b = torch.randn(2, 8, 8, generator=g) + 1j * torch.randn(2, 8, 8, generator=g)
    I = (psi_a + psi_b).abs().square()
    I[0, 0, 0] = float("inf")
    with pytest.raises(ValueError):
        accumulate_rect_basis(psi_a, psi_b, I)


def test_zero_psi_b_is_singular_and_raises():
    # Degenerate basis: psi_b == 0 makes both B and C vanish, so the (v, w)
    # directions carry no information and the 3x3 normal matrix is exactly
    # singular. Per the solver contract this fail-fast case raises ValueError.
    g = torch.Generator().manual_seed(5)
    psi_a = torch.randn(8, 64, 64, generator=g) + 1j * torch.randn(8, 64, 64, generator=g)
    psi_b = torch.zeros(8, 64, 64, dtype=torch.complex64)
    I = (2.75 * psi_a).abs().square()
    A, B = psi_a.abs().square(), psi_b.abs().square()
    C = (psi_a.conj() * psi_b).real
    with pytest.raises(ValueError):
        solve_rect_scales(A, C, B, I)


# ---------------------------------------------------------------------------
# Task B3: inference-time per-dataset (s1, s2) refit through run_torch_inference.
#
# These exercise the opt-in ``--rect-s1s2-refit dataset`` path end-to-end: the
# runner recomputes the VarPro basis from the SAME model physics used to plant
# the measured intensities, streams it through ``accumulate_rect_basis``, and
# solves once with ``solve_from_state`` (Eq. 8). The default ('off') path and
# the config guard are also pinned.
# ---------------------------------------------------------------------------


def _rect_scaler_single_dataset():
    """A real RectangularScaledDiffraction (num_datasets=1) for basis math."""
    from ptycho_torch.model import RectangularScaledDiffraction
    from ptycho_torch.config_params import ModelConfig

    return RectangularScaledDiffraction(ModelConfig(num_datasets=1))


def test_basis_images_matches_training_forward_identity():
    """Fidelity anchor: basis_images(x, probe_physical, 1.0) must reproduce the
    production autograd=True forward run with the training-time tensors
    (probe=probe_training, scale=norm), i.e. Sum_p |s1*Psi_a + s2*Psi_b|^2 ==
    RectangularScaledDiffraction.forward(...). This pins the
    P_eff = output_scale * probe_training == probe_physical identity
    (audit CI-SCALE-001) against the real forward, including P=2 incoherent
    mode summation -- independent of the refit path's own basis math."""
    rect = _rect_scaler_single_dataset()
    s1_true, s2_true = 3.0, 0.8
    with torch.no_grad():
        rect.s1.data.fill_(s1_true)
        rect.s2.data.fill_(s2_true)

    n, N, P = 3, 8, 2
    g = torch.Generator().manual_seed(4)
    textures = (
        torch.randn(n, 1, N, N, generator=g)
        + 1j * torch.randn(n, 1, N, N, generator=g)
    ).to(torch.complex64)
    probe_physical = (
        torch.randn(P, N, N, generator=g) + 1j * torch.randn(P, N, N, generator=g)
    ).to(torch.complex64)

    # Training-side tensors: normalize_probe_like_tf returns
    # (probe_physical/norm, probe_normalization=1/norm); compute_loss feeds the
    # CI forward output_scale = probe_normalization.reciprocal() = norm.
    norm = 2.5
    probe_training = probe_physical / norm
    output_scale = torch.full((n, 1, 1, 1), norm)

    with torch.no_grad():
        I_fwd = rect.forward(
            textures,
            None,  # I_raw: dead argument on the autograd=True path
            probe_training,
            output_scale,
            experiment_ids=torch.zeros(n, dtype=torch.long),
            autograd=True,
        )
    psi_a, psi_b = rect.basis_images(textures, probe_physical, 1.0)
    I_basis = (s1_true * psi_a + s2_true * psi_b).abs().square().sum(dim=2)

    torch.testing.assert_close(
        I_basis,
        I_fwd,
        rtol=1e-5,
        atol=1e-5 * float(I_fwd.abs().max()),
    )

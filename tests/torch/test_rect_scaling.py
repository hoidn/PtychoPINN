import copy

import numpy as np
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


class _FakeCIModel(torch.nn.Module):
    """Minimal CI inference stand-in: fixed textures + a real rect scaler.

    ``forward_predict`` ignores its input and returns pre-planted complex
    textures so the runner's refit sees exactly the (x_tilde) that generated
    the measured intensities. The genuine RectangularScaledDiffraction is
    registered as a real submodule under ``forward_model`` (an ``nn.Module``),
    so the runner's structural ``_find_rect_scaler`` (isinstance search over
    ``model.modules()``) locates it exactly as it does on a real nested
    ``PtychoPINN_Lightning`` -- the double no longer hides the Phase C bug.
    """

    def __init__(self, rect, textures):
        super().__init__()
        self.forward_model = torch.nn.Module()
        self.forward_model.rect_scaler = rect
        self._textures = textures
        self._cursor = 0

    def forward_predict(self, x, positions, probe, input_scale_factor):
        # Return this batch's planted textures (batched inference calls this
        # once per slice, in order).
        batch = self._textures[self._cursor:self._cursor + x.shape[0]]
        self._cursor += x.shape[0]
        return batch.to(x.device)


def _planted_ci_container(rect, s1_true, s2_true, *, n=4, N=8, seed=0,
                          real_only=False):
    """Build a prepared CI container whose measured intensity is exactly
    ``sum_p |s1* Psi_a + s2* Psi_b|^2`` for the given textures + probe."""
    g = torch.Generator().manual_seed(seed)
    if real_only:
        # Collapsed/degenerate texture: zero imaginary part => Psi_b == 0 =>
        # singular normal matrix at solve time.
        textures = torch.full((n, 1, N, N), 0.37, dtype=torch.complex64)
    else:
        textures = (
            torch.randn(n, 1, N, N, generator=g)
            + 1j * torch.randn(n, 1, N, N, generator=g)
        ).to(torch.complex64)
    probe_phys = (
        torch.randn(N, N, generator=g) + 1j * torch.randn(N, N, generator=g)
    ).to(torch.complex64)

    psi_a, psi_b = rect.basis_images(textures, probe_phys, 1.0)  # (n,1,1,N,N)
    measured_cf = (s1_true * psi_a + s2_true * psi_b).abs().square().sum(dim=2)
    measured_channel_last = measured_cf.permute(0, 2, 3, 1).contiguous()  # (n,N,N,1)

    container = {
        "X": measured_channel_last.numpy(),
        "measured_intensity": measured_channel_last,
        "probe_training": probe_phys,
        "probe_physical": probe_phys,
        "rms_input_scale": np.array(1.0, dtype=np.float32),
        "coords_relative": np.zeros((n, 1, 2, 1), dtype=np.float32),
    }
    return container, textures


def _ci_refit_cfg(tmp_path, refit="dataset", forward_mode="rectangular_scaled",
                  N=8, **extra):
    from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig
    from ptycho_torch.scaling_contract import CI_SCALE_CONTRACT, COUNT_INTENSITY

    return TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path,
        architecture="cnn",
        N=N,
        scale_contract_version=CI_SCALE_CONTRACT,
        measurement_domain=COUNT_INTENSITY,
        physics_forward_mode=forward_mode,
        rect_s1s2_refit=refit,
        **extra,
    )


def test_runner_config_rect_s1s2_refit_defaults_off():
    """Default is 'off' so the historical inference path is unchanged."""
    from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig

    cfg = TorchRunnerConfig(
        train_npz="train.npz",
        test_npz="test.npz",
        output_dir="out",
        architecture="cnn",
    )
    assert cfg.rect_s1s2_refit == "off"


def test_refit_guard_rejects_non_rectangular_config(tmp_path):
    """'dataset' with a non-rectangular (amplitude) config fails before inference."""
    from scripts.studies.grid_lines_torch_runner import run_torch_inference

    cfg = _ci_refit_cfg(tmp_path, refit="dataset", forward_mode="amplitude")
    model = _FakeCIModel(_rect_scaler_single_dataset(),
                         torch.zeros(1, 1, 8, 8, dtype=torch.complex64))
    with pytest.raises(ValueError, match="rect_s1s2_refit"):
        run_torch_inference(model, {"diffraction": np.zeros((1, 8, 8, 1))}, cfg)


def test_refit_dataset_recovers_planted_s1_s2(tmp_path):
    """End-to-end: the per-dataset refit recovers the planted (s1*, s2*).

    infer_batch_size=2 with n=4 patterns forces TWO accumulation batches, so
    the streaming state chaining and per-batch measured-intensity slice
    alignment are genuinely exercised.
    """
    from scripts.studies.grid_lines_torch_runner import run_torch_inference

    rect = _rect_scaler_single_dataset()
    s1_true, s2_true = 3.0, 0.8
    container, textures = _planted_ci_container(rect, s1_true, s2_true, seed=1)
    model = _FakeCIModel(rect, textures)
    cfg = _ci_refit_cfg(tmp_path, refit="dataset", infer_batch_size=2)

    preds = run_torch_inference(model, container, cfg)

    assert preds.shape[0] == 4  # inference still returns predictions
    block = getattr(model, "_rect_s1s2_refit")
    assert block["mode"] == "dataset"
    assert "error" not in block
    assert block["n_patterns"] == 4
    assert block["s1"] == pytest.approx(s1_true, rel=1e-3)
    assert block["s2"] == pytest.approx(s2_true, rel=1e-3)
    # c_A, c_phi are the paper's Eq. (6) contrast coordinates.
    assert block["c_A"] == pytest.approx((s1_true ** 2 + s2_true ** 2) ** 0.5, rel=1e-3)
    # The results block is the SOLE hand-off (B4 applies the pair
    # arithmetically): the model's scaler must stay at its init values so the
    # refit can never leak into checkpoints on any eval path.
    assert float(rect.s1.detach()[0]) == pytest.approx(1.0)
    assert float(rect.s2.detach()[0]) == pytest.approx(1.0)


def test_refit_dataset_degenerate_textures_record_error_and_continue(tmp_path):
    """A collapsed (constant real) texture is singular: the error is recorded
    in the block and inference still returns predictions."""
    from scripts.studies.grid_lines_torch_runner import run_torch_inference

    rect = _rect_scaler_single_dataset()
    container, textures = _planted_ci_container(
        rect, 2.0, 1.0, seed=2, real_only=True
    )
    model = _FakeCIModel(rect, textures)
    cfg = _ci_refit_cfg(tmp_path, refit="dataset")

    preds = run_torch_inference(model, container, cfg)

    assert preds.shape[0] == 4  # inference not killed by the failed refit
    block = getattr(model, "_rect_s1s2_refit")
    assert block["mode"] == "dataset"
    assert "error" in block and block["error"]
    assert "s1" not in block
    assert block["n_patterns"] == 4


def test_refit_dataset_multimode_probe_records_error_and_continues(tmp_path):
    """A mode-first (P>1) probe_physical is rejected at refit prep: streaming
    per-mode fields as independent samples is not the Eq. (5) mode-summed
    estimator, so the refit records the error block instead of fitting
    silently wrong values. Inference itself continues."""
    from scripts.studies.grid_lines_torch_runner import run_torch_inference

    rect = _rect_scaler_single_dataset()
    container, textures = _planted_ci_container(rect, 2.0, 1.0, seed=3)
    g = torch.Generator().manual_seed(23)
    container["probe_physical"] = (
        torch.randn(2, 8, 8, generator=g) + 1j * torch.randn(2, 8, 8, generator=g)
    ).to(torch.complex64)  # mode-first multi-mode stack (P=2, N, N)
    model = _FakeCIModel(rect, textures)
    cfg = _ci_refit_cfg(tmp_path, refit="dataset")

    preds = run_torch_inference(model, container, cfg)

    assert preds.shape[0] == 4
    block = getattr(model, "_rect_s1s2_refit")
    assert block["mode"] == "dataset"
    assert "multi-mode" in block["error"]
    assert "s1" not in block
    assert block["n_patterns"] == 0  # rejected before any accumulation


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


# ---------------------------------------------------------------------------
# Task B4: measurement-consistent reconstruction output (paper Eq. 1 applied to
# the reported reconstruction).
#
# The runner scales the STITCHED complex reconstruction by the active rect
# (s1, s2) -- Z_scaled = s1*Re(Z) + i*s2*Im(Z) -- emitting an additive
# ``YY_pred_scaled`` NPZ key, a labeled ``metrics_scaled`` block, and the
# ``scaled_recon`` (s1, s2, source) provenance. The (s1, s2) source is the
# error-free refit block when present, else the model's TRAINED scaler values;
# emission is skipped entirely when no rect scaler / valid pair exists.
# ---------------------------------------------------------------------------


def _write_recon_npz(tmp_path, recon):
    """Materialize a production-shaped recon.npz (YY_pred/amp/phase)."""
    from ptycho.workflows.grid_lines_workflow import save_recon_artifact

    return save_recon_artifact(tmp_path, "pinn", recon)


def _complex_field(shape, seed):
    g = np.random.default_rng(seed)
    return (g.standard_normal(shape) + 1j * g.standard_normal(shape)).astype(
        np.complex64
    )


def test_scaled_recon_emits_field_and_labeled_metrics(tmp_path):
    """Trained-source emission: hand-computed YY_pred_scaled, labeled
    metrics_scaled present, and existing metrics content byte-identical."""
    from scripts.studies.grid_lines_torch_runner import (
        _emit_scaled_recon,
        compute_metrics,
    )

    recon = _complex_field((16, 16), seed=0)
    gt = _complex_field((16, 16), seed=1)
    recon_path = _write_recon_npz(tmp_path, recon)

    metrics = compute_metrics(recon, gt, "pinn")
    control = copy.deepcopy(metrics)

    s1, s2 = 3.2, 0.7
    src = _emit_scaled_recon(
        refit_block=None,
        trained_rect_s1s2=(s1, s2),
        pred_for_metrics=recon,
        ground_truth=gt,
        label="pinn",
        recon_path=recon_path,
        metrics=metrics,
    )

    assert src == {"s1": s1, "s2": s2, "source": "trained"}

    with np.load(recon_path) as data:
        assert "YY_pred_scaled" in data.files
        yy = data["YY_pred"]
        expected = (s1 * yy.real + 1j * (s2 * yy.imag)).astype(np.complex64)
        np.testing.assert_array_equal(data["YY_pred_scaled"], expected)
        assert data["YY_pred_scaled"].shape == yy.shape

    # metrics_scaled is the full compute_metrics output on the scaled field.
    assert "metrics_scaled" in metrics
    expected_scaled = compute_metrics(
        (s1 * recon.real + 1j * (s2 * recon.imag)).astype(np.complex64), gt, "pinn"
    )
    np.testing.assert_equal(metrics["metrics_scaled"], expected_scaled)

    # Existing metrics content unchanged apart from the additive key.
    existing = {k: v for k, v in metrics.items() if k != "metrics_scaled"}
    np.testing.assert_equal(existing, control)


def test_scaled_recon_prefers_valid_refit_block(tmp_path):
    """A present, error-free refit block wins over the trained pair."""
    from scripts.studies.grid_lines_torch_runner import _emit_scaled_recon, compute_metrics

    recon = _complex_field((12, 12), seed=2)
    gt = _complex_field((12, 12), seed=3)
    recon_path = _write_recon_npz(tmp_path, recon)

    refit_block = {"mode": "dataset", "s1": 2.5, "s2": -1.1, "n_patterns": 4}
    metrics = compute_metrics(recon, gt, "pinn")
    src = _emit_scaled_recon(
        refit_block=refit_block,
        trained_rect_s1s2=(9.9, 9.9),  # must be ignored in favor of the refit pair
        pred_for_metrics=recon,
        ground_truth=gt,
        label="pinn",
        recon_path=recon_path,
        metrics=metrics,
    )

    assert src == {"s1": 2.5, "s2": -1.1, "source": "refit"}
    with np.load(recon_path) as data:
        yy = data["YY_pred"]
        expected = (2.5 * yy.real + 1j * (-1.1 * yy.imag)).astype(np.complex64)
        np.testing.assert_array_equal(data["YY_pred_scaled"], expected)


def test_scaled_recon_falls_back_to_trained_on_refit_error(tmp_path):
    """A refit block carrying an ``error`` falls back to the trained pair."""
    from scripts.studies.grid_lines_torch_runner import _emit_scaled_recon, compute_metrics

    recon = _complex_field((16, 16), seed=4)
    gt = _complex_field((16, 16), seed=5)
    recon_path = _write_recon_npz(tmp_path, recon)

    refit_block = {"mode": "dataset", "error": "singular normal matrix", "n_patterns": 4}
    metrics = compute_metrics(recon, gt, "pinn")
    src = _emit_scaled_recon(
        refit_block=refit_block,
        trained_rect_s1s2=(2.0, 0.5),
        pred_for_metrics=recon,
        ground_truth=gt,
        label="pinn",
        recon_path=recon_path,
        metrics=metrics,
    )

    assert src == {"s1": 2.0, "s2": 0.5, "source": "trained"}
    with np.load(recon_path) as data:
        yy = data["YY_pred"]
        expected = (2.0 * yy.real + 1j * (0.5 * yy.imag)).astype(np.complex64)
        np.testing.assert_array_equal(data["YY_pred_scaled"], expected)


def test_scaled_recon_skips_when_no_scaler(tmp_path):
    """No rect scaler and no valid refit (amplitude mode): emit nothing."""
    from scripts.studies.grid_lines_torch_runner import _emit_scaled_recon, compute_metrics

    recon = _complex_field((16, 16), seed=6)
    gt = _complex_field((16, 16), seed=7)
    recon_path = _write_recon_npz(tmp_path, recon)

    metrics = compute_metrics(recon, gt, "pinn")
    control = copy.deepcopy(metrics)

    src = _emit_scaled_recon(
        refit_block=None,
        trained_rect_s1s2=None,
        pred_for_metrics=recon,
        ground_truth=gt,
        label="pinn",
        recon_path=recon_path,
        metrics=metrics,
    )

    assert src is None
    assert "metrics_scaled" not in metrics
    np.testing.assert_equal(metrics, control)
    with np.load(recon_path) as data:
        assert "YY_pred_scaled" not in data.files


def test_scaled_recon_npz_failure_leaves_metrics_unmutated(tmp_path):
    """Atomicity: a failure during emission must not commit metrics_scaled."""
    from scripts.studies.grid_lines_torch_runner import _emit_scaled_recon, compute_metrics

    recon = _complex_field((16, 16), seed=8)
    gt = _complex_field((16, 16), seed=9)
    metrics = compute_metrics(recon, gt, "pinn")
    control = copy.deepcopy(metrics)

    missing = tmp_path / "no_such_dir" / "recon.npz"  # np.load will raise
    with pytest.raises(Exception):
        _emit_scaled_recon(
            refit_block=None,
            trained_rect_s1s2=(2.0, 0.5),
            pred_for_metrics=recon,
            ground_truth=gt,
            label="pinn",
            recon_path=missing,
            metrics=metrics,
        )

    assert "metrics_scaled" not in metrics
    np.testing.assert_equal(metrics, control)


def test_augment_npz_midwrite_failure_preserves_original(tmp_path, monkeypatch):
    """Crash-atomicity: a mid-write save failure (disk full / IO error) must
    not corrupt the primary recon.npz -- the augment writes a sibling temp file
    and only replaces the original after a complete write."""
    from scripts.studies.grid_lines_torch_runner import _augment_recon_npz_with_scaled

    recon = _complex_field((16, 16), seed=10)
    recon_path = _write_recon_npz(tmp_path, recon)
    original_bytes = recon_path.read_bytes()

    def _partial_write_then_fail(path, **arrays):
        # Simulate a mid-write failure: truncated garbage lands at the target
        # path before the error surfaces.
        with open(path, "wb") as fh:
            fh.write(b"PK\x03\x04 truncated npz")
        raise OSError("No space left on device")

    monkeypatch.setattr(np, "savez", _partial_write_then_fail)
    with pytest.raises(OSError):
        _augment_recon_npz_with_scaled(recon_path, 2.0, 0.5)

    # Primary artifact byte-identical and still loadable.
    assert recon_path.read_bytes() == original_bytes
    with np.load(recon_path) as data:
        np.testing.assert_array_equal(
            data["YY_pred"], np.squeeze(recon).astype(np.complex64)
        )
        assert "YY_pred_scaled" not in data.files
    # No temp-file debris left in the recon directory.
    assert [p.name for p in recon_path.parent.iterdir()] == ["recon.npz"]


def test_read_trained_rect_s1s2_from_model():
    """The trained (s1, s2) are read read-only from the model's rect scaler."""
    from scripts.studies.grid_lines_torch_runner import _read_trained_rect_s1s2

    rect = _rect_scaler_single_dataset()
    with torch.no_grad():
        rect.s1.data.fill_(4.0)
        rect.s2.data.fill_(1.7)
    model = _FakeCIModel(rect, torch.zeros(1, 1, 8, 8, dtype=torch.complex64))

    pair = _read_trained_rect_s1s2(model)
    assert pair == pytest.approx((4.0, 1.7))
    # Read-only: the scaler is not mutated.
    assert float(rect.s1.detach()[0]) == pytest.approx(4.0)
    assert float(rect.s2.detach()[0]) == pytest.approx(1.7)


def test_read_trained_rect_s1s2_none_for_amplitude_model():
    """A model without a rect scaler (amplitude mode) yields None."""
    import types

    from scripts.studies.grid_lines_torch_runner import _read_trained_rect_s1s2

    model = types.SimpleNamespace(forward_model=types.SimpleNamespace())
    assert _read_trained_rect_s1s2(model) is None
    assert _read_trained_rect_s1s2(types.SimpleNamespace()) is None


# ---------------------------------------------------------------------------
# Phase C regression (real PtychoPINN_Lightning module).
#
# The real Lightning module nests the scaler at
# ``model.model.forward_model.rect_scaler`` -- Lightning wraps the network under
# ``.model`` -- and exposes NO top-level ``.forward_model``. The prior
# attribute-path lookup assumed a top-level ``.forward_model``, so on a real
# checkpoint the B3 refit AttributeError'd (killing inference) and the B4
# trained-source read silently returned None. The ``_FakeCIModel`` double masked
# both because it put the scaler one hop below the top. These drive the REAL
# module (via the tripwires fixture) so the double can no longer hide the bug.
# ---------------------------------------------------------------------------


def _real_lightning_rect_model():
    """A real, uncalibrated PtychoPINN_Lightning under the CI rectangular_scaled
    contract (s1 == s2 == 1.0 at init), built by the tripwires fixture."""
    from tests.torch.test_rect_s1s2_initialization import _tiny_rect_scaled_module

    model, _batch = _tiny_rect_scaled_module()
    return model


def test_find_rect_scaler_locates_scaler_on_real_lightning_module():
    """(a) Structural resolution finds the scaler nested under ``.model``."""
    from scripts.studies.grid_lines_torch_runner import _find_rect_scaler
    from ptycho_torch.model import RectangularScaledDiffraction

    model = _real_lightning_rect_model()
    assert not hasattr(model, "forward_model")  # no top-level attribute path
    scaler = _find_rect_scaler(model)
    assert isinstance(scaler, RectangularScaledDiffraction)
    # It is the exact submodule the network owns (nested under .model).
    assert scaler is model.model.forward_model.rect_scaler


def test_read_trained_rect_s1s2_on_real_lightning_module():
    """(b) The trained pair is read from the nested scaler, not None.

    Pre-fix this returned None (``getattr(model, 'forward_model', None)`` misses
    the Lightning nesting), so B4 scaled emission was silently skipped."""
    from scripts.studies.grid_lines_torch_runner import _read_trained_rect_s1s2

    model = _real_lightning_rect_model()
    pair = _read_trained_rect_s1s2(model)
    assert pair is not None
    assert pair == pytest.approx((1.0, 1.0))  # fresh/uncalibrated init pair


def test_refit_dataset_runs_on_real_lightning_module(tmp_path):
    """(c) Load-bearing: ``--rect-s1s2-refit dataset`` yields a finite, non-error
    block on a REAL Lightning checkpoint.

    Pre-fix the basis lookup (``target_model.forward_model.rect_scaler``)
    AttributeError'd and killed inference. A helper scaler only plants a valid,
    positive measured-intensity target of the right shape (N=64, to match the
    real model); the refit fits the real model's OWN forward_predict textures
    against it, so specific (s1, s2) values are not asserted (random init)."""
    import math

    from scripts.studies.grid_lines_torch_runner import run_torch_inference

    model = _real_lightning_rect_model()
    rect = _rect_scaler_single_dataset()
    container, _textures = _planted_ci_container(rect, 3.0, 0.8, n=4, N=64, seed=1)
    cfg = _ci_refit_cfg(tmp_path, refit="dataset", N=64, infer_batch_size=2)

    preds = run_torch_inference(model, container, cfg)

    assert preds.shape[0] == 4  # inference still returns predictions
    block = getattr(model, "_rect_s1s2_refit")
    assert block["mode"] == "dataset"
    assert "error" not in block, block.get("error")
    assert block["n_patterns"] == 4
    assert math.isfinite(block["s1"]) and math.isfinite(block["s2"])

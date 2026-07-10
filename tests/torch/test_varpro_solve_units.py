"""VARPRO-SOLVE-UNITS-001: explicit legacy VarPro must compare
observed diffraction and basis images in the SAME units the training loss used.

Training (physics_forward_mode='rectangular_scaled') converges the object so
that ``I_obs ~= sum_p |F{output_scale * (s1*P*Re(O) + i*s2*P*Im(O))}|^2`` on the
FULL detector frame, with ``output_scale = sqrt(1/(probe_scaling^2 *
physics_scale + 1e-9))`` (ptycho_torch/model.py::PtychoPINN_Lightning.
compute_loss rect branch; RectangularScaledDiffraction.forward). The
inference-side solve in ``reassembly.reconstruct_image_barycentric`` must
therefore accumulate full-frame ``I_raw`` against a basis with that same
``output_scale`` folded in -- so that on a converged model it recovers
s1 ~= s2 ~= 1 instead of absorbing ``output_scale`` x center-crop distortion
(measured 5.49/22.49 on the mainparity_p1full/gs1_trainable checkpoint).
"""

import dataclasses

import pytest
import torch
from tensordict import TensorDict

from ptycho_torch import reassembly
from ptycho_torch.config_params import DataConfig, InferenceConfig, ModelConfig, TrainingConfig

# ---------------------------------------------------------------------------
# compute_varpro_basis: optional training-unit scale folding
# ---------------------------------------------------------------------------

def test_compute_varpro_basis_scale_matches_training_forward_convention():
    """``compute_varpro_basis(..., scale=k)`` must fold ``k`` into the exit
    waves exactly like the training forward (model.py:1395 ``exit_wave =
    scale * (s1*P*a + i*s2*P*b)``): Psi scales by k, basis images by k**2,
    with a per-batch (B,1,1,1) scale broadcast over modes."""
    torch.manual_seed(0)
    B, C, P, H, W = 2, 1, 3, 16, 16
    probe = torch.randn(B, C, P, H, W, dtype=torch.complex64)
    a_tilde = torch.randn(B, C, H, W)
    b_tilde = torch.randn(B, C, H, W)
    scale = torch.tensor([2.0, 24.0]).view(B, 1, 1, 1)

    Psi_a0, Psi_b0, X10, X20, X30 = reassembly.compute_varpro_basis(probe, a_tilde, b_tilde)
    Psi_a, Psi_b, X1, X2, X3 = reassembly.compute_varpro_basis(
        probe, a_tilde, b_tilde, scale=scale
    )

    k5 = scale.unsqueeze(2).to(torch.complex64)  # (B,1,1,1,1), model.py:1393 convention
    k2 = (scale ** 2)
    torch.testing.assert_close(Psi_a, k5 * Psi_a0)
    torch.testing.assert_close(Psi_b, k5 * Psi_b0)
    torch.testing.assert_close(X1, k2 * X10)
    torch.testing.assert_close(X2, k2 * X20)
    torch.testing.assert_close(X3, k2 * X30, atol=1e-3, rtol=1e-4)


def test_compute_varpro_basis_default_scale_is_identity():
    """Omitting ``scale`` must keep the historical basis byte-for-byte (the
    two pre-existing convention tests in test_varpro_probe_weighted_
    reassembly.py call it positionally)."""
    torch.manual_seed(1)
    probe = torch.randn(1, 1, 2, 16, 16, dtype=torch.complex64)
    a_tilde = torch.randn(1, 1, 16, 16)
    b_tilde = torch.randn(1, 1, 16, 16)

    out_default = reassembly.compute_varpro_basis(probe, a_tilde, b_tilde)
    out_none = reassembly.compute_varpro_basis(probe, a_tilde, b_tilde, scale=None)

    for got, want in zip(out_none, out_default):
        assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# End-to-end: the solve recovers KNOWN scalars from training-unit observations
# ---------------------------------------------------------------------------

_N = 64
_OUT_SCALE = 24.0  # matches the measured deadleaves output_scale (24.0357)


class _FixedTextureModel(torch.nn.Module):
    """Stub whose ``forward_predict`` returns a fixed complex texture stack,
    standing in for a converged real_imag model."""

    def __init__(self, textures: torch.Tensor):
        super().__init__()
        self.register_buffer("_textures", textures)

    def forward_predict(self, x, positions, probe, input_scale_factor):
        return self._textures[: x.shape[0]]


class _SyntheticVarProDataset:
    """Minimal PtychoDataset stand-in: TensorDictDataLoader.__iter__ only does
    ``dataset[batch_indices]`` -> ``(tensor_dict, probe, probe_scaling)``
    (dataloader.py:879-897, PtychoDataset.__getitem__ contract).

    ``com`` defaults to absent: center_of_mass is always derived from
    ``coords_global`` regardless of whether a ``'com'`` key is present (see
    reassembly.py's center-of-mass derivation), so ``data_dict`` need not
    carry one for these tests."""

    def __init__(self, td: TensorDict, probe: torch.Tensor, probe_scaling: torch.Tensor,
                 com: "torch.Tensor | None" = None):
        self.mmap_ptycho = td
        self.data_dict = {} if com is None else {"com": com}
        self.n_files = 1
        self._probe = probe
        self._probe_scaling = probe_scaling

    def __len__(self):
        return self.mmap_ptycho.batch_size[0]

    def __getitem__(self, idx):
        td = self.mmap_ptycho[idx]
        B = td.batch_size[0]
        probe = self._probe.unsqueeze(0).expand(B, -1, -1, -1, -1).clone()
        scaling = self._probe_scaling.expand(B, 1, 1, 1).clone()
        return td, probe, scaling


def _training_unit_intensity(probe, textures, s1, s2, out_scale):
    """Reference physics: RectangularScaledDiffraction.forward autograd=True
    (model.py:1395-1400) -- I = sum_p |F_ortho{out*(s1*P*a + i*s2*P*b)}|^2."""
    a = textures.real.unsqueeze(2)
    b = textures.imag.unsqueeze(2)
    exit_wave = out_scale * (s1 * (probe * a) + 1j * s2 * (probe * b))
    psi_f = torch.fft.fftshift(torch.fft.fft2(exit_wave, norm="ortho"), dim=(-2, -1))
    return torch.sum(torch.abs(psi_f) ** 2, dim=2)


def _build_synthetic_case(s1_true: float, s2_true: float, *, com: "torch.Tensor | None" = None):
    torch.manual_seed(7)
    n, C, P = 4, 1, 1

    yy, xx = torch.meshgrid(
        torch.arange(_N, dtype=torch.float32) - _N // 2,
        torch.arange(_N, dtype=torch.float32) - _N // 2,
        indexing="ij",
    )
    disk = torch.exp(-(xx ** 2 + yy ** 2) / (2 * (_N / 6) ** 2))
    probe = (disk * torch.exp(1j * 0.3 * xx / _N)).to(torch.complex64)
    probe = probe.view(1, 1, P, _N, _N).expand(C, C, P, _N, _N)[0:1]

    textures = (0.8 + 0.2 * torch.rand(n, C, _N, _N)) * torch.exp(
        1j * 0.4 * (torch.rand(n, C, _N, _N) - 0.5)
    )
    textures = textures.to(torch.complex64)

    probe_batch = probe.expand(n, C, P, _N, _N)
    I_obs = _training_unit_intensity(
        probe_batch, textures, s1_true, s2_true, _OUT_SCALE
    )

    coords = torch.tensor(
        [[96.0, 96.0], [104.0, 96.0], [96.0, 104.0], [104.0, 104.0]]
    ).view(n, C, 1, 2)

    td = TensorDict(
        {
            "images": I_obs,
            "coords_relative": torch.zeros(n, C, 1, 2),
            "coords_global": coords,
            "rms_scaling_constant": torch.ones(n, 1, 1, 1),
            # sqrt(1/(probe_scaling^2 * physics_scale + 1e-9)) == _OUT_SCALE
            "physics_scaling_constant": torch.full((n, 1, 1, 1), 1.0 / _OUT_SCALE ** 2),
        },
        batch_size=[n],
    )
    dataset = _SyntheticVarProDataset(td, probe[0], torch.ones(1, 1, 1, 1), com=com)
    model = _FixedTextureModel(textures)
    return model, dataset


def _run_reconstruction(model, dataset, *, varpro_scaling: bool,
                         physics_forward_mode: str = "rectangular_scaled",
                         verbose: bool = False):
    data_config = DataConfig(
        N=_N,
        C=1,
        scale_contract_version="legacy_v1",
        measurement_domain="normalized_amplitude",
    )
    model_config = ModelConfig(physics_forward_mode=physics_forward_mode)
    training_config = dataclasses.replace(TrainingConfig(), device="cpu", num_workers=1)
    inference_config = InferenceConfig(
        patch_weighting="uniform", varpro_scaling=varpro_scaling
    )
    return reassembly.reconstruct_image_barycentric(
        model, dataset, training_config, data_config, model_config, inference_config,
        gpu_ids=None, verbose=verbose, swap_detection="None", return_diagnostics=True,
    )


def test_varpro_solve_recovers_known_scalars_from_training_unit_observations():
    """Observations generated at KNOWN (s1, s2) = (1.3, 0.7) in the training
    count-unit convention: the solve must recover them. Pre-fix (units
    mismatch: cropped I_raw vs unscaled cropped basis) it instead recovers
    ~output_scale x crop-distorted scalars (order 10, the 5.49/22.49 defect
    measured on the real checkpoint)."""
    s1_true, s2_true = 1.3, 0.7
    model, dataset = _build_synthetic_case(s1_true, s2_true)

    _canvas, _subset, stats, _prescale = _run_reconstruction(
        model, dataset, varpro_scaling=True
    )
    s1, s2 = float(stats[4]), float(stats[5])

    assert s1 == pytest.approx(s1_true, rel=5e-2), f"s1={s1} (true {s1_true})"
    assert s2 == pytest.approx(s2_true, rel=5e-2), f"s2={s2} (true {s2_true})"


def test_varpro_solve_is_unit_for_converged_model_at_unit_scalars():
    """The Task VP acceptance shape: a model whose textures already explain the
    observations at s1=s2=1 must solve to ~1/~1 -- VarPro must not rescale an
    already-correct reconstruction."""
    model, dataset = _build_synthetic_case(1.0, 1.0)

    _canvas, _subset, stats, _prescale = _run_reconstruction(
        model, dataset, varpro_scaling=True
    )
    s1, s2 = float(stats[4]), float(stats[5])

    assert s1 == pytest.approx(1.0, rel=5e-2), f"s1={s1}"
    assert s2 == pytest.approx(1.0, rel=5e-2), f"s2={s2}"


def test_amplitude_mode_takes_scale_none_branch_matching_pre_fix_basis():
    """Amplitude-mode models (physics_forward_mode default 'amplitude') must
    NOT get the rectangular_scaled output_scale fold: the reviewer-required
    guard routes them through ``compute_varpro_basis(probe, a_tilde, b_tilde)``
    (``scale=None``), which is byte-identical to the pre-d755b2ae unscaled
    basis call (locked down by
    ``test_compute_varpro_basis_default_scale_is_identity`` above). Verify by
    recomputing the full-frame reference basis directly and comparing against
    the loop's returned Psi_a/Psi_b diagnostics."""
    model, dataset = _build_synthetic_case(1.0, 1.0)

    _canvas, _subset, stats, _prescale = _run_reconstruction(
        model, dataset, varpro_scaling=True, physics_forward_mode="amplitude",
    )
    Psi_a, Psi_b = stats[2], stats[3]

    # Recompute the reference (scale=None) basis directly from the same
    # full-frame probe/texture the loop consumed -- the whole dataset is a
    # single batch here (n=4, no dataloader batching split).
    td = dataset.mmap_ptycho
    n = td.batch_size[0]
    probe_full = dataset._probe.unsqueeze(0).expand(n, -1, -1, -1, -1)
    textures = model._textures
    Psi_a_ref, Psi_b_ref, _X1, _X2, _X3 = reassembly.compute_varpro_basis(
        probe_full, textures.real, textures.imag
    )

    torch.testing.assert_close(Psi_a, Psi_a_ref)
    torch.testing.assert_close(Psi_b, Psi_b_ref)


def test_novarpro_canvas_untouched_by_the_units_fix_path():
    """Byte-identity guard for the recorded-evidence basis: the
    varpro_scaling=False canvas must equal the varpro run's pre-rescale canvas
    exactly -- the solve only ever rescales AFTER stitching, so the units fix
    cannot perturb novarpro numerics."""
    model, dataset = _build_synthetic_case(1.3, 0.7)

    canvas_off, _s, _stats, prescale_off = _run_reconstruction(
        model, dataset, varpro_scaling=False
    )
    _canvas_on, _s2_, _stats_on, prescale_on = _run_reconstruction(
        model, dataset, varpro_scaling=True
    )

    assert torch.equal(canvas_off, prescale_off)
    assert torch.equal(prescale_off, prescale_on)


# ---------------------------------------------------------------------------
# Center-of-mass derivation: always from coords_global, never from a stored
# 'com' (reassembly.py's only production 'com' writer, dataloader.py::
# memory_map_data, unconditionally overwrites data_dict['com'] per file, so a
# multi-file dataset's stored value is a stale last-file centroid -- reading
# it would silently mis-center reconstruction; deriving from this subset's
# own coords_global is the only value that is always correct).
# ---------------------------------------------------------------------------

def test_comless_dataset_does_not_raise():
    """A dataset with no 'com' key in data_dict must reconstruct successfully
    (center_of_mass is always derived from coords_global)."""
    model, dataset = _build_synthetic_case(1.0, 1.0, com=None)

    _canvas, _subset, stats, _prescale = _run_reconstruction(
        model, dataset, varpro_scaling=True
    )

    assert stats[6]["scan_com"] is not None


def test_stored_com_is_ignored_in_favor_of_derived_coords():
    """Inertness pin (contract, not a bugfix regression): a dataset carrying a
    deliberately WRONG stored 'com' (true centroid + 50) must produce the
    identical derived center_of_mass as the comless dataset -- the stored
    value is never read."""
    true_coords_com = torch.mean(
        torch.tensor([[96.0, 96.0], [104.0, 96.0], [96.0, 104.0], [104.0, 104.0]]),
        dim=0,
    )
    wrong_com = true_coords_com + 50.0

    model_comless, dataset_comless = _build_synthetic_case(1.0, 1.0, com=None)
    _c1, _s1, stats_comless, _p1 = _run_reconstruction(
        model_comless, dataset_comless, varpro_scaling=True
    )

    model_wrong, dataset_wrong = _build_synthetic_case(1.0, 1.0, com=wrong_com)
    _c2, _s2, stats_wrong, _p2 = _run_reconstruction(
        model_wrong, dataset_wrong, varpro_scaling=True
    )

    torch.testing.assert_close(
        stats_comless[6]["scan_com"], stats_wrong[6]["scan_com"]
    )
    torch.testing.assert_close(stats_comless[6]["scan_com"], true_coords_com)


# ---------------------------------------------------------------------------
# Verbose "Scalars solved" print guard: must not crash when output_scale is
# None (amplitude mode) and must not reference an unbound local.
# ---------------------------------------------------------------------------

def test_verbose_scalars_solved_print_is_safe_when_output_scale_is_none(capsys):
    """Amplitude-mode models take the ``output_scale=None`` branch (no
    rectangular_scaled fold). The verbose 'Scalars solved' print must render
    a safe representation for it instead of crashing."""
    model, dataset = _build_synthetic_case(1.0, 1.0)

    _run_reconstruction(
        model, dataset, varpro_scaling=True, physics_forward_mode="amplitude",
        verbose=True,
    )

    out = capsys.readouterr().out
    assert "Scalars solved" in out
    assert "effective output_scale = None" in out

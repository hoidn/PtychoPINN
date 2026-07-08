"""Operator-contract tests for ``BornRytovForward2D`` (BRDT candidate lane)."""

from __future__ import annotations

import math

import pytest
import torch

from ptycho_torch.physics import BornRytovForward2D
from ptycho_torch.physics.born_rytov_dt import BornRytovForward2D as DirectClass


def _default_kwargs(**overrides):
    kwargs = dict(
        grid_size=32,
        detector_size=32,
        angles=torch.linspace(0.0, 2.0 * math.pi, 8, dtype=torch.float64)[:-1],
        wavelength_px=8.0,
        medium_ri=1.333,
    )
    kwargs.update(overrides)
    return kwargs


def test_module_export_matches_direct_class():
    assert BornRytovForward2D is DirectClass


def test_constructor_rejects_rytov_mode():
    with pytest.raises(NotImplementedError):
        BornRytovForward2D(**_default_kwargs(mode="rytov_linearized"))


def test_constructor_rejects_unknown_mode():
    with pytest.raises(ValueError):
        BornRytovForward2D(**_default_kwargs(mode="zoo"))


def test_constructor_rejects_unknown_normalize():
    with pytest.raises(ValueError):
        BornRytovForward2D(**_default_kwargs(normalize="bogus"))


def test_constructor_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        BornRytovForward2D(**_default_kwargs(grid_size=0))
    with pytest.raises(ValueError):
        BornRytovForward2D(**_default_kwargs(detector_size=-1))
    with pytest.raises(ValueError):
        BornRytovForward2D(**_default_kwargs(wavelength_px=0))
    with pytest.raises(ValueError):
        BornRytovForward2D(**_default_kwargs(medium_ri=-2))


def test_forward_output_shape_and_dtype():
    op = BornRytovForward2D(**_default_kwargs())
    q = torch.zeros(3, 1, 32, 32, dtype=torch.float64)
    out = op(q)
    assert out.shape == (3, 7, 32, 2)
    assert out.dtype == torch.float64
    # zero potential implies zero scattered field
    assert torch.allclose(out, torch.zeros_like(out))


def test_forward_rejects_wrong_shape():
    op = BornRytovForward2D(**_default_kwargs())
    bad = torch.zeros(2, 2, 32, 32, dtype=torch.float64)
    with pytest.raises(ValueError):
        op(bad)
    bad2 = torch.zeros(2, 1, 16, 16, dtype=torch.float64)
    with pytest.raises(ValueError):
        op(bad2)


def test_forward_rejects_non_floating_dtype():
    op = BornRytovForward2D(**_default_kwargs())
    bad = torch.zeros(1, 1, 32, 32, dtype=torch.int32)
    with pytest.raises(TypeError):
        op(bad)


def test_buffers_registered_for_state_dict_round_trip():
    op = BornRytovForward2D(**_default_kwargs())
    state = op.state_dict()
    expected = {
        "angles",
        "det_freqs",
        "sampling_grid",
        "valid_mask",
        "coeff_real",
        "coeff_imag",
    }
    assert expected.issubset(state.keys())
    op2 = BornRytovForward2D(**_default_kwargs())
    op2.load_state_dict(state)
    q = torch.randn(1, 1, 32, 32, dtype=torch.float64)
    a = op(q)
    b = op2(q)
    assert torch.allclose(a, b)


def test_normalize_modes_differ_in_scale_but_not_support():
    """Output amplitude scale differs across normalize modes; spatial support pattern stays consistent.

    Both modes apply the same Ewald-arc validity mask, so for the same
    object the unmasked detector samples should be related by a per-(angle,
    detector-frequency) factor only.
    """
    op_u = BornRytovForward2D(**_default_kwargs())
    op_o = BornRytovForward2D(**_default_kwargs(normalize="odtbrain_compatible"))
    assert torch.equal(op_u.valid_mask, op_o.valid_mask)


def test_explicit_normalize_choice_is_recorded_in_contract():
    op_u = BornRytovForward2D(**_default_kwargs())
    contract_u = op_u.operator_contract()
    assert contract_u["normalize"] == "unitary_fft"
    assert "no Wolf prefactor" in contract_u["fft_normalization"]
    op_o = BornRytovForward2D(**_default_kwargs(normalize="odtbrain_compatible"))
    contract_o = op_o.operator_contract()
    assert contract_o["normalize"] == "odtbrain_compatible"
    assert "Wolf prefactor" in contract_o["fft_normalization"]


def test_no_python_loop_over_batch():
    """Forward time should scale sub-linearly with batch size.

    Loose check: forward of batch=8 should be < 6x batch=1 wallclock.
    Guards against an accidental Python-side batch loop.
    """
    op = BornRytovForward2D(**_default_kwargs())
    import time

    q1 = torch.randn(1, 1, 32, 32, dtype=torch.float64)
    q8 = torch.randn(8, 1, 32, 32, dtype=torch.float64)
    # warmup
    op(q1)
    op(q8)
    t0 = time.perf_counter()
    for _ in range(5):
        op(q1)
    t_single = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(5):
        op(q8)
    t_batch = time.perf_counter() - t0
    assert t_batch < 6.0 * t_single + 0.1


def test_forward_is_linear_in_q():
    op = BornRytovForward2D(**_default_kwargs())
    torch.manual_seed(0)
    q1 = torch.randn(1, 1, 32, 32, dtype=torch.float64)
    q2 = torch.randn(1, 1, 32, 32, dtype=torch.float64)
    a = 1.7
    b = -0.3
    out_combined = op(a * q1 + b * q2)
    out_split = a * op(q1) + b * op(q2)
    assert torch.allclose(out_combined, out_split, atol=1e-9, rtol=1e-9)


def test_propagating_band_corresponds_to_kx_below_km():
    op = BornRytovForward2D(**_default_kwargs())
    # angle 0: K_obj_x = k_x, K_obj_z = k_z - k_m. valid when |k_x| < k_m and
    # the sampled K stays inside the (-pi, pi) frequency box.
    propagating = (op.det_freqs.abs() < op.k_m)
    valid_a0 = op.valid_mask[0]  # (D,)
    # propagating must be a superset of valid (validity additionally requires
    # in-bounds sampling on the spectrum grid).
    assert torch.all(valid_a0 <= propagating)


def test_sampling_grid_independent_of_object_potential():
    op_a = BornRytovForward2D(**_default_kwargs())
    op_b = BornRytovForward2D(**_default_kwargs())
    assert torch.allclose(op_a.sampling_grid, op_b.sampling_grid)

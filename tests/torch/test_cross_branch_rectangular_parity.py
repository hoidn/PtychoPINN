"""Task 2.8 (B-verify): cross-branch rectangular_scaled parity verification.

Verification capstone for merging main's VarPro/rectangular-scaled forward into
``fno-stable``. Distinct from the Task 2.6 *acceptance* oracle
(``test_rectangular_scaled_forward.py``): this module records the *cross-branch
verification result* -- it quantifies where the ported forward is numerically
equivalent to the frozen fixtures under the registered cross-build tolerance
versus where the only remaining residual lives (the out-of-scope
reassembly-canvas padding divergence).

Three claims, each an independent verification gate:

1. ``c1_bigF`` (object_big=False) reproduces the frozen ``expected_forward``
   intensity AND both frozen loss scalars within ``rtol=1e-5, atol=1e-6`` under
   fno-stable's *real* ``get_padded_size`` (buffer=0) -- object_big=False never
   touches the reassembly canvas, so padding is irrelevant to the parity
   contract. This is the hard proof that the B5 rectangular port is numerically
   correct on this branch.

2. Mode-forcing is TWO knobs, not one (amendment #16). The frozen fixtures'
   ``metadata_json`` carries no ``physics_forward_mode`` key, so a naive
   config-rebuild defaults it to ``'amplitude'`` and never exercises the
   rectangular path. Verifying rectangular parity requires forcing
   ``physics_forward_mode='rectangular_scaled'`` AND confirming the fixture's own
   ``training_patch_weighting`` (``'probe'`` for the probe fixtures) survives the
   rebuild rather than silently falling back to a default.

3. The object_big (bigT) fixtures do not replay under fno-stable's real padding
   at the frozen values -- they were frozen under main's ``get_padded_size(
   buffer=max_position_jitter)`` while fno-stable deliberately uses ``buffer=0``
   (commit ba3f705d, shared by the amplitude + rectangular reassembly paths;
   reverting it would change the amplitude default and break the Task 2.1 pin).
   This test records that the real-padding residual is finite/bounded and proves
   it falls within the registered cross-build tolerance once main's padding is
   restored (matched-padding monkeypatch) -- isolating the remaining gap to the
   reassembly-canvas size, a physics-reconciliation BACKLOG item (Task 2.9),
   not a rectangular-port defect.

Fixtures: ``tests/fixtures/varpro_parity/c*.npz`` (frozen on varpro-ablation).
This file imports only ``ptycho_torch`` public modules + torch/numpy/pytest so it
stays branch-portable.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ptycho_torch.config_params import DataConfig, ModelConfig
from ptycho_torch.model import (
    ForwardModel,
    RectangularPoissonLoss,
    RectangularMAELoss,
)
import ptycho_torch.helper as hh

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "varpro_parity"
FROZEN_RTOL = 1e-5
FROZEN_ATOL = 1e-6


def _assert_frozen_close(actual, expected):
    torch.testing.assert_close(
        actual, expected, rtol=FROZEN_RTOL, atol=FROZEN_ATOL)


def _fixture_paths():
    if not FIXTURE_DIR.is_dir():
        return []
    out = []
    for path in sorted(FIXTURE_DIR.glob("c*.npz")):
        with np.load(path) as npz:
            if "metadata_json" in npz.files:
                out.append(path)
    return out


def _init_field_names(cls):
    return {f.name for f in dataclasses.fields(cls) if f.init}


def _filtered_kwargs(stored, cls):
    return {k: v for k, v in stored.items() if k in _init_field_names(cls)}


def _rebuild_configs(metadata):
    """Portable rebuild that FORCES the rectangular mode (amendment #16).

    Only ``physics_forward_mode`` is injected; every other field -- crucially
    ``training_patch_weighting`` -- comes straight from the fixture's own stored
    metadata so the second knob is verified to survive, not hardcoded here.
    """
    data_cfg = DataConfig(**_filtered_kwargs(metadata["data_config"], DataConfig))
    model_kwargs = _filtered_kwargs(metadata["model_config"], ModelConfig)
    model_kwargs["physics_forward_mode"] = "rectangular_scaled"
    return data_cfg, ModelConfig(**model_kwargs)


def _rebuild_probe(npz, model_cfg, B, C):
    probe_2d = torch.from_numpy(npz["probe_real"] + 1j * npz["probe_imag"]).to(torch.complex64)
    N = probe_2d.shape[-1]
    return probe_2d.view(1, 1, 1, N, N).expand(B, C, 1, N, N)


def _load_fixture(fixture_path):
    npz = np.load(fixture_path)
    metadata = json.loads(str(npz["metadata_json"]))
    data_cfg, model_cfg = _rebuild_configs(metadata)
    B, C = metadata["shapes"]["B"], metadata["shapes"]["C"]
    t = dict(
        x=torch.from_numpy(npz["x_real"] + 1j * npz["x_imag"]).to(torch.complex64),
        positions=torch.from_numpy(npz["positions"]).to(torch.float32),
        scale=torch.from_numpy(npz["scale"]).to(torch.float32),
        experiment_ids=torch.from_numpy(npz["experiment_ids"]).to(torch.long),
        I_raw=torch.from_numpy(npz["I_raw"]).to(torch.float32),
        expected_forward=torch.from_numpy(npz["expected_forward"]).to(torch.float32),
        probe=_rebuild_probe(npz, model_cfg, B, C),
        expected_poisson=torch.tensor(float(npz["expected_poisson_loss"])),
        expected_mae=torch.tensor(float(npz["expected_mae_loss"])),
    )
    return data_cfg, model_cfg, t


def _run_forward(data_cfg, model_cfg, t):
    model = ForwardModel(model_cfg, data_cfg)
    model.eval()
    with torch.no_grad():
        return model.forward(
            t["x"], None, t["positions"], t["probe"], t["scale"],
            experiment_ids=t["experiment_ids"],
        )


_SKIP = pytest.mark.skipif(
    not _fixture_paths(),
    reason="tests/fixtures/varpro_parity has no c*.npz fixtures yet",
)


@_SKIP
def test_cross_build_tolerance_accepts_roundoff_and_rejects_material_drift():
    """The frozen cross-build contract accepts roundoff but rejects drift."""
    path = FIXTURE_DIR / "c1_bigF.npz"
    assert path.exists(), "c1_bigF fixture is required for the tolerance boundary"
    data_cfg, model_cfg, t = _load_fixture(path)

    out = _run_forward(data_cfg, model_cfg, t)
    poisson = RectangularPoissonLoss()(out, t["I_raw"]).mean()
    mae = RectangularMAELoss()(out, t["I_raw"]).mean()

    _assert_frozen_close(out, t["expected_forward"])
    _assert_frozen_close(poisson, t["expected_poisson"])
    _assert_frozen_close(mae, t["expected_mae"])

    expected_forward = t["expected_forward"]
    forward_bound = FROZEN_ATOL + FROZEN_RTOL * expected_forward.abs()
    forward_within = expected_forward + 0.5 * forward_bound
    forward_outside = expected_forward.clone()
    forward_outside.reshape(-1)[1] = (
        expected_forward.reshape(-1)[1] + 100.0 * forward_bound.reshape(-1)[1]
    )

    _assert_frozen_close(forward_within, expected_forward)
    with pytest.raises(AssertionError):
        _assert_frozen_close(forward_outside, expected_forward)

    for expected in (t["expected_poisson"], t["expected_mae"]):
        bound = FROZEN_ATOL + FROZEN_RTOL * expected.abs()
        within = expected + 0.5 * bound
        outside = expected + 100.0 * bound

        _assert_frozen_close(within, expected)
        with pytest.raises(AssertionError):
            _assert_frozen_close(outside, expected)


@_SKIP
def test_c1_bigF_rectangular_close_under_real_padding():
    """HARD PROOF: object_big=False fixture matches within the frozen tolerance."""
    path = FIXTURE_DIR / "c1_bigF.npz"
    assert path.exists(), "c1_bigF fixture is required for the cross-branch hard proof"
    data_cfg, model_cfg, t = _load_fixture(path)
    assert model_cfg.object_big is False
    assert model_cfg.physics_forward_mode == "rectangular_scaled"

    out = _run_forward(data_cfg, model_cfg, t)
    _assert_frozen_close(out, t["expected_forward"])
    poisson = RectangularPoissonLoss()(out, t["I_raw"]).mean()
    mae = RectangularMAELoss()(out, t["I_raw"]).mean()
    _assert_frozen_close(poisson, t["expected_poisson"])
    _assert_frozen_close(mae, t["expected_mae"])


@_SKIP
@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=lambda p: p.stem)
def test_two_knob_mode_forcing_preserves_stored_weighting(fixture_path):
    """Amendment #16: forcing physics_forward_mode='rectangular_scaled' must NOT
    clobber the fixture's own training_patch_weighting; the probe fixtures must
    still rebuild with 'probe'."""
    npz = np.load(fixture_path)
    metadata = json.loads(str(npz["metadata_json"]))
    _, model_cfg = _rebuild_configs(metadata)

    assert model_cfg.physics_forward_mode == "rectangular_scaled"
    stored_weighting = metadata["model_config"]["training_patch_weighting"]
    assert model_cfg.training_patch_weighting == stored_weighting, (
        "config rebuild silently dropped/overrode the stored training_patch_weighting"
    )
    if "probe" in fixture_path.stem:
        assert model_cfg.training_patch_weighting == "probe"


@_SKIP
@pytest.mark.parametrize(
    "fixture_path",
    [p for p in _fixture_paths() if "bigT" in p.stem],
    ids=lambda p: p.stem,
)
def test_bigT_residual_is_padding_only_and_matches_under_matched_padding(fixture_path, monkeypatch):
    """BACKLOG isolation: under fno-stable real padding the object_big residual is
    finite/bounded (documented, not a divergence-to-NaN); once main's
    get_padded_size(buffer=max_position_jitter) is restored it falls within the
    registered cross-build tolerance -- isolating the remaining gap to the
    reassembly-canvas size (commit ba3f705d), a Task 2.9 reconciliation item, not
    a rectangular-port defect."""
    data_cfg, model_cfg, t = _load_fixture(fixture_path)
    assert model_cfg.object_big is True

    # (a) real fno-stable padding: residual must be finite (bounded), documenting
    #     that the port does not blow up -- the only effect is a canvas-size shift.
    out_real = _run_forward(data_cfg, model_cfg, t)
    assert torch.isfinite(out_real).all()
    if out_real.shape == t["expected_forward"].shape:
        real_resid = (out_real - t["expected_forward"]).abs().max().item()
        assert np.isfinite(real_resid)

    # (b) matched (main) padding: cross-build-tolerant reproduction.
    _orig_bigN = hh.get_bigN
    monkeypatch.setattr(
        hh, "get_padded_size",
        lambda dc, mc: _orig_bigN(dc, mc) + mc.max_position_jitter,
    )
    out_matched = _run_forward(data_cfg, model_cfg, t)
    _assert_frozen_close(out_matched, t["expected_forward"])
    poisson = RectangularPoissonLoss()(out_matched, t["I_raw"]).mean()
    mae = RectangularMAELoss()(out_matched, t["I_raw"]).mean()
    _assert_frozen_close(poisson, t["expected_poisson"])
    _assert_frozen_close(mae, t["expected_mae"])

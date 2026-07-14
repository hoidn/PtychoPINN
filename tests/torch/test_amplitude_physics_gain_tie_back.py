"""Quantitative tie-back: amplitude_physics_gain=16 reproduces the measured
flat-layout accident (PROBE-RANK-001; design 2026-07-12 §8 case 3).

At the sealed rung0 reference weights on the deterministic first raster
batch of the sealed rung1e mmap work dir (real staged N128 data, CPU), the
Task 21 step-parity diagnostics measured: documented probe rank -> training
loss 0.9361-0.9366; flat (B, H, W) rank (the accidental x16 broadcast) ->
0.0732-0.0755 (task-21a report "0.9361 -> 0.0755"; task-21c review
independent reproduction 0.936635 -> 0.073175 on exactly this batch).

This test ties the explicit mechanism to the measured accident: under the
DOCUMENTED rank, gain 1 must land on the 0.9366 point and gain 16 must land
in the flat-layout band — because for per-batch identical probes the flat
broadcast was exactly a x16 amplitude gain. Loss bands (not bit-exact pins)
absorb cross-machine CPU float variation; the measured effect is 12.8x, far
above the ~2e-5 cross-process noise floor.

Skips when the sealed artifacts are absent (they live under git-ignored
.artifacts/ and are READ-ONLY; this test only loads them).
"""

from pathlib import Path

import pytest
import torch

from scripts.studies.ablation.runtime_ladder_step_parity_cli import (
    REFERENCE_CHECKPOINT,
    REFERENCE_EVIDENCE,
    RUNG1E_MEMMAP_ROOT,
    load_first_raster_batch,
    load_reference_model,
    loss_with_gain,
)

_REPO = Path(__file__).resolve().parents[2]
_CKPT = _REPO / REFERENCE_CHECKPOINT
_EVIDENCE = _REPO / REFERENCE_EVIDENCE
_MEMMAP = _REPO / RUNG1E_MEMMAP_ROOT

pytestmark = [
    pytest.mark.torch,
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_CKPT.exists() and _EVIDENCE.exists() and (_MEMMAP / "train" / "meta.json").exists()),
        reason="sealed rung0 checkpoint / rung1e staged batch artifacts not present",
    ),
]

# Measured bands (see module docstring for provenance). The gain-1 band pins
# the corrected physics; the gain-16 band is the pre-fix flat-layout loss on
# this exact batch and weights.
GAIN1_BAND = (0.93, 0.945)
GAIN16_BAND = (0.070, 0.077)


@pytest.fixture(scope="module")
def sealed_setup():
    model = load_reference_model(_CKPT, _EVIDENCE)
    batch = load_first_raster_batch(_MEMMAP, batch_size=16)
    return model, batch


def test_documented_rank_gain1_matches_corrected_physics_loss(sealed_setup):
    model, batch = sealed_setup
    loss = loss_with_gain(model, batch.as_batch(), 1.0)
    assert GAIN1_BAND[0] < loss < GAIN1_BAND[1], (
        f"documented-rank gain-1 loss {loss:.6f} outside measured band "
        f"{GAIN1_BAND} (review reproduction: 0.936635)"
    )


def test_gain16_reproduces_flat_layout_loss_band(sealed_setup):
    """The tie-back proper: the explicit, batch-size-independent gain 16
    under the documented rank reproduces the accidental flat-layout loss
    (0.9361 -> ~0.0755 step-parity isolation; 0.073175 on this batch)."""
    model, batch = sealed_setup
    loss = loss_with_gain(model, batch.as_batch(), 16.0)
    assert GAIN16_BAND[0] < loss < GAIN16_BAND[1], (
        f"documented-rank gain-16 loss {loss:.6f} outside the flat-layout "
        f"band {GAIN16_BAND} (review reproduction: 0.073175)"
    )


def test_gain16_moves_loss_by_order_of_magnitude(sealed_setup):
    """Structure check independent of absolute bands: the explicit gain must
    move the fixed-weights loss by ~12.8x, the measured accident magnitude."""
    model, batch = sealed_setup
    loss_1 = loss_with_gain(model, batch.as_batch(), 1.0)
    loss_16 = loss_with_gain(model, batch.as_batch(), 16.0)
    assert loss_1 / loss_16 > 10.0


def test_flat_layout_is_banned_on_the_sealed_batch(sealed_setup):
    """The accident itself is no longer runnable: the same batch with the
    probe flattened to (B, H, W) fails fast."""
    from ptycho_torch.model import ProbeLayoutError

    model, batch = sealed_setup
    fields, probe, scale = batch.as_batch()
    flat = probe.reshape(probe.shape[0], probe.shape[-2], probe.shape[-1])
    with pytest.raises(ProbeLayoutError):
        with torch.no_grad():
            model.compute_loss((fields, flat, scale))

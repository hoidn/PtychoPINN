"""Probe-rank physics contract enforcement (PROBE-RANK-001).

``ProbeIllumination.forward`` computes ``x.unsqueeze(2) * probe``; any probe
below rank 5 right-align-broadcasts so its leading axis lands in the MODE
slot. A legacy flat ``(B, H, W)`` probe therefore becomes ``(1, 1, B, H, W)``
-> illuminated ``(B, C, P=B, H, W)`` and ``pad_and_diffract``'s coherent mode
sum multiplies the predicted field by B (silent batch-size-dependent physics
gain), and, for per-sample-distinct probes, MIXES different samples' fields.

Contract under test (design 2026-07-12 "probe-rank physics contract fix",
docs/specs/spec-ptycho-torch-probe-layout.md, docs/findings.md
PROBE-RANK-001): ``probe.ndim == 5``, ``probe.shape[-2:] == (N, N)``,
``probe.shape[0] in (1, B)``, ``probe.shape[1] in (1, C)``; any violation
raises the typed, module-level ``ProbeLayoutError``. Behavior for conforming
inputs (the mmap ``(B, 1, 1, N, N)`` emission) is bit-unchanged.
"""

import pytest
import torch

from ptycho_torch.config_params import DataConfig, ModelConfig

N = 8


def _layer():
    from ptycho_torch.model import ProbeIllumination

    return ProbeIllumination(ModelConfig(), DataConfig(N=N, C=1, grid_size=(1, 1)))


def _x(batch=4, channels=1):
    torch.manual_seed(0)
    return (
        torch.randn(batch, channels, N, N) + 1j * torch.randn(batch, channels, N, N)
    ).to(torch.complex64)


def _probe(*shape):
    torch.manual_seed(1)
    return (torch.randn(*shape) + 1j * torch.randn(*shape)).to(torch.complex64)


def _rectangular_forward_model():
    from ptycho_torch.model import ForwardModel

    model_config = ModelConfig(
        object_big=False,
        C_model=1,
        C_forward=1,
        physics_forward_mode="rectangular_scaled",
    )
    data_config = DataConfig(N=N, C=1, grid_size=(1, 1))
    return ForwardModel(model_config, data_config)


@pytest.mark.torch
class TestProbeLayoutErrorMatrix:
    """RED case 1 (+5a): every sub-rank-5 / malformed layout fails fast."""

    def test_flat_batch_probe_raises(self):
        """The legacy dictionary-flow layout: (B, H, W). This is the exact
        tensor whose broadcast produced the accidental xB amplitude gain."""
        from ptycho_torch.model import ProbeLayoutError

        with pytest.raises(ProbeLayoutError, match=r"\(4, 8, 8\)"):
            _layer()(_x(batch=4), _probe(4, N, N))

    def test_rank2_probe_raises(self):
        from ptycho_torch.model import ProbeLayoutError

        with pytest.raises(ProbeLayoutError):
            _layer()(_x(), _probe(N, N))

    def test_rank4_probe_raises(self):
        from ptycho_torch.model import ProbeLayoutError

        with pytest.raises(ProbeLayoutError):
            _layer()(_x(), _probe(4, 1, N, N))

    def test_multi_probe_flat_batch_raises(self):
        """Design case 5a: per-sample-DISTINCT probes under the flat layout
        would coherently mix different samples' fields (the 2a9ee2ad9 review
        demonstrated sample 0's output changes when sample 3's probe is
        perturbed). The contract must ban the layout, not tolerate it where
        it happens to be a pure gain."""
        from ptycho_torch.model import ProbeLayoutError

        distinct = torch.stack(
            [(i + 1.0) * _probe(N, N) for i in range(4)]
        )  # (B, H, W), rows genuinely different
        with pytest.raises(ProbeLayoutError):
            _layer()(_x(batch=4), distinct)

    def test_wrong_spatial_dims_raise(self):
        from ptycho_torch.model import ProbeLayoutError

        with pytest.raises(ProbeLayoutError):
            _layer()(_x(), _probe(4, 1, 1, N, 2 * N))

    def test_batch_axis_mismatch_raises(self):
        from ptycho_torch.model import ProbeLayoutError

        with pytest.raises(ProbeLayoutError):
            _layer()(_x(batch=4), _probe(3, 1, 1, N, N))

    def test_channel_axis_mismatch_raises(self):
        from ptycho_torch.model import ProbeLayoutError

        with pytest.raises(ProbeLayoutError):
            _layer()(_x(batch=4, channels=2), _probe(4, 3, 1, N, N))

    def test_error_names_contract_and_finding(self):
        """Fail-fast message must point violators at the documented layout
        and its provenance so any remaining flat-probe producer stops loudly
        with a migration pointer."""
        from ptycho_torch.model import ProbeLayoutError

        with pytest.raises(ProbeLayoutError) as excinfo:
            _layer()(_x(batch=4), _probe(4, N, N))
        message = str(excinfo.value)
        assert "(B, C, P, H, W)" in message
        assert "PROBE-RANK-001" in message


@pytest.mark.torch
class TestConformingLayouts:
    """GREEN half of design case 5: documented layouts pass bit-unchanged."""

    def test_mmap_emission_layout_accepted_bit_identical(self):
        """(B, 1, 1, N, N) — the PtychoDataset emission — must produce output
        bit-identical to the raw broadcast product (no behavior change for
        contract-conforming inputs)."""
        layer = _layer()
        x = _x(batch=4)
        probe = _probe(4, 1, 1, N, N)
        illuminated, masked_probe = layer(x, probe)
        assert illuminated.shape == (4, 1, 1, N, N)
        expected = x.unsqueeze(2) * probe  # probe_mask disabled -> all-ones
        assert torch.equal(illuminated, expected)
        assert torch.equal(masked_probe, probe)

    def test_shared_multimode_probe_accepted(self):
        layer = _layer()
        x = _x(batch=4)
        probe = _probe(1, 1, 3, N, N)
        illuminated, _ = layer(x, probe)
        assert illuminated.shape == (4, 1, 3, N, N)

    def test_documented_per_sample_probes_do_not_couple_samples(self):
        """Under the documented layout, perturbing sample 3's probe leaves
        sample 0's output bit-unchanged (the mixing hazard of case 5a is
        structurally unreachable)."""
        layer = _layer()
        x = _x(batch=4)
        probe = _probe(4, 1, 1, N, N)
        baseline, _ = layer(x, probe)
        perturbed_probe = probe.clone()
        perturbed_probe[3] = perturbed_probe[3] * 7.0
        perturbed, _ = layer(x, perturbed_probe)
        assert torch.equal(perturbed[0], baseline[0])
        assert not torch.equal(perturbed[3], baseline[3])


@pytest.mark.torch
class TestForwardModelRectangularProbeBoundary:
    """Every physics branch enforces PROBE-RANK-001 before arithmetic."""

    def test_flat_batch_probe_raises_before_rectangular_physics(self):
        from ptycho_torch.model import ProbeLayoutError

        model = _rectangular_forward_model()
        batch = 4
        scale = torch.ones(batch, 1, 1, 1)
        experiment_ids = torch.zeros(batch, dtype=torch.long)

        with pytest.raises(ProbeLayoutError, match=r"\(4, 8, 8\)"):
            model.forward(
                _x(batch=batch),
                None,
                None,
                _probe(batch, N, N),
                scale,
                experiment_ids,
            )

    def test_conforming_probe_preserves_rectangular_output_bit_exactly(self):
        model = _rectangular_forward_model()
        batch = 4
        x = _x(batch=batch)
        probe = _probe(batch, 1, 1, N, N)
        scale = torch.ones(batch, 1, 1, 1)
        experiment_ids = torch.zeros(batch, dtype=torch.long)

        with torch.no_grad():
            expected = model.rect_scaler(
                x=x,
                I_raw=None,
                probe=probe,
                scale=scale,
                experiment_ids=experiment_ids,
                autograd=True,
            )
            actual = model.forward(
                x, None, None, probe, scale, experiment_ids
            )

        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)

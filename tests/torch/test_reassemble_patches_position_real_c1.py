import torch

from ptycho_torch import helper as hh
from ptycho_torch.config_params import DataConfig, ModelConfig


def test_reassemble_patches_position_real_c1_uses_centermask_norm():
    """The central_mask merge co-masks numerator and denominator by the same central
    prototype window (periphery zeroed), so a uniform patch maps to ~1 inside the
    central N/2 window and EXACTLY 0 outside -- never the old full-numerator /
    central-count normalization that divided the periphery by 0.001 (~1000x).
    See docs/findings.md TORCH-GS2-CENTRAL-MASK-MERGE-001.
    """
    N = 8
    inputs = torch.ones((1, 1, N, N), dtype=torch.complex64)
    offsets = torch.zeros((1, 1, 1, 2), dtype=torch.float32)

    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(C_forward=1, max_position_jitter=0)

    merged, mask_out, _ = hh.reassemble_patches_position_real(
        inputs, offsets, data_cfg, model_cfg, padded_size=N
    )

    mask = torch.zeros((N, N), dtype=torch.float32)
    center = slice(N // 4, N // 4 + N // 2)
    mask[center, center] = 1.0
    boolean_mask = mask > 1e-6
    # Numerator masked by the same central window and the boolean mask applied:
    # central window -> ~1/(1.001), periphery -> exactly 0.
    expected = (inputs[0, 0] * mask * boolean_mask) / (mask + 0.001)

    assert torch.allclose(merged[0], expected, atol=1e-6)
    assert torch.equal(mask_out[0], boolean_mask)
    assert float(merged.abs().max()) < 2.0

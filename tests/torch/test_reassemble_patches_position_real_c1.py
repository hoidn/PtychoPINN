import torch

from ptycho_torch import helper as hh
from ptycho_torch.config_params import DataConfig, ModelConfig


def test_reassemble_patches_position_real_c1_uses_centermask_norm():
    N = 8
    inputs = torch.ones((1, 1, N, N), dtype=torch.complex64)
    offsets = torch.zeros((1, 1, 1, 2), dtype=torch.float32)

    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(C_forward=1, max_position_jitter=0)

    merged, _, _ = hh.reassemble_patches_position_real(
        inputs, offsets, data_cfg, model_cfg, padded_size=N
    )

    mask = torch.zeros((N, N), dtype=torch.float32)
    center = slice(N // 4, N // 4 + N // 2)
    mask[center, center] = 1.0
    expected = inputs[0, 0] / (mask + 0.001)

    assert torch.allclose(merged[0], expected, atol=1e-6)

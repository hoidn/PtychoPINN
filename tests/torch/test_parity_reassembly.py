import numpy as np
import pytest


def test_reassembly_parity_simple_case():
    tf = pytest.importorskip("tensorflow")
    import torch

    from ptycho import params
    from ptycho.tf_helper import reassemble_position
    from ptycho_torch.config_params import DataConfig, ModelConfig
    from ptycho_torch.helper import reassemble_patches_position_real

    N = 4
    patches = np.zeros((2, N, N, 1), dtype=np.complex64)
    patches[0, 1, 1, 0] = 1.0 + 0.0j
    patches[1, 1, 1, 0] = 1.0 + 0.0j

    offsets = np.array([[[[0.0, 0.0]]], [[[1.0, 0.0]]]], dtype=np.float64)

    old_n = params.get("N")
    try:
        params.set("N", N)
        tf_out = reassemble_position(patches, offsets, M=4).numpy()
    finally:
        params.set("N", old_n)

    torch_patches = torch.from_numpy(patches[..., 0]).unsqueeze(0)  # (1, 2, N, N)
    torch_offsets = torch.from_numpy(offsets.astype(np.float32)).permute(1, 0, 2, 3)  # (1, 2, 1, 2)

    data_cfg = DataConfig(N=N, grid_size=(1, 1))
    model_cfg = ModelConfig()
    model_cfg.C_forward = 2

    torch_out, _, _ = reassemble_patches_position_real(
        torch_patches,
        torch_offsets,
        data_cfg,
        model_cfg,
        crop_size=4,
    )

    tf_amp = np.abs(tf_out[..., 0])
    torch_amp = torch.abs(torch_out[0]).cpu().numpy()

    assert np.allclose(tf_amp, torch_amp, atol=1e-5)

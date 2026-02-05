import numpy as np
import torch
import tensorflow as tf

from ptycho.projective_warp_xla import translate_xla
from ptycho_torch.helper import reassemble_patches_position_real
from ptycho_torch.config_params import DataConfig, ModelConfig


def test_reassembly_offset_sign_matches_tf():
    # Single patch with a single nonzero pixel
    N = 8
    patch_tf = np.zeros((1, N, N, 1), dtype=np.complex64)
    patch_tf[0, N // 2, N // 2, 0] = 1 + 0j
    patch_torch = np.zeros((1, 1, N, N), dtype=np.complex64)
    patch_torch[0, 0, N // 2, N // 2] = 1 + 0j

    # Offset: shift right by +2, down by +1 in TF coords (x, y)
    offsets = np.zeros((1, 1, 1, 2), dtype=np.float64)
    offsets[0, 0, 0, 0] = 2.0  # x
    offsets[0, 0, 0, 1] = 1.0  # y

    # TF reference: direct translation with TF's XLA warp (dx, dy order)
    tf_out = translate_xla(
        tf.convert_to_tensor(patch_tf),
        tf.convert_to_tensor(offsets.reshape(1, 2)),
        interpolation="bilinear",
        use_jit=False,
    ).numpy()

    # Torch output
    data_cfg = DataConfig(N=N, grid_size=(1, 1), C=1)
    model_cfg = ModelConfig(C_forward=1, C_model=1, object_big=True)
    torch_out = reassemble_patches_position_real(
        torch.from_numpy(patch_torch),
        torch.from_numpy(offsets.astype(np.float32)),
        data_cfg,
        model_cfg,
        agg=False,
        padded_size=N,
    )
    torch_out = torch_out.detach().cpu().numpy()

    assert np.allclose(tf_out[..., 0].real, torch_out[:, 0].real, atol=1e-6)
    assert np.allclose(tf_out[..., 0].imag, torch_out[:, 0].imag, atol=1e-6)

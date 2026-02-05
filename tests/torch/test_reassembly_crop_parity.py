import numpy as np
import tensorflow as tf
import torch

from ptycho import params as tf_params
from ptycho.tf_helper import reassemble_position
from ptycho_torch.helper import reassemble_patches_position_real
from ptycho_torch.config_params import DataConfig, ModelConfig


def test_reassembly_crop_matches_tf():
    N = 64
    crop_size = 20
    patch = np.zeros((1, N, N, 1), dtype=np.complex64)
    patch[0, 10:20, 10:20, 0] = 1 + 0j
    offsets = np.zeros((1, 1, 1, 2), dtype=np.float64)

    old_n = tf_params.get('N')
    try:
        tf_params.set('N', N)
        tf_out = reassemble_position(
            tf.convert_to_tensor(patch),
            tf.convert_to_tensor(offsets),
            M=crop_size,
        ).numpy()
    finally:
        tf_params.set('N', old_n)
    if tf_out.ndim == 3 and tf_out.shape[-1] == 1:
        tf_out = np.squeeze(tf_out, axis=-1)

    data_cfg = DataConfig(N=N, grid_size=(1, 1), C=1)
    model_cfg = ModelConfig(C_forward=1, C_model=1, object_big=True)
    torch_out, _, _ = reassemble_patches_position_real(
        torch.from_numpy(patch).permute(0, 3, 1, 2),
        torch.from_numpy(offsets.astype(np.float32)),
        data_cfg,
        model_cfg,
        crop_size=crop_size,
    )
    torch_out = torch_out.detach().cpu().numpy()

    assert np.allclose(tf_out.real, torch_out.real, atol=1e-6)
    assert np.allclose(tf_out.imag, torch_out.imag, atol=1e-6)

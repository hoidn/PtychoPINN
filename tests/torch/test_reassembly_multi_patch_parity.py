import numpy as np
import tensorflow as tf
import torch

from ptycho.tf_helper import reassemble_position
from ptycho_torch.helper import reassemble_patches_position_real
from ptycho_torch.config_params import DataConfig, ModelConfig


def test_reassembly_applies_distinct_offsets():
    N = 64
    patches = np.zeros((2, N, N, 1), dtype=np.complex64)
    patches[0, 8:12, 8:12, 0] = 1 + 0j
    patches[1, 20:24, 20:24, 0] = 1 + 0j

    offsets_tf = np.array([
        [[[0.0], [0.0]]],
        [[[12.0], [10.0]]],
    ], dtype=np.float32)  # (B, 1, 2, 1)

    from ptycho import params
    params.cfg['N'] = N

    tf_out = reassemble_position(
        tf.convert_to_tensor(patches),
        tf.convert_to_tensor(offsets_tf.astype(np.float64)),
        M=20,
    ).numpy()

    data_cfg = DataConfig(N=N, grid_size=(1, 1), C=1)
    model_cfg = ModelConfig(C_forward=1, C_model=1, object_big=True)
    offsets_torch = torch.from_numpy(offsets_tf).permute(0, 3, 1, 2)  # (B, 1, 1, 2)
    torch_out, _, _ = reassemble_patches_position_real(
        torch.from_numpy(patches).permute(0, 3, 1, 2),
        offsets_torch,
        data_cfg,
        model_cfg,
        crop_size=20,
    )
    torch_out = torch_out.detach().cpu().numpy()

    tf_mask = np.abs(tf_out[0]) > 0
    torch_mask = np.abs(torch_out[0]) > 0
    assert torch_mask.sum() == tf_mask.sum()

import numpy as np
import pytest


def test_translation_offsets_parity():
    tf = pytest.importorskip("tensorflow")
    import torch

    from ptycho.tf_helper import Translation as TfTranslation
    from ptycho_torch.helper import Translation as TorchTranslation

    img = np.zeros((1, 5, 5, 1), dtype=np.float32)
    img[0, 2, 2, 0] = 1.0
    offsets = np.array([[1.0, 0.0]], dtype=np.float32)

    tf_img = tf.convert_to_tensor(img)
    tf_offsets = tf.convert_to_tensor(offsets)
    tf_out = TfTranslation(jitter_stddev=0.0, use_xla=False)([tf_img, tf_offsets]).numpy()

    torch_img = torch.tensor(img[..., 0])
    torch_offsets = torch.tensor(offsets).view(1, 1, 2)
    torch_out = TorchTranslation(torch_img, torch_offsets, 0.0).squeeze(1).cpu().numpy()

    assert np.allclose(tf_out[..., 0], torch_out, atol=1e-5)

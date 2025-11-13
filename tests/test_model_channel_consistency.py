"""Regression tests for TensorFlow PINN channel handling."""

import pytest

from ptycho import params, probe as probe_module

# Ensure required legacy params exist before importing ptycho.model
if 'intensity_scale' not in params.cfg:
    params.cfg['intensity_scale'] = 1.0
if 'probe' not in params.cfg:
    probe_module.set_default_probe()

import ptycho.model as MODEL_MODULE  # noqa: E402
import tensorflow as tf  # noqa: E402


@pytest.fixture
def object_big_model_module():
    """Configure params.cfg for gridsize-2 object.big inference and access model module."""
    original_cfg = params.cfg.copy()
    try:
        params.cfg.update({
            'N': 64,
            'gridsize': 2,
            'offset': 4,
            'object.big': True,
            'probe.big': False,
            'pad_object': True,
            'intensity_scale': 1.0,
        })
        yield MODEL_MODULE
    finally:
        params.cfg.clear()
        params.cfg.update(original_cfg)


def test_amp_head_matches_patch_channels(object_big_model_module):
    """Amp decoder must emit C=gridsize**2 channels for object.big workflows."""
    gridsize = params.get('gridsize')
    N = params.get('N')
    autoencoder, _ = MODEL_MODULE.create_model_with_gridsize(gridsize, N)

    amp_shape = autoencoder.get_layer('amp').output.shape
    assert amp_shape[-1] == gridsize ** 2, (
        "Amplitude head must output one channel per patch when object.big=True "
        "(spec-ptycho-core §Coordinate Semantics)."
    )


def test_diffraction_to_obj_accepts_grouped_inputs(object_big_model_module):
    """Inference graph should handle grouped diffraction + coords without shape errors."""
    gridsize = params.get('gridsize')
    N = params.get('N')
    _, diffraction_to_obj = MODEL_MODULE.create_model_with_gridsize(gridsize, N)

    batch = 2
    channels = gridsize ** 2
    diffraction = tf.random.uniform((batch, N, N, channels), dtype=tf.float32)
    coords = tf.zeros((batch, 1, 2, channels), dtype=tf.float32)

    reconstructed_obj = diffraction_to_obj([diffraction, coords])
    assert reconstructed_obj.shape == (batch, N, N, 1)

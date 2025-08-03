"""Global parameter management and configuration state.

Central parameter registry for the PtychoPINN system providing singleton-like 
global state container for all configuration parameters.

Architecture Role:
    Configuration Loading → params.py (global state) → All consumers
    Root of dependency tree, consumed by 23+ modules.

Public Interface:
    `get(key)` - Retrieve parameters; auto-computes 'bigN' from N/gridsize/offset
    `set(key, value)` - Update parameters with validation; affects global state
    `params()` - Complete parameter snapshot including derived values

Usage Example:
    ```python
    import ptycho.params as params
    params.set('N', 64)
    patch_size = params.get('N')       # 64
    grid_coverage = params.get('bigN') # Auto-computed from N/gridsize/offset
    ```

Warnings:
- Mutable global state; values may change during execution
- Legacy system; prefer explicit config objects for new code
"""
import numpy as np
import tensorflow as tf
# TODO naming convention for different types of parameters
# TODO what default value and initialization for the probe scale?
cfg = {
    'N': 128, 'offset': 4, 'gridsize': 2,
    'outer_offset_train': None, 'outer_offset_test': None, 'batch_size': 16,
    'nepochs': 60, 'n_filters_scale': 2, 'output_prefix': 'outputs',
    'big_gridsize': 10, 'max_position_jitter': 10, 'sim_jitter_scale': 0.,
    'default_probe_scale': 0.7, 'mae_weight': 0., 'nll_weight': 1., 'tv_weight': 0.,
    'realspace_mae_weight': 0., 'realspace_weight': 0., 'nphotons': 1e9,
    'nimgs_train': 9, 'nimgs_test': 3,
    'data_source': 'generic', 'probe.trainable': False,
    'intensity_scale.trainable': False, 'positions.provided': False,
    'object.big': True, 'probe.big': False, 'probe_scale': 10., 'set_phi': False,
    'probe.mask': True, 'pad_object': True, 'model_type': 'pinn', 'label': '', 'size': 392,
    'amp_activation': 'sigmoid', 'h5_path': 'wts.h5', 'npseed': 42,
    'debug': True,
    'gaussian_smoothing_sigma': 0.0,  # New parameter for Gaussian smoothing sigma
    'use_xla_translate': True  # Enable XLA-compatible translation by default for better performance
    }

# TODO parameter description
# probe.big: if True, increase the real space solution from 32x32 to 64x64

# TODO bigoffset should be a derived quantity, at least for simulation
def get_bigN():
    N = cfg['N']
    gridsize = cfg['gridsize']
    offset = cfg['offset']
    return N + (gridsize - 1) * offset

def get_padding_size():
    buffer = cfg['max_position_jitter']
    gridsize = cfg['gridsize']
    offset = cfg['offset']
    return (gridsize - 1) * offset + buffer

def get_padded_size():
    bigN = get_bigN()
    buffer = cfg['max_position_jitter']
    return bigN + buffer

def params():
    d = {k:v for k, v in cfg.items()}
    d['bigN'] = get_bigN()
    return d

# TODO refactor
def validate():
    valid_data_sources = ['lines', 'grf', 'experimental', 'points',
        'testimg', 'diagonals', 'xpp', 'V', 'generic']
    assert cfg['data_source'] in valid_data_sources, \
        f"Invalid data source: {cfg['data_source']}. Must be one of {valid_data_sources}."
    if cfg['realspace_mae_weight'] > 0.:
        assert cfg['realspace_weight'] > 0
    return True

def set(key, value):
    print("DEBUG: Setting", key, "to", value, "in params")
    cfg[key] = value
    assert validate()

def get(key):
    if key == 'bigN':
        cfg['bigN'] = get_bigN()
        return cfg['bigN']
    return cfg[key]

def print_params():
    """Print all parameters with special handling for arrays/tensors"""
    all_params = params()
    print("Current Parameters:")
    print("-" * 20)
    for key, value in sorted(all_params.items()):
        if isinstance(value, (np.ndarray, tf.Tensor)):
            print(f"{key}:")
            print(f"  shape: {value.shape}")
            print(f"  mean: {np.mean(value):.3f}")
            print(f"  std: {np.std(value):.3f}")
            print(f"  min: {np.min(value):.3f}")
            print(f"  max: {np.max(value):.3f}")
        else:
            print(f"{key}: {value}")

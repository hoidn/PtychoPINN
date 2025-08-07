"""Global parameter management and configuration state.

Central parameter registry for the PtychoPINN system providing singleton-like 
global state container for all configuration parameters. This module sits at
the root of the dependency tree and is consumed by virtually every other module.

Architecture Role:
    Configuration Loading → params.py (global state) → All consumers (23+ modules)
    
    Critical: This is a legacy system maintained for backward compatibility.
    Modern code should use ptycho.config dataclasses with update_legacy_dict().

Public Interface:
    Core Functions:
        `get(key)` - Retrieve parameter; auto-computes 'bigN' if requested
        `set(key, value)` - Update parameter with validation; prints debug info
        `params()` - Complete parameter snapshot including derived values
    
    Validation & Debug:
        `validate()` - Validate current configuration; raises AssertionError
        `print_params()` - Debug utility displaying all params with array stats
    
    Derived Parameter Functions:
        `get_bigN()` - Compute object coverage: N + (gridsize - 1) * offset
        `get_padding_size()` - Compute padding for position jitter
        `get_padded_size()` - Total padded size including buffer

Migration Guide:
    Modern code should use dataclass configuration:
    ```python
    from ptycho.config import TrainingConfig, update_legacy_dict
    config = TrainingConfig(...)
    update_legacy_dict(params.cfg, config)  # One-way sync
    ```

Usage Example:
    ```python
    import ptycho.params as params
    
    # Basic parameter access
    params.set('N', 64)                    # Prints: DEBUG: Setting N to 64
    params.set('gridsize', 2)
    params.set('offset', 4)
    
    # Retrieve parameters
    patch_size = params.get('N')           # 64
    grid_coverage = params.get('bigN')     # Auto-computed: 64 + (2-1)*4 = 68
    
    # Validation example (will raise AssertionError on invalid data)
    try:
        params.set('data_source', 'invalid')  # Not in allowed list
    except AssertionError:
        print("Invalid data source")
    
    # Debug utilities
    params.print_params()                   # Display all params with array stats
    params.validate()                       # Explicit validation check
    ```

Warnings:
- Mutable global state that can change during execution
- set() operations print debug messages to stdout
- set() includes validation that raises AssertionError on failure
- Initialization order matters - modules depending on params.cfg must import
  after proper initialization (see DEVELOPER_GUIDE.md)
- Legacy system - new code should use explicit dataclass configurations
- Auto-computed parameters (bigN) recalculated on each get() call
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

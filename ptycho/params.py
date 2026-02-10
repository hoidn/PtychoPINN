"""Legacy global configuration system for PtychoPINN.

⚠️  DEPRECATED: This module implements a global dictionary-based parameter management
system that is being phased out in favor of the modern dataclass configuration system
in `ptycho.config.config`. Despite its deprecated status, this module remains CRITICAL
for backward compatibility as it is the most imported module in the codebase (23+ consumers).

Architecture Role:
    This module represents the old architecture pattern where configuration was managed
    through a global mutable dictionary (`cfg`). While this pattern has known limitations
    (global state, lack of type safety, difficult testing), it is still essential for
    maintaining compatibility with existing code during the modernization transition.

Global Configuration Dictionary:
    The `cfg` dictionary contains all system parameters including:
    - Model architecture: N (patch size), gridsize, n_filters_scale
    - Training: batch_size, nepochs, various loss weights
    - Physics simulation: nphotons, probe parameters, positioning
    - Data processing: offsets, padding, object/probe sizing

Core Functions:
    - get(key): Retrieve parameter value with special handling for derived values
    - set(key, value): Update parameter with automatic validation
    - params(): Get complete parameter dictionary with derived values
    - validate(): Ensure parameter consistency and valid combinations

Migration Strategy:
    Modern workflows should use the dataclass system from `ptycho.config.config`.
    The modern system updates this legacy `cfg` dictionary at initialization to
    maintain compatibility with existing modules that still use `params.get()`.

Usage Examples:
    Legacy pattern (deprecated):
        import ptycho.params as p
        p.set('N', 128)
        batch_size = p.get('batch_size')
    
    Modern pattern (preferred):
        from ptycho.config.config import TrainingConfig
        config = TrainingConfig(N=128, batch_size=32)
        # Modern config automatically updates legacy params.cfg

Deprecation Timeline:
    This module will remain until all 23+ consumer modules are migrated to accept
    configuration dataclasses directly. Current high-priority consumers requiring
    migration include: baselines, diffsim, evaluation, loader, model, train_pinn,
    and workflows.components.

Warnings:
    - Avoid using this module in new code
    - Global state makes testing and concurrency difficult  
    - Parameter changes affect all code using this module
    - Type safety is not enforced on parameter values
"""
import numpy as np
import tensorflow as tf
# TODO naming convention for different types of parameters
# TODO what default value and initialization for the probe scale?
DEFAULT_CFG = {
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
cfg = DEFAULT_CFG.copy()

def ensure_defaults():
    """Ensure legacy params.cfg contains baseline defaults."""
    for key, value in DEFAULT_CFG.items():
        if key not in cfg:
            cfg[key] = value

# TODO parameter description
# probe.big: if True, increase the real space solution from 32x32 to 64x64

# TODO bigoffset should be a derived quantity, at least for simulation
def get_bigN():
    ensure_defaults()
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
    ensure_defaults()
    d = {k:v for k, v in cfg.items()}
    d['bigN'] = get_bigN()
    return d

# TODO refactor
def validate():
    ensure_defaults()
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

# Sentinel value to distinguish between no default and None as default
_NO_DEFAULT = object()

def get(key, default=_NO_DEFAULT):
    ensure_defaults()
    if key == 'bigN':
        cfg['bigN'] = get_bigN()
        return cfg['bigN']
    # If no default provided, raise KeyError for backward compatibility
    if default is _NO_DEFAULT:
        return cfg[key]  # Will raise KeyError if key doesn't exist
    return cfg.get(key, default)

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

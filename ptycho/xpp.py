"""
X-ray Pulse Probe (XPP) experimental data interface for PtychoPINN.

This module provides a specialized data loading interface for ptychographic data
collected from X-ray Free Electron Laser (XFEL) experiments, specifically from
the XPP beamline. It serves as a compatibility layer between raw experimental
datasets and the PtychoPINN processing pipeline.

## Core Functionality

The module loads and pre-processes experimental ptychographic data from the
standardized Run1084 dataset, providing:

- Pre-configured grid sampling parameters (32x32 scan positions)
- Direct access to loaded experimental probe and object estimates
- Integration with the PtychoPINN data preprocessing pipeline

## Data Pipeline Integration

```
XPP Beamline Data → NPZ Format → xpp.py Interface → RawData Objects → PtychoDataContainer
```

This module acts as the first stage in processing real experimental data, feeding
into the standard PtychoPINN workflow through the `data_preprocessing.py` module.

## Legacy System Context

Part of the original PtychoPINN experimental data handling system. Uses the legacy
`ptycho.params` configuration for scan parameters. The module demonstrates the
transition from beamline-specific data formats to the standardized data contracts
required by the reconstruction algorithms.

## Typical Usage

```python
import ptycho.xpp as xpp
from ptycho.data_preprocessing import process_xpp_data

# Access experimental data (loaded on-demand)
diffraction_patterns = xpp.obj['diffraction']  # Shape: (n_images, N, N)
probe_estimate = xpp.obj['probeGuess']         # Shape: (N, N) complex
object_estimate = xpp.obj['objectGuess']       # Shape: (M, M) complex

# Alternative: direct import of the loader function
from ptycho.xpp import load_ptycho_data
ptycho_data, ptycho_data_train, obj = load_ptycho_data(data_file_path)

# Integration with preprocessing pipeline
processed_data = process_xpp_data(xpp.ptycho_data_train)
```

## Configuration Dependencies

- **Grid Parameters**: Fixed 32x32 scan grid from experimental setup
- **Image Size**: N=64 pixel diffraction patterns
- **Train Fraction**: 50% train/test split for experimental validation
- **Seed**: Fixed random seed (7) for reproducible dataset splits

## Data Contract Compliance

The loaded data follows the standard PtychoPINN data contracts:
- `diffraction`: Amplitude data (not intensity) as (n_images, N, N) real array
- `probeGuess`: Complex probe estimate as (N, N) array
- `objectGuess`: Complex object estimate as (M, M) array where M >> N
- `xcoords`, `ycoords`: Scan position arrays as (n_images,) arrays

## Implementation Notes

- Uses `pkg_resources` for dataset packaging and distribution
- Hardcoded reference to Run1084 experimental dataset
- **Lazy loading**: Data loading deferred until first attribute access (import-safe)
- Module-level attribute access via `__getattr__` for backward compatibility
- Global variables cache loaded data for downstream processing

This module represents the experimental data entry point into the PtychoPINN
system and demonstrates the integration patterns required for processing
real XFEL ptychographic datasets.
"""
import numpy as np
import pkg_resources

from .loader import load_xpp_npz as load_ptycho_data

train_frac = .5
N = 64
gridh, gridw = 32, 32

np.random.seed(7)

def get_data(**kwargs):
    return dset, train_frac


# Global variables to cache loaded data
_ptycho_data = None
_ptycho_data_train = None
_obj = None
_probeGuess = None
_objectGuess = None

def _ensure_data_loaded():
    """Lazy loading function to load data only when accessed."""
    global _ptycho_data, _ptycho_data_train, _obj, _probeGuess, _objectGuess
    
    if _ptycho_data is None:
        data_file_path = pkg_resources.resource_filename(__name__, 'datasets/Run1084_recon3_postPC_shrunk_3.npz')
        _ptycho_data, _ptycho_data_train, _obj = load_ptycho_data(data_file_path)
        print('raw diffraction shape', _obj['diffraction'].shape)
        # TODO cast to complex64?
        _probeGuess = _obj['probeGuess']
        _objectGuess = _obj['objectGuess']

def get_ptycho_data():
    """Access to the full ptycho dataset."""
    _ensure_data_loaded()
    return _ptycho_data

def get_ptycho_data_train():
    """Access to the training subset of ptycho dataset."""
    _ensure_data_loaded()
    return _ptycho_data_train

def get_obj():
    """Access to the loaded data dictionary."""
    _ensure_data_loaded()
    return _obj

def get_probeGuess():
    """Access to the probe guess."""
    _ensure_data_loaded() 
    return _probeGuess

def get_objectGuess():
    """Access to the object guess."""
    _ensure_data_loaded()
    return _objectGuess

# For backward compatibility, provide module-level access
def __getattr__(name):
    """Module-level attribute access for backward compatibility."""
    if name == 'ptycho_data':
        return get_ptycho_data()
    elif name == 'ptycho_data_train':
        return get_ptycho_data_train()
    elif name == 'obj':
        return get_obj()
    elif name == 'probeGuess':
        return get_probeGuess()
    elif name == 'objectGuess':
        return get_objectGuess()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

## TODO refactor actual / nominal positions

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
from ptycho.xpp import ptycho_data, ptycho_data_train, obj
from ptycho.data_preprocessing import process_xpp_data

# Access pre-loaded experimental data
diffraction_patterns = obj['diffraction']  # Shape: (n_images, N, N)
probe_estimate = obj['probeGuess']         # Shape: (N, N) complex
object_estimate = obj['objectGuess']       # Shape: (M, M) complex

# Integration with preprocessing pipeline
processed_data = process_xpp_data(ptycho_data_train)
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
- Direct execution of data loading on module import (legacy pattern)
- Global variables expose loaded data for downstream processing

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


data_file_path = pkg_resources.resource_filename(__name__, 'datasets/Run1084_recon3_postPC_shrunk_3.npz')
ptycho_data, ptycho_data_train, obj = load_ptycho_data(data_file_path)
print('raw diffraction shape', obj['diffraction'].shape)
# TODO cast to complex64?
probeGuess = obj['probeGuess']
objectGuess = obj['objectGuess']

## TODO refactor actual / nominal positions

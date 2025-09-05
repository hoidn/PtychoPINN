"""Data orchestration module coordinating raw data sources with model-ready containers.

This module serves as the high-level data preparation orchestrator for PtychoPINN workflows,
bridging between raw data sources (raw_data.py), data containers (loader.py), physics
simulation (diffsim), and probe generation. It provides a unified entry point for creating
complete training/test datasets from various data sources while handling format consistency
and preprocessing requirements.

Orchestration Flow:
    1. Data Source Selection: Routes to appropriate loader based on params.cfg['data_source']
    2. Format Coordination: Ensures consistent array shapes between raw_data, diffsim, and loader
    3. Container Packaging: Creates standardized PtychoDataset objects for model consumption
    4. State Management: Updates global params.cfg with derived values (intensity_scale, etc.)

Public Interface:
    generate_data(probeGuess=None) -> (X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, 
                                       Y_phi_test, ground_truth, dataset, full_test, norm)
        Main orchestration entry point returning 10-element tuple:
        - X_train, X_test: Diffraction amplitude arrays (n_images, N, N)
        - Y_I_train, Y_phi_train, Y_I_test, Y_phi_test: Object amplitude/phase arrays
        - ground_truth: Clipped ground truth object for evaluation (if available)
        - dataset: Complete PtychoDataset container with train/test splits
        - full_test: Full-size test object for stitching operations
        - norm: Test data normalization factor

Data Source Support:
    - Simulated ('lines', 'grf', 'points', 'testimg', 'diagonals', 'V'): Uses diffsim
    - Experimental ('experimental'): Real measurement data with single image constraint  
    - XPP ('xpp'): LCLS beamline-specific format
    - Generic ('generic'): Arbitrary .npz files via raw_data.py

State Dependencies:
    Heavily relies on params.cfg for configuration including:
    - data_source: Determines loading pathway
    - N, gridsize, offset: Array dimension parameters  
    - outer_offset_train/test: Clipping boundaries for grid-based stitching
    - nimgs_train/test: Dataset size constraints
    - intensity_scale: Derived normalization factor (updated by this module)

Usage:
    # Standard workflow - all parameters from params.cfg
    data_tuple = generate_data()
    X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, \
        ground_truth, dataset, full_test, norm = data_tuple
    
    # Access structured containers
    _, _, _, _, _, _, _, dataset, _, _ = generate_data()
    train_container = dataset.train_data  # PtychoDataContainer
    test_container = dataset.test_data    # PtychoDataContainer

Note:
    Grid-based stitching operations (stitch_data) only work with gridsize > 1.
    Non-grid mode (gridsize=1) datasets cannot be reassembled into regular grids.
"""

from sklearn.utils import shuffle
import numpy as np

from ptycho import params
from ptycho import diffsim as datasets
import tensorflow as tf

from .loader import PtychoDataset, PtychoDataContainer
from ptycho import loader
from ptycho import probe

if params.get('outer_offset_train') is None or params.get('outer_offset_test') is None:
    assert params.get('data_source') == 'generic'

def load_simulated_data(size, probe, outer_offset_train, outer_offset_test, jitter_scale, intensity_scale=None):
    np.random.seed(1)
    X_train, Y_I_train, Y_phi_train, intensity_scale, YY_train_full, _, (coords_train_nominal, coords_train_true) = \
        datasets.mk_simdata(params.get('nimgs_train'), size, probe, outer_offset_train, jitter_scale=jitter_scale, which = 'train')
    params.cfg['intensity_scale'] = intensity_scale

    np.random.seed(2)
    X_test, Y_I_test, Y_phi_test, _, YY_test_full, norm_Y_I_test, (coords_test_nominal, coords_test_true) = \
        datasets.mk_simdata(params.get('nimgs_test'), size, probe, outer_offset_test, intensity_scale, jitter_scale=jitter_scale, which = 'test')

    return X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, intensity_scale, YY_train_full, YY_test_full, norm_Y_I_test, coords_train_nominal, coords_train_true, coords_test_nominal, coords_test_true

def load_experimental_data(probe, outer_offset_train, outer_offset_test, jitter_scale):
    from ptycho import experimental
    YY_I, YY_phi = experimental.get_full_experimental('train')
    X_train, Y_I_train, Y_phi_train, intensity_scale, YY_train_full, _, (coords_train_nominal, coords_train_true) = \
        datasets.mk_simdata(params.get('nimgs_train'), experimental.train_size, probe, outer_offset_train, jitter_scale=jitter_scale, YY_I=YY_I, YY_phi=YY_phi)

    YY_I, YY_phi = experimental.get_full_experimental('test')
    X_test, Y_I_test, Y_phi_test, _, YY_test_full, norm_Y_I_test, (coords_test_nominal, coords_test_true) = \
        datasets.mk_simdata(params.get('nimgs_test'), experimental.test_size, probe, outer_offset_test, intensity_scale, jitter_scale=jitter_scale, YY_I=YY_I, YY_phi=YY_phi)

    return X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, intensity_scale, YY_train_full, YY_test_full, norm_Y_I_test, coords_train_nominal, coords_train_true, coords_test_nominal, coords_test_true

def load_xpp_data(probeGuess):
    from ptycho import xpp
    train_data_container = loader.load(xpp.get_data, probeGuess, which='train')
    test_data_container = loader.load(xpp.get_data, probeGuess, which='test')
    return train_data_container, test_data_container

def load_generic_data(probeGuess, N):
    from ptycho.raw_data import RawData
    train_data_file_path = params.get('train_data_file_path')
    test_data_file_path = params.get('test_data_file_path')

    train_raw = RawData.from_file(train_data_file_path)
    test_raw = RawData.from_file(test_data_file_path)

    gridsize = params.get('gridsize', 1)
    dset_train = train_raw.generate_grouped_data(N, K=7, nsamples=1, gridsize=gridsize)
    dset_test = test_raw.generate_grouped_data(N, K=7, nsamples=1, gridsize=gridsize)

    train_data_container = loader.load(lambda: dset_train, probeGuess, which=None, create_split=False)
    test_data_container = loader.load(lambda: dset_test, probeGuess, which=None, create_split=False)
    return train_data_container, test_data_container

def shuffle_data(X, Y_I, Y_phi, random_state=0):
    indices = np.arange(len(Y_I))
    indices_shuffled = shuffle(indices, random_state=random_state)

    X_shuffled = X[indices_shuffled]
    Y_I_shuffled = Y_I[indices_shuffled]
    Y_phi_shuffled = Y_phi[indices_shuffled]

    return X_shuffled, Y_I_shuffled, Y_phi_shuffled, indices_shuffled

def get_clipped_object(YY_full, outer_offset):
    borderleft, borderright, clipleft, clipright = get_clip_sizes(outer_offset)

    extra_size = (YY_full.shape[1] - (params.cfg['N'] + (params.cfg['gridsize'] - 1) * params.cfg['offset'])) % (outer_offset // 2)
    if extra_size > 0:
        YY_ground_truth = YY_full[:, :-extra_size, :-extra_size]
    else:
        print('discarding length {} from test image'.format(extra_size))
        YY_ground_truth = YY_full
    YY_ground_truth = YY_ground_truth[:, clipleft:-clipright, clipleft:-clipright]
    return YY_ground_truth

def get_clip_sizes(outer_offset):
    N = params.cfg['N']
    gridsize = params.cfg['gridsize']
    offset = params.cfg['offset']
    bordersize = (N - outer_offset / 2) / 2
    borderleft = int(np.ceil(bordersize))
    borderright = int(np.floor(bordersize))
    clipsize = (bordersize + ((gridsize - 1) * offset) // 2)
    clipleft = int(np.ceil(clipsize))
    clipright = int(np.floor(clipsize))
    return borderleft, borderright, clipleft, clipright

def stitch_data(b, norm_Y_I_test=1, norm=True, part='amp', outer_offset=None, nimgs=None):
    # Check if we're in non-grid mode (gridsize=1)
    if params.get('gridsize') == 1:
        # For gridsize=1, we can't do grid-based stitching
        # Return None or raise an informative error
        raise ValueError("Grid-based stitching is not supported for gridsize=1 (non-grid mode). "
                        "Individual patches cannot be arranged in a regular grid.")
    
    # channel size must be 1, or not present
    if b.shape[-1] != 1:
        assert b.shape[-1] == params.get(['N'])
    if nimgs is None:
        nimgs = params.get('nimgs_test')
    if outer_offset is None:
        outer_offset = params.get('outer_offset_test')
    nsegments = int(np.sqrt((int(tf.size(b)) / nimgs) / (params.cfg['N']**2)))
    if part == 'amp':
        getpart = np.absolute
    elif part == 'phase':
        getpart = np.angle
    elif part == 'complex':
        getpart = lambda x: x
    else:
        raise ValueError
    if norm:
        img_recon = np.reshape((norm_Y_I_test * getpart(b)), (-1, nsegments, nsegments, params.cfg['N'], params.cfg['N'], 1))
    else:
        img_recon = np.reshape((getpart(b)), (-1, nsegments, nsegments, params.cfg['N'], params.cfg['N'], 1))
    borderleft, borderright, clipleft, clipright = get_clip_sizes(outer_offset)
    img_recon = img_recon[:, :, :, borderleft:-borderright, borderleft:-borderright, :]
    tmp = img_recon.transpose(0, 1, 3, 2, 4, 5)
    stitched = tmp.reshape(-1, np.prod(tmp.shape[1:3]), np.prod(tmp.shape[1:3]), 1)
    return stitched

def reassemble(b, norm_Y_I = 1., part='amp', **kwargs):
    stitched = stitch_data(b, norm_Y_I, norm=False, part=part, **kwargs)
    return stitched

def process_simulated_data(X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, YY_test_full, outer_offset_test):
    X_train, Y_I_train, Y_phi_train, indices_shuffled = shuffle_data(np.array(X_train), np.array(Y_I_train), np.array(Y_phi_train))
    if params.get('outer_offset_train') is not None:
        YY_ground_truth_all = get_clipped_object(YY_test_full, outer_offset_test)
        YY_ground_truth = YY_ground_truth_all[0, ...]
        print('DEBUG: generating grid-mode ground truth image')
    else:
        YY_ground_truth = None
        print('DEBUG: No ground truth image in non-grid mode')
    return X_train, Y_I_train, Y_phi_train, YY_ground_truth

def create_ptycho_dataset(X_train, Y_I_train, Y_phi_train, intensity_scale, YY_train_full, coords_train_nominal, coords_train_true,
                          X_test, Y_I_test, Y_phi_test, YY_test_full, coords_test_nominal, coords_test_true):
    return PtychoDataset(
        PtychoDataContainer(X_train, Y_I_train, Y_phi_train, intensity_scale, YY_train_full, coords_train_nominal, coords_train_true, None, None, None, probe.get_probe(params)),
        PtychoDataContainer(X_test, Y_I_test, Y_phi_test, intensity_scale, YY_test_full, coords_test_nominal, coords_test_true, None, None, None, probe.get_probe(params)),
    )

def generate_data(probeGuess = None):
    # TODO handle probeGuess None case
    data_source = params.params()['data_source']
    probe_np = probe.get_probe(params)
    outer_offset_train = params.cfg['outer_offset_train']
    outer_offset_test = params.cfg['outer_offset_test']
    YY_test_full = None
    norm_Y_I_test = None

    if data_source in ['lines', 'grf', 'points', 'testimg', 'diagonals', 'V']:
        size = params.cfg['size']
        X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, intensity_scale, YY_train_full, YY_test_full, norm_Y_I_test, coords_train_nominal, coords_train_true, coords_test_nominal, coords_test_true = \
            load_simulated_data(size, probe_np, outer_offset_train, outer_offset_test, params.params()['sim_jitter_scale'])
        X_train, Y_I_train, Y_phi_train, YY_ground_truth = process_simulated_data(X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, YY_test_full, outer_offset_test)
        ptycho_dataset = create_ptycho_dataset(X_train, Y_I_train, Y_phi_train, intensity_scale, YY_train_full, coords_train_nominal, coords_train_true,
                                               X_test, Y_I_test, Y_phi_test, YY_test_full, coords_test_nominal, coords_test_true)
    elif data_source == 'experimental':
        # Ensure nimgs parameters are 1 for experimental data
        assert params.get('nimgs_train') == 1, "nimgs_train must be 1 for experimental data"
        assert params.get('nimgs_test') == 1, "nimgs_test must be 1 for experimental data"
        
        X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, intensity_scale, YY_train_full, YY_test_full, norm_Y_I_test, coords_train_nominal, coords_train_true, coords_test_nominal, coords_test_true = \
            load_experimental_data(probe_np, outer_offset_train, outer_offset_test, params.params()['sim_jitter_scale'])
        X_train, Y_I_train, Y_phi_train, YY_ground_truth = process_simulated_data(X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, YY_test_full, outer_offset_test)
        ptycho_dataset = create_ptycho_dataset(X_train, Y_I_train, Y_phi_train, intensity_scale, YY_train_full, coords_train_nominal, coords_train_true,
                                               X_test, Y_I_test, Y_phi_test, YY_test_full, coords_test_nominal, coords_test_true)
    elif data_source == 'xpp':
        test_data_container, train_data_container = load_xpp_data(probeGuess)
        intensity_scale = train_data_container.norm_Y_I
        ptycho_dataset = PtychoDataset(train_data_container, test_data_container)
        YY_ground_truth = None
        YY_test_full = None
    elif data_source == 'generic':
        train_data_container, test_data_container = load_generic_data(probeGuess, params.cfg['N'])
        intensity_scale = train_data_container.norm_Y_I
        ptycho_dataset = PtychoDataset(train_data_container, test_data_container)
        YY_ground_truth = None
        print('INFO: train data:')
        print(train_data_container)
        print('INFO: test data:')
        print(test_data_container)
    else:
        raise ValueError("Invalid data source")

    params.cfg['intensity_scale'] = intensity_scale
    return ptycho_dataset.train_data.X, ptycho_dataset.train_data.Y_I, ptycho_dataset.train_data.Y_phi, ptycho_dataset.test_data.X, ptycho_dataset.test_data.Y_I, ptycho_dataset.test_data.Y_phi, YY_ground_truth, ptycho_dataset, YY_test_full, norm_Y_I_test


from sklearn.utils import shuffle
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt

from ptycho import params
from ptycho import diffsim as datasets
from ptycho import fourier as f
import tensorflow as tf

"""
Initialize probe and other parameters; build (simulated) training / evaluation data
"""

# TODO dataset should go to a PtychoData object

# data parameters
offset = params.cfg['offset']
N = params.cfg['N']
gridsize = params.cfg['gridsize']
jitter_scale = params.params()['sim_jitter_scale']

# training parameters
nepochs = params.cfg['nepochs']
batch_size = params.cfg['batch_size']

# Initialize the probe function outside of the dataset-specific code
# to ensure it is shared between training and testing
from ptycho import probe as probe_module
probe = probe_module.get_probe(fmt='np')

def normed_ff_np(arr):
    return (f.fftshift(np.absolute(f.fft2(np.array(arr)))) / np.sqrt(N))

def shuffle_data(X, Y_I, Y_phi, random_state=0):
    """
    Function to shuffle data.
    X, Y_I, Y_phi are numpy arrays to be shuffled along the first axis.
    """
    indices = np.arange(len(Y_I))
    indices_shuffled = shuffle(indices, random_state=random_state)

    X_shuffled = X[indices_shuffled]
    Y_I_shuffled = Y_I[indices_shuffled]
    Y_phi_shuffled = Y_phi[indices_shuffled]

    return X_shuffled, Y_I_shuffled, Y_phi_shuffled, indices_shuffled

def unshuffle_data(Y_I):
    """
    Function to unshuffle data.
    Y_I is the shuffled numpy array.
    indices_shuffled is the numpy array of shuffled indices.
    """
    unshuffle_indices = np.argsort(indices_shuffled)
    Y_I_unshuffled = Y_I[unshuffle_indices]

    return Y_I_unshuffled

def get_clip_sizes(outer_offset):
    """
    How much to clip so that the ground truth images and reconstructions match.
    """
    # TODO remove the factor of 2 assumption
    # TODO check that the rounding behavior is okay.
    ## It might be safer to assert bigoffset % 4 == 0
    bordersize = (N - outer_offset / 2) / 2

    # Amount to trim from NxN reconstruction patches
    borderleft = int(np.ceil(bordersize))
    borderright = int(np.floor(bordersize))

    # Amount to trim from the ground truth object
    clipsize = (bordersize + ((gridsize - 1) * offset) // 2)
    clipleft = int(np.ceil(clipsize))
    clipright = int(np.floor(clipsize))
    return borderleft, borderright, clipleft, clipright

def get_clipped_object(YY_full, outer_offset):
    # TODO bigN, outer_offset_test factor of 2
    # Ground truth needs to be trimmed to line up with the reconstruction
    # Remove the portion of the test image that wasn't converted into
    # patches of evaluation data
    borderleft, borderright, clipleft, clipright = get_clip_sizes(outer_offset)

    extra_size = (YY_full.shape[1] - (N + (gridsize - 1) * offset)) % (outer_offset // 2)
    if extra_size > 0:
        YY_ground_truth = YY_full[:, :-extra_size, :-extra_size]
    else:
        print('discarding length {} from test image'.format(extra_size))
        YY_ground_truth = YY_full
    YY_ground_truth = YY_ground_truth[:, clipleft: -clipright, clipleft: -clipright]
    return YY_ground_truth

# TODO normalization not needed?
def stitch(b, norm_Y_I_test = 1, norm = True, part = 'amp', outer_offset = None,
        nimgs = None):
    if nimgs is None:
        nimgs = params.get('nimgs_test')
    if outer_offset is None:
        outer_offset = params.get('outer_offset_test')
    nsegments = int(np.sqrt((int(tf.size(b)) / nimgs) / (N**2)))
    if part == 'amp':
        getpart = np.absolute
    elif part == 'phase':
        getpart = np.angle
    elif part == 'complex':
        getpart = lambda x: x
    else:
        raise ValueError
    if norm:
        img_recon = np.reshape((norm_Y_I_test * getpart(b)), (-1, nsegments,
                                                              nsegments, N, N, 1))
    else:
        img_recon = np.reshape((getpart(b)), (-1, nsegments,
                                                              nsegments, N, N, 1))
    borderleft, borderright, clipleft, clipright = get_clip_sizes(outer_offset)
    img_recon = img_recon[:, :, :, borderleft: -borderright, borderleft: -borderright, :]
    tmp = img_recon.transpose(0, 1, 3, 2, 4, 5)
    stitched = tmp.reshape(-1, np.prod(tmp.shape[1:3]), np.prod(tmp.shape[1:3]), 1)
    return stitched

# TODO refactor
def reassemble(b, part = 'amp', **kwargs):
    stitched = stitch(b, norm_Y_I_test, norm = False, part = part, **kwargs)
    return stitched

# TODO refactor
if params.params()['data_source'] in ['lines', 'grf', 'points', 'testimg', 'diagonals', 'V']:
    bigN = params.params()['bigN']

    # Smaller stride so that solution regions overlap enough
    if params.cfg['outer_offset_train'] is None:
        # TODO move to params
        outer_offset_train = (gridsize - 1) * offset + N // 2
        # TODO get rid of this parameter
        outer_offset_train = params.cfg['outer_offset_train'] = outer_offset_train // 2
    else:
        outer_offset_train = params.cfg['outer_offset_train']
    if params.cfg['outer_offset_test'] is None:
        outer_offset_test = params.cfg['outer_offset_test']  = outer_offset_train
    else:
        outer_offset_test = params.cfg['outer_offset_test']

    size = params.cfg['size']

    # simulate data
    np.random.seed(1)
    # Ensure the probe is initialized with the correct format and dimensionality
    probe = probe_module.get_probe(fmt='np')
    assert probe.ndim == 2, "Probe function must be a 2D array"

    # Generate simulated data and enforce dimensionality
    train_data = datasets.mk_simdata(params.get('nimgs_train'), size, probe,
                                     params.get('outer_offset_train'), jitter_scale=jitter_scale)
    X_train, Y_I_train, Y_phi_train, intensity_scale, YY_train_full, _, coords_train = train_data
    assert X_train.ndim == 4, "X_train must be a 4D tensor (batch, height, width, channels)"
    assert Y_I_train.ndim == 4, "Y_I_train must be a 4D tensor (batch, height, width, channels)"
    assert Y_phi_train.ndim == 4, "Y_phi_train must be a 4D tensor (batch, height, width, channels)"
    params.cfg['intensity_scale'] = intensity_scale

    #bigoffset = params.cfg['bigoffset'] = bigoffset * 2
    np.random.seed(2)
    (X_test, Y_I_test, Y_phi_test,
        _, YY_test_full, norm_Y_I_test,
        (coords_test_nominal, coords_test_true)) =\
        datasets.mk_simdata(params.get('nimgs_test'), size, probe.get_probe(fmt = 'np'),
        params.get('outer_offset_test'), intensity_scale,
        jitter_scale = jitter_scale)

# TODO distinguish between bigoffset for train and test. Should have two
# variables instead of one
elif params.params()['data_source'] == 'experimental':
    # TODO refactor
    from ptycho import experimental
    params.set('nimgs_train', 1)
    params.set('nimgs_test', 1)
    if params.cfg['outer_offset_train'] is None:
        params.cfg['outer_offset_train'] = 4

    if params.cfg['outer_offset_test'] is None:
        outer_offset_test = params.cfg['outer_offset_test'] = 20
    else:
        outer_offset_test = params.cfg['outer_offset_test']
    bigN = N + (gridsize - 1) * offset

    YY_I, YY_phi = experimental.get_full_experimental('train')
    (X_train, Y_I_train, Y_phi_train,
        intensity_scale, YY_train_full, _,
        (coords_train_nominal, coords_train_true)) =\
        datasets.mk_simdata(params.get('nimgs_train'), experimental.train_size,
            probe.get_probe(fmt = 'np'), params.get('outer_offset_train'), jitter_scale = jitter_scale,
            YY_I = YY_I, YY_phi = YY_phi)

    params.cfg['intensity_scale'] = intensity_scale

    YY_I, YY_phi = experimental.get_full_experimental('test')
    (X_test, Y_I_test, Y_phi_test,
        _, YY_test_full, norm_Y_I_test,
        (coords_test_nominal, coords_test_true)) =\
        datasets.mk_simdata(params.get('nimgs_test'), experimental.test_size,
        probe.get_probe(fmt = 'np'), params.get('outer_offset_test'), intensity_scale,
        jitter_scale = jitter_scale,
        YY_I = YY_I, YY_phi = YY_phi)
    size = int(YY_test_full.shape[1])

elif params.params()['data_source'] == 'xpp':
    from ptycho import xpp
    params.set('nimgs_train', 1)
    params.set('nimgs_test', 1)
    outer_offset_test = params.cfg['outer_offset_test']

    train_data = xpp.load('train')
    X_train = train_data['X']
    Y_I_train = train_data['Y_I']
    Y_phi_train = train_data['Y_phi']
    intensity_scale = train_data['norm_Y_I']
    YY_train_full = train_data['YY_full']
    coords_train_nominal, coords_train_true = train_data['coords']

    params.cfg['intensity_scale'] = intensity_scale

    # Loading test data
    test_data = xpp.load('test')
    X_test = test_data['X']
    Y_I_test = test_data['Y_I']
    Y_phi_test = test_data['Y_phi']
    YY_test_full = test_data['YY_full']
    norm_Y_I_test = test_data['norm_Y_I']
    coords_test_nominal, coords_test_true = test_data['coords']

else:
    raise ValueError

# TODO shuffle should be after flatten. unecessary copies
#X_train, Y_I_train, Y_phi_train =\
#    shuffle(np.array(X_train), np.array(Y_I_train), np.array(Y_phi_train),
#        random_state=0)
X_train, Y_I_train, Y_phi_train, indices_shuffled =\
    shuffle_data(np.array(X_train), np.array(Y_I_train), np.array(Y_phi_train))

(Y_I_test).shape, Y_I_train.shape

print(np.linalg.norm(ptycho_dataset.train_data.X[0]) /  np.linalg.norm(np.abs(ptycho_dataset.train_data.Y[0])))

# inversion symmetry
assert np.isclose(normed_ff_np(Y_I_train[0, :, :, 0]),
                tf.math.conj(normed_ff_np(Y_I_train[0, ::-1, ::-1, 0])),
                atol = 1e-4).all()

print('nphoton',np.log10(np.sum((X_train[:, :, :] * intensity_scale)**2,
    axis = (1, 2))).mean())

if params.params()['probe.trainable']:
    probe.set_probe_guess(X_train)

# TODO rename / refactor
if params.get('outer_offset_train') is not None:
    YY_ground_truth_all = get_clipped_object(YY_test_full, outer_offset_test)
    YY_ground_truth = YY_ground_truth_all[0, ...]

# TODO refactor
from . import tf_helper as hh
# Create PtychoDataset instance containing both training and test data
ptycho_dataset = PtychoDataset(
    train_data=PtychoData(X_train, Y_I_train, Y_phi_train, YY_train_full, coords_train_nominal, coords_train_true, probe),
    test_data=PtychoData(X_test, Y_I_test, Y_phi_test, YY_test_full, coords_test_nominal, coords_test_true, probe)
)
class PtychoDataset:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

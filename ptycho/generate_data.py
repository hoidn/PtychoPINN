from sklearn.utils import shuffle
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt

from ptycho import params
from ptycho import datasets
from ptycho import fourier as f
import tensorflow as tf

"""
Initialize probe and other parameters; build (simulated) training / evaluation data
"""

# data parameters
offset = params.cfg['offset']
h = w = N = params.cfg['N']
gridsize = params.cfg['gridsize']
jitter_scale = params.params()['sim_jitter_scale']

# training parameters
nepochs = params.cfg['nepochs']
batch_size = params.cfg['batch_size']

# TODO need to enforce that configs are set before initializing the probe
from ptycho import probe

def normed_ff_np(arr):
    return (f.fftshift(np.absolute(f.fft2(np.array(arr)))) / np.sqrt(h * w))

#import pdb
#pdb.set_trace()
if params.params()['data_source'] in ['lines', 'grf', 'points', 'testimg', 'diagonals']:
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

#    big_gridsize = params.params()['big_gridsize'] = int(10 * 18 / outer_offset_train)
#    # TODO
#    size = 2 * outer_offset_train * (big_gridsize - 1) + bigN
    size = 392
    params.cfg['size'] = size

#    # TODO move this to a more sensible place
#    bordersize = (N - bigoffset / 2) / 2

    # simulate data
    np.random.seed(1)
    (X_train, Y_I_train, Y_phi_train,
        intensity_scale, YY_train_full, _,
        (coords_train_nominal, coords_train_true)) =\
        datasets.mk_simdata(params.get('nimgs_train'), size, probe.probe,
            params.get('outer_offset_train'), jitter_scale = jitter_scale)
    params.cfg['intensity_scale'] = intensity_scale

    #bigoffset = params.cfg['bigoffset'] = bigoffset * 2
    np.random.seed(2)
    (X_test, Y_I_test, Y_phi_test,
        _, YY_test_full, norm_Y_I_test,
        (coords_test_nominal, coords_test_true)) =\
        datasets.mk_simdata(params.get('nimgs_test'), size, probe.probe,
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
            probe.probe, params.get('outer_offset_train'), jitter_scale = jitter_scale,
            YY_I = YY_I, YY_phi = YY_phi)

    params.cfg['intensity_scale'] = intensity_scale

    YY_I, YY_phi = experimental.get_full_experimental('test')
    (X_test, Y_I_test, Y_phi_test,
        _, YY_test_full, norm_Y_I_test,
        (coords_test_nominal, coords_test_true)) =\
        datasets.mk_simdata(params.get('nimgs_test'), experimental.test_size,
        probe.probe, params.get('outer_offset_test'), intensity_scale,
        jitter_scale = jitter_scale,
        YY_I = YY_I, YY_phi = YY_phi)
    size = int(YY_test_full.shape[1])

else:
    raise ValueError


# TODO shuffle should be after flatten. unecessary copies
X_train, Y_I_train, Y_phi_train =\
    shuffle(np.array(X_train), np.array(Y_I_train), np.array(Y_phi_train),
        random_state=0)

(Y_I_test).shape, Y_I_train.shape

print(np.linalg.norm(X_train[0]) /  np.linalg.norm(Y_I_train[0]))

# inversion symmetry
assert np.isclose(normed_ff_np(Y_I_train[0, :, :, 0]),
                tf.math.conj(normed_ff_np(Y_I_train[0, ::-1, ::-1, 0])),
                atol = 1e-6).all()

print('nphoton',np.log10(np.sum((X_train[:, :, :] * intensity_scale)**2,
    axis = (1, 2))).mean())

if params.params()['probe.trainable']:
    probe.set_probe_guess(X_train)

def mae(target, pred, normalize = True):
    """
    mae for an entire (stitched-together) reconstruction.
    """
    if normalize:
        scale = np.mean(target) / np.mean(pred)
    else:
        scale = 1
    print('mean scale adjustment:', scale)
    return np.mean(np.absolute(target - scale * pred))
#mae = lambda target, pred: np.mean(np.absolute(target - pred))

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

def get_clip_sizes(outer_offset):
    """
    How much to clip so that the ground truth images and reconstructions match.
    """
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

def get_clipped_object(YY_full, outer_offset, i = 0):
    # TODO bigN, outer_offset_test factor of 2
    # Ground truth needs to be trimmed to line up with the reconstruction
    # Remove the portion of the test image that wasn't converted into
    # patches of evaluation data
    borderleft, borderright, clipleft, clipright = get_clip_sizes(outer_offset)

    extra_size = (YY_full.shape[1] - (N + (gridsize - 1) * offset)) % (outer_offset // 2)
    if extra_size > 0:
        YY_ground_truth = YY_full[i, :-extra_size, :-extra_size]
    else:
        print('discarding length {} from test image'.format(extra_size))
        YY_ground_truth = YY_full[i, ...]
    YY_ground_truth = YY_ground_truth[clipleft: -clipright, clipleft: -clipright]
    return YY_ground_truth


YY_ground_truth = get_clipped_object(YY_test_full, outer_offset_test, 0)

# TODO test this refactored version of the module:
#
#from sklearn.utils import shuffle
#import numpy as np
#import matplotlib.pyplot as plt
#from ptycho import params, datasets, fourier as f, probe, experimental
#import tensorflow as tf
#
#def normed_ff_np(arr, h, w):
#    return (f.fftshift(np.absolute(f.fft2(np.array(arr)))) / np.sqrt(h * w))
#
#def set_params():
#    if params.params()['data_source'] in ['lines', 'grf', 'points', 'testimg', 'diagonals']:
#        big_gridsize = params.params()['big_gridsize'] = 10
#        bigN = params.params()['bigN']
#        return big_gridsize, bigN
#    else:
#        return None, None
#
#def set_outer_offsets_train_test(gridsize, offset, N, outer_offset_train=None, outer_offset_test=None):
#    if outer_offset_train is None:
#        outer_offset_train = (gridsize - 1) * offset + N // 2
#        outer_offset_train = params.cfg['outer_offset_train'] = outer_offset_train // 2
#    if outer_offset_test is None:
#        outer_offset_test = params.cfg['outer_offset_test']  = outer_offset_train
#    return outer_offset_train, outer_offset_test
#
#def simulate_data(size, probe, outer_offset_train, outer_offset_test, jitter_scale, intensity_scale):
#    np.random.seed(1)
#    (X_train, Y_I_train, Y_phi_train, intensity_scale, YY_train_full, _,
#    (coords_train_nominal, coords_train_true)) =\
#    datasets.mk_simdata(params.get('nimgs_train'), size, probe.probe,
#        outer_offset_train, jitter_scale = jitter_scale)
#    params.cfg['intensity_scale'] = intensity_scale
#
#    np.random.seed(2)
#    (X_test, Y_I_test, Y_phi_test, _, YY_test_full, norm_Y_I_test,
#    (coords_test_nominal, coords_test_true)) =\
#    datasets.mk_simdata(params.get('nimgs_test'), size, probe.probe,
#    outer_offset_test, intensity_scale, jitter_scale = jitter_scale)
#
#    return X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, YY_train_full, YY_test_full, norm_Y_I_test, coords_train_nominal, coords_train_true, coords_test_nominal, coords_test_true
#
#def shuffle_data(X_train, Y_I_train, Y_phi_train):
#    X_train, Y_I_train, Y_phi_train =\
#    shuffle(np.array(X_train), np.array(Y_I_train), np.array(Y_phi_train),
#        random_state=0)
#    return X_train, Y_I_train, Y_phi_train
#
## data parameters
#offset = params.cfg['offset']
#h = w = N = params.cfg['N']
#gridsize = params.cfg['gridsize']
#jitter_scale = params.params()['sim_jitter_scale']
#
#big_gridsize, bigN = set_params()
#
#outer_offset_train, outer_offset_test = set_outer_offsets_train_test(gridsize, offset, N, params.cfg['outer_offset_train'], params.cfg['outer_offset_test'])
#
#if big_gridsize is not None and bigN is not None:
#    size = 2 * outer_offset_train * (big_gridsize - 1) + bigN
#    params.cfg['size'] = size
#
#    X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, YY_train_full, YY_test_full, norm_Y_I_test, coords_train_nominal, coords_train_true, coords_test_nominal, coords_test_true = simulate_data(size, probe, outer_offset_train, outer_offset_test, jitter_scale, params.cfg['intensity_scale'])
#
#elif params.params()['data_source'] == 'experimental':
#    # Refactoring needed in experimental
#    outer_offset_train, outer_offset_test = set_outer_offsets_train_test(gridsize, offset, N, params.cfg['outer_offset_train'], params.cfg['outer_offset_test'])
#    bigN = N + (gridsize - 1) * offset
#    YY_I, YY_phi = experimental.get_full_experimental('train')
#    (X_train, Y_I_train, Y_phi_train, intensity_scale, YY_train_full, _,
#    (coords_train_nominal, coords_train_true)) =\
#    datasets.mk_simdata(params.get('nimgs_train'), experimental.train_size,
#        probe.probe, outer_offset_train, jitter_scale = jitter_scale,
#        YY_I = YY_I, YY_phi = YY_phi)
#
#    params.cfg['intensity_scale'] = intensity_scale
#
#    YY_I, YY_phi = experimental.get_full_experimental('test')
#    (X_test, Y_I_test, Y_phi_test, _, YY_test_full, norm_Y_I_test,
#    (coords_test_nominal, coords_test_true)) =\
#    datasets.mk_simdata(params.get('nimgs_test'), experimental.test_size,
#    probe.probe, outer_offset_test, intensity_scale,
#    jitter_scale = jitter_scale,
#    YY_I = YY_I, YY_phi = YY_phi)
#    size = int(YY_test_full.shape[1])
#else:
#    raise ValueError
#
#bordersize = (N - outer_offset_test / 2) / 2
#
#X_train, Y_I_train, Y_phi_train = shuffle_data(X_train, Y_I_train, Y_phi_train)
#
#(Y_I_test).shape, Y_I_train.shape
#
#print(np.linalg.norm(X_train[0]) /  np.linalg.norm(Y_I_train[0]))
#
## inversion symmetry
#assert np.isclose(normed_ff_np(Y_I_train[0, :, :, 0], h, w),
#                tf.math.conj(normed_ff_np(Y_I_train[0, ::-1, ::-1, 0], h, w)),
#                atol = 1e-6).all()
#
#print('nphoton',np.log10(np.sum((X_train[:, :, :] * intensity_scale)**2,
#    axis = (1, 2))).mean())
#
#if params.params()['probe.trainable']:
#    probe.set_probe_guess(X_train)
#
#def mae(target, pred, normalize = True):
#    """
#    mae for an entire (stitched-together) reconstruction.
#    """
#    if normalize:
#        scale = np.mean(target) / np.mean(pred)
#    else:
#        scale = 1
#    print('mean scale adjustment:', scale)
#    return np.mean(np.absolute(target - scale * pred))
##mae = lambda target, pred: np.mean(np.absolute(target - pred))
#
## TODO normalization not needed?
#def stitch(b, norm_Y_I_test = 1,
#           norm = True, part = 'amp'):
#    nimgs = params.get('nimgs_test')
#    nsegments = int(np.sqrt((int(tf.size(b)) / nimgs) / (N**2)))
#    if part == 'amp':
#        getpart = np.absolute
#    elif part == 'phase':
#        getpart = np.angle
#    elif part == 'complex':
#        getpart = lambda x: x
#    else:
#        raise ValueError
#    if norm:
#        img_recon = np.reshape((norm_Y_I_test * getpart(b)), (-1, nsegments,
#                                                              nsegments, N, N, 1))
#    else:
#        img_recon = np.reshape((getpart(b)), (-1, nsegments,
#                                                              nsegments, N, N, 1))
#    img_recon = img_recon[:, :, :, borderleft: -borderright, borderleft: -borderright, :]
#    tmp = img_recon.transpose(0, 1, 3, 2, 4, 5)
#    stitched = tmp.reshape(-1, np.prod(tmp.shape[1:3]), np.prod(tmp.shape[1:3]), 1)
#    return stitched
#
## TODO refactor
#def reassemble(b, part = 'amp'):
#    stitched = stitch(b, norm_Y_I_test, norm = False, part = part)
#    return stitched
#
## Amount to trim from NxN reconstruction patches
#borderleft = int(np.ceil(bordersize))
#borderright = int(np.floor(bordersize))
#
## Amount to trim from the ground truth object
#clipsize = (bordersize + ((gridsize - 1) * offset) // 2)
#clipleft = int(np.ceil(clipsize))
#clipright = int(np.floor(clipsize))
#
## TODO bigN, outer_offset_test factor of 2
## Ground truth needs to be trimmed to line up with the reconstruction
## Remove the portion of the test image that wasn't converted into
## patches of evaluation data
#extra_size = (YY_test_full.shape[1] - (N + (gridsize - 1) * offset)) % (outer_offset_test // 2)
#if extra_size > 0:
#    YY_ground_truth = YY_test_full[0, :-extra_size, :-extra_size]
#else:
#    print('discarding length {} from test image'.format(extra_size))
#    YY_ground_truth = YY_test_full[0, ...]
#YY_ground_truth = YY_ground_truth[clipleft: -clipright, clipleft: -clipright]

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, UpSampling2D
from tensorflow.keras.layers import Lambda
from tensorflow.signal import fft2d, fftshift
import tensorflow_probability as tfp

from .params import params, cfg, get, get_padded_size

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

support_threshold = .0
def get_mask(input, support_threshold):
    mask = tf.where(input > support_threshold, tf.ones_like(input),
                    tf.zeros_like(input))
    return mask

def combine_complex(amp, phi):
    output = tf.cast(amp, tf.complex64) * tf.exp(
        1j * tf.cast(phi, tf.complex64))
    return output

def pad_obj(input, h, w):
    return tfkl.ZeroPadding2D((h // 4, w // 4), name = 'padded_obj')(input)

# TODO nested lambdas?
@tf.function
def pad_and_diffract(input, h, w, pad = True):
    """
    zero-pad the real-space object and then calculate the far field
    diffraction amplitude
    """
    print('input shape', input.shape)
    if pad:
        input = pad_obj(input, h, w)
    padded = input
    assert input.shape[-1] == 1
    input = (((fft2d(
        #tf.squeeze # this destroys shape information so need to use slicing instead
        (tf.cast((input), tf.complex64))[..., 0]
        ))))
    input = (( tf.math.real(tf.math.conj((input)) * input) / (h * w)))
    input = (( tf.expand_dims(
                              tf.math.sqrt(
            fftshift(input, (-2, -1))), 3)
        ))
    return padded, input

def _fromgrid(img):
    """
    Reshape (-1, gridsize, gridsize, N, N) to (-1, N, N, 1)
    """
    N = params()['N']
    return tf.reshape(img, (-1, N, N, 1))

def _togrid(img, gridsize = None, N = None):
    """
    Reshape (-1, N, N, 1) to (-1, gridsize, gridsize, N, N, 1)

    i.e. from flat format to grid format
    """
    if gridsize is None:
        gridsize = params()['gridsize']
    if N is None:
        N = params()['N']
    return tf.reshape(img, (-1, gridsize, gridsize, N, N, 1))

def togrid(*imgs):
    """
    Reshape (-1, N, N, 1) to (-1, gridsize, gridsize, N, N)
    """
    return [_togrid(img) for img in imgs]

def _grid_to_channel(grid):
    """
    Reshape (-1, gridsize, gridsize, N, N) to (-1, N, N, gridsize * gridsize)
    """
    gridsize = params()['gridsize']
    img = tf.transpose(grid, [0, 3, 4, 1, 2, 5], conjugate=False)
    _, ww, hh = img.shape[:3]
    img = tf.reshape(img, (-1, ww, hh, gridsize**2))
    return img

def grid_to_channel(*grids):
    return [_grid_to_channel(g) for g in grids]

def _flat_to_channel(img, N = None):
    # TODO N should be picked up automatically
    gridsize = params()['gridsize']
    if N is None:
        N = params()['N']
    img = tf.reshape(img, (-1, gridsize**2, N, N))
    img = tf.transpose(img, [0, 2, 3, 1], conjugate=False)
    return img

# TODO rename
def _channel_to_flat(img):
    """
    Reshape (b, N, N, gridsize * gridsize) to (-1, N, N, 1)
    """
    _, h, w, c = img.shape
    #assert h == w == params()['N']
    img = tf.transpose(img, [0, 3, 1, 2], conjugate=False)
    img = tf.reshape(img, (-1, h, w, 1))
    return img

def _channel_to_patches(channel):
    """
    reshape (-1, N, N, gridsize * gridsize) to (-1, gridsize, gridsize, N**2)
    """
    gridsize = params()['gridsize']
    N = params()['N']
    img = tf.transpose(channel, [0, 3, 1, 2], conjugate=False)
    img = tf.reshape(img, (-1, gridsize, gridsize, N**2))
    return img

def channel_to_flat(*imgs):
    return [_channel_to_flat(g) for g in imgs]

def extract_patches(x, N, offset):
    return tf.image.extract_patches(
        x,
        [1, N, N, 1],
        [1, offset,offset, 1],
        [1, 1, 1, 1],
        padding="VALID"
    )

def extract_outer(img, fmt = 'grid',
        bigN = None, bigoffset = None, test = False):#,
        #test = False):
    """
        Extract big patches (overlapping bigN x bigN regions over an
        entire input img)
    """
    print('is test:', test)
    if bigN is None:
        bigN = get('bigN')
    if bigoffset is None:
        bigoffset = cfg['bigoffset'] // 2
#        if test:
#            bigoffset = cfg['bigoffset'] // 2
#        else:
#            bigoffset = cfg['bigoffset']
    assert img.shape[-1] == 1
    # Reason for the stride of the outer patches to be half of the grid
    # spacing is so that the patches have sufficient overlap (i.e., we
    # know that the boundary of a solution region will not be properly
    # reconstructed, so it's necessary to have overlaps)
    grid = tf.reshape(
        extract_patches(img, bigN, bigoffset),
        #extract_patches(img, padded_size, bigoffset // 2),
        (-1, bigN, bigN, 1))
        #(-1, padded_size, padded_size, 1))
    if fmt == 'flat':
        return _fromgrid(grid)
    elif fmt == 'grid':
        return grid
    elif fmt == 'channel':
        return _grid_to_channel(grid)
    else:
        raise ValueError

def extract_inner_grid(grid):
    N = cfg['N']
    offset = params()['offset']
    return extract_patches(grid, N, offset)

# TODO turn extract_inner_fn into a positional argument
def extract_nested_patches(img, fmt = 'flat',
        extract_inner_fn = extract_inner_grid,
        **kwargs):
    """
    Extract small patches (overlapping N x N regions on a gridsize x gridsize
        grid) within big patches (overlapping bigN x bigN regions over the
        entire input img)

    fmt == 'channel': patches within a solution region go in the channel dimension
    fmt == 'flat': patches within a solution go in the batch dimension; size of output
        channel dimension is 1
    fmt == 'grid': ...

    This function and extract_outer are only used to extract nominal
    coordinates, so it is not necessary for them to use jitter padding
    """
    N = cfg['N']
    offset = params()['offset']
    gridsize = params()['gridsize']
    assert img.shape[-1] == 1
    # First, extract 'big' patches, each of which is a grid of
    # overlapping solution regions.
    outer_grid = extract_outer(img, fmt = 'grid', **kwargs)
    # Then, extract individual solution regions within each patch
    grid = tf.reshape(
        extract_inner_fn(outer_grid),
        (-1, gridsize, gridsize, N, N, 1))
    if fmt == 'flat':
        return _fromgrid(grid)
    elif fmt == 'grid':
        return grid
    elif fmt == 'channel':
        return _grid_to_channel(grid)#, outer_grid # TODO second output is for debugging
    else:
        raise ValueError

def mk_extract_inner_position(offsets_xy):
    def inner(grid):
        return extract_patches_position(grid, offsets_xy),
    return inner

def extract_nested_patches_position(img, offsets_xy, fmt = 'flat',
        **kwargs):
    """
    Extract small patches (overlapping N x N regions on a gridsize x gridsize
        grid) within big patches (overlapping bigN x bigN regions over the
        entire input img)

    fmt == 'channel': patches within a solution region go in the channel dimension
    fmt == 'flat': patches within a solution go in the batch dimension; size of output
        channel dimension is 1
    fmt == 'grid': ...
    """
    return extract_nested_patches(img, fmt = fmt,
        extract_inner_fn = mk_extract_inner_position(offsets_xy),
        **kwargs)

@tf.function
def extract_patches_inverse(y, N, average, gridsize = None, offset = None):
    if gridsize is None:
        gridsize = params()['gridsize']
    if offset is None:
        offset = params()['offset']
    target_size = N + (gridsize - 1) * offset
    b = tf.shape(y)[0]

    _x = tf.zeros((b, target_size, target_size, 1), dtype = y.dtype)
    _y = extract_patches(_x, N, offset)
    if average:
        # Divide by grad, to "average" together the overlapping patches
        # otherwise they would simply sum up
        grad = tf.gradients(_y, _x)[0]
        return tf.gradients(_y, _x, grad_ys=y)[0] / grad
    else:
        return tf.gradients(_y, _x, grad_ys=y)[0]

def reassemble_patches_real(channels, average = True, **kwargs):
    """
    Given image patches (shaped such that the channel dimension indexes
    patches within a single solution region), reassemble into an image
    for the entire solution region. Overlaps between patches are
    averaged.
    """
    real = _channel_to_patches(channels)
    N = params()['N']
    return extract_patches_inverse(real, N, average, **kwargs)

def pad_patches(imgs, padded_size):
    padded_size = get_padded_size()
    return tfkl.ZeroPadding2D(((padded_size - N) // 2, (padded_size - N) // 2))(imgs)

def trim_reconstruction(x, N = None):
    """
    Trim from shape (_, M, M, _) to (_, N, N, _), where M >= N

    When dealing with an input with a static shape, assume M = get_padded_size()
    """
    if N is None:
        N = cfg['N']
    shape = x.shape
    #shape = tf.shape(x)
    if shape[1] is not None:
        assert int(shape[1]) == int(shape[2])
    try:
        clipsize = (int(shape[1]) - N) // 2
    except TypeError:
        clipsize = (get_padded_size() - N) // 2
    return x[:, clipsize: -clipsize,
            clipsize: -clipsize, :]

def extract_patches_position(imgs, offsets_xy, jitter = 0.):
    """
    Expects offsets_xy in channel format.

    imgs must be in flat format with a single image per solution region, i.e.
    (batch size, M, M, 1) where M = N + some padding size.
    """
    #pdb.set_trace()
    #print(imgs.shape, offsets_xy.shape
    if  imgs.get_shape()[0] is not None:
        assert int(imgs.get_shape()[0]) == int(offsets_xy.get_shape()[0])
    assert int(imgs.get_shape()[3]) == 1
    assert int(offsets_xy.get_shape()[2]) == 2
    assert int(imgs.get_shape()[3]) == 1
    gridsize = params()['gridsize']
    assert int(offsets_xy.get_shape()[3]) == gridsize**2
    offsets_flat = flatten_offsets(offsets_xy)
    stacked = tf.repeat(imgs, gridsize**2, axis = 3)
    flat_padded = _channel_to_flat(stacked)
    channels_translated = trim_reconstruction(
        Translation()([flat_padded, offsets_flat, jitter]))
    return channels_translated

def center_channels(channels, offsets_xy):
    """
    Undo image patch offsets
    """
    ct = Translation()([_channel_to_flat(channels), flatten_offsets(-offsets_xy), 0.])
    channels_centered = _flat_to_channel(ct)
    return channels_centered

# TODO use this everywhere where applicable
def complexify_function(fn):
    """
    Turn a function of a real tensorflow floating point data type into its
    complex version.

    It's assumed that the first argument is complex and must be
    converted before calling fn on its real and imaginary components.
    All other arguments are left unchanged.
    """
    def newf(*args, **kwargs):
        channels = args[0]
        if channels.dtype == tf.complex64:
            real = tf.math.real(channels)
            imag = tf.math.imag(channels)
            assembled_real = fn(real, *args[1:], **kwargs)
            assembled_imag = fn(imag, *args[1:], **kwargs)
            return tf.dtypes.complex(assembled_real, assembled_imag)
        else:
            return fn(*args, **kwargs)
    return newf

from tensorflow_addons.image import translate
translate = complexify_function(translate)
class Translation(tf.keras.layers.Layer):
    def __init__(self):
        super(Translation, self).__init__()
    def call(self, inputs):
        imgs, offsets, jitter = inputs
        jitter = tf.random.normal(tf.shape(offsets), stddev = jitter)
        #return translate(imgs, offsets, interpolation = 'nearest')
        return translate(imgs, offsets + jitter, interpolation = 'bilinear')

def flatten_offsets(channels):
    return _channel_to_flat(channels)[:, 0, :, 0]

def pad_reconstruction(channels):
    padded_size = get_padded_size()
    imgs_flat = _channel_to_flat(channels)
    return pad_patches(imgs_flat, padded_size)

def _reassemble_patches_position_real(imgs, offsets_xy, agg = True, **kwargs):
    """
    Pass this function as an argument to reassemble_patches by wrapping it, e.g.:
        def reassemble_patches_position_real(imgs, **kwargs):
            return _reassemble_patches_position_real(imgs, coords)
    """
    padded_size = get_padded_size()
    offsets_flat = flatten_offsets(offsets_xy)
    imgs_flat = _channel_to_flat(imgs)
    imgs_flat_bigN = pad_patches(imgs_flat, padded_size)
    imgs_flat_bigN_translated = Translation()([imgs_flat_bigN, -offsets_flat, 0.])
    if agg:
        imgs_merged = tf.reduce_sum(
                _flat_to_channel(imgs_flat_bigN_translated, N = padded_size),
                    axis = 3)[..., None]
        return imgs_merged
    else:
        print('no aggregation in patch reassembly')
        return _flat_to_channel(imgs_flat_bigN_translated, N = padded_size)

gridsize = params()['gridsize']
N = params()['N']

def mk_norm(channels, fn_reassemble_real):
    # TODO global / local
    N = params()['N']
    gridsize = params()['gridsize']
    b = tf.shape(channels)[0]
    #b = channels.shape[0]
    ones = tf.ones((b, N // 2, N // 2, gridsize**2))
    ones =   tfkl.ZeroPadding2D((N // 4, N // 4))(ones)
    assembled_ones = fn_reassemble_real(ones, average = False)
    norm = assembled_ones + .001
    return norm

def reassemble_patches(channels, fn_reassemble_real = reassemble_patches_real,
        average = False, **kwargs):
    """
    Given image patches (shaped such that the channel dimension indexes
    patches within a single solution region), reassemble into an image
    for the entire solution region. Overlaps between patches are
    averaged.
    """
    # TODO:
#    fn_reassemble_real_complex = complexify_function(fn_reassemble_real)
    real = tf.math.real(channels)
    imag = tf.math.imag(channels)
    assembled_real = fn_reassemble_real(real, average = average, **kwargs) / mk_norm(real,
        fn_reassemble_real)
    assembled_imag = fn_reassemble_real(imag, average = average, **kwargs) / mk_norm(imag,
        fn_reassemble_real)
    return tf.dtypes.complex(assembled_real, assembled_imag)

# TODO compare agg true false
def mk_reassemble_position_real(input_positions, **outer_kwargs):
    def reassemble_patches_position_real(imgs, **kwargs):
        return _reassemble_patches_position_real(imgs, input_positions,
            **outer_kwargs)
    return reassemble_patches_position_real

def reassemble_patches_position(channels, offsets_xy,
        average = False, **kwargs):
    fn_reassemble_real = mk_reassemble_position_real(offsets_xy, **kwargs)
    return reassemble_patches(channels,
        fn_reassemble_real = fn_reassemble_real,
        average = False)

def reassemble_nested_average(output_tensor, cropN = None, M = None, n_imgs = 1,
        offset = 4):
    """
    Stitch reconstruction patches from (first) model output into full
    reconstructed images, averaging the overlaps
    """
    assert len(output_tensor.shape) == 4
    #assert output_tensor.shape[-1] == 1
    bsize = int(output_tensor.shape[0] / n_imgs)
    output_tensor = output_tensor[:bsize, ...]
    if M is None:
        # assume only one image
        M = int(np.sqrt(bsize))
    if cropN is None:
        cropN = params.params()['cropN']
    patches = _togrid(trim_reconstruction(output_tensor, cropN), gridsize = M,
        N = cropN)
    patches = tf.reshape(patches, (-1, M, M, cropN**2))
    obj_recon = complexify_function(extract_patches_inverse)(patches, cropN,
        True, gridsize = M, offset = offset)
    return obj_recon

def Conv_Pool_block(x0,nfilters,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last'):
    x0 = Conv2D(nfilters, (w1, w2), activation='relu', padding=padding, data_format=data_format)(x0)
    x0 = Conv2D(nfilters, (w1, w2), activation='relu', padding=padding, data_format=data_format)(x0)
    x0 = MaxPool2D((p1, p2), padding=padding, data_format=data_format)(x0)
    return x0

def Conv_Up_block(x0,nfilters,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last',
        activation = 'relu'):
    x0 = Conv2D(nfilters, (w1, w2), activation='relu', padding=padding, data_format=data_format)(x0)
    x0 = Conv2D(nfilters, (w1, w2), activation=activation, padding=padding, data_format=data_format)(x0)
    x0 = UpSampling2D((p1, p2), data_format=data_format)(x0)
    return x0

########
## Loss functions
########
# TODO move these to another file
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]
    return x_var, y_var

def total_variation_loss(target, pred):
    pred = Lambda(lambda x: tf.math.abs(x))(pred)
    #pred = tf.keras.layers.AveragePooling2D(padding = 'valid')(pred)
    x_deltas, y_deltas = high_pass_x_y(pred)
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)

pp = tfk.Sequential([
    Lambda(lambda x: tf.image.grayscale_to_rgb(x)),
])
def perceptual_loss(target, pred):
    """
    """
    target = pp(target)
    pred = pp(pred)

    # vgg = VGG16(weights='imagenet', include_top=False, input_shape=(N // 2,N // 2,3))
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(N, N, 3))
    vgg.trainable = False

    outputs = [vgg.get_layer('block2_conv2').output]
    feat_model = Model(vgg.input, outputs)
# feat_model.trainable = False
    activatedModelVal = feat_model(pred)
    actualModelVal = feat_model(target)
    return meanSquaredLoss(gram_matrix(actualModelVal),gram_matrix(activatedModelVal))

def meanSquaredLoss(y_true,y_pred, center_target = True):
    return tf.reduce_mean(tf.keras.losses.MSE(y_true,y_pred))

# TODO this doesn't work if the intensity scale is set to trainable
def masked_MAE_loss(target, pred):
    """
    bigN
    """
    mae = tf.keras.metrics.mean_absolute_error
    mask = params()['probe_mask']
    pred = trim_reconstruction(
            reassemble_patches(mask * pred))
    target = trim_reconstruction(
            reassemble_patches(tf.math.abs(mask) * target))
    return mae(target, pred)

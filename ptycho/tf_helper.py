import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
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

from .params import params, cfg, get_bigN

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

def fromgrid(*imgs):
    """
    Reshape (-1, gridsize, gridsize, N, N) to (-1, N, N, 1)
    """
    return [_fromgrid(img) for img in imgs]

def _togrid(img):
    """
    Reshape (-1, N, N, 1) to (-1, gridsize, gridsize, N, N)
    """
    gridsize = params()['gridsize']
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
    gridsize = params()['gridsize']
    if N is None:
        N = params()['N']
    img = tf.reshape(img, (-1, gridsize**2, N, N))
    img = tf.transpose(img, [0, 2, 3, 1], conjugate=False)
    return img

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

def extract_outer(img, fmt = 'grid'):
    """
        Extract big patches (overlapping bigN x bigN regions over an
        entire input img)
    """
    bigN = get_bigN()
    bigoffset = cfg['bigoffset']
    assert img.shape[-1] == 1
    # Reason for the stride of the outer patches to be half of the grid
    # spacing is so that the patches have sufficient overlap (i.e., we
    # know that the boundary of a solution region will not be properly
    # reconstructed, so it's necessary to have overlaps)
    grid = tf.reshape(
        extract_patches(img, bigN, bigoffset // 2),
        (-1, bigN, bigN, 1))
    if fmt == 'flat':
        return _fromgrid(grid)
    elif fmt == 'grid':
        return grid
    elif fmt == 'channel':
        return _grid_to_channel(grid)
    else:
        raise ValueError

def extract_nested_patches(img, fmt = 'flat'):
    """
    Extract small patches (overlapping N x N regions on a gridsize x gridsize
        grid) within big patches (overlapping bigN x bigN regions over the
        entire input img)

    fmt == 'channel': patches within a solution region go in the channel dimension
    fmt == 'flat': patches within a solution go in the batch dimension; size of output
        channel dimension is 1
    fmt == 'grid': ...
    """
    N = cfg['N']
    offset = params()['offset']
    gridsize = params()['gridsize']
    assert img.shape[-1] == 1
    # First, extract 'big' patches, each of which is a grid of
    # overlapping solution regions.
    grid = extract_outer(img, fmt = 'grid')
    # Then, extract individual solution regions within each patch
    grid = tf.reshape(
        extract_patches(grid, N, offset),
        (-1, gridsize, gridsize, N, N, 1))
    if fmt == 'flat':
        return _fromgrid(grid)
    elif fmt == 'grid':
        return grid
    elif fmt == 'channel':
        return _grid_to_channel(grid)
    else:
        raise ValueError

@tf.function
def extract_patches_inverse(inputs, gridsize = None, offset = None):
    N = params()['N']
    if gridsize is None:
        gridsize = params()['gridsize']
    if offset is None:
        offset = params()['offset']
    # TODO don't pass inputs this way
    y, N, average = inputs
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

def reassemble_patches_real(channels, average = True):
    """
    Given image patches (shaped such that the channel dimension indexes
    patches within a single solution region), reassemble into an image
    for the entire solution region. Overlaps between patches are
    averaged.
    """
    real = _channel_to_patches(channels)
    N = params()['N']
    return extract_patches_inverse((real, N, average))


def pad_bigN(imgs):
    bigN = get_bigN()
    return tfkl.ZeroPadding2D(((bigN - N) // 2, (bigN - N) // 2))(imgs)

def trim_reconstruction(x):
    """Trim from bigN x bigN to N x N
    """
    gridsize = params()['gridsize']
    offset = params()['offset']
    return x[:, (offset * (gridsize - 1)) // 2: -(offset * (gridsize - 1)) // 2,
            (offset * (gridsize - 1)) // 2: -(offset * (gridsize - 1)) // 2, :]

def extract_patches_position(imgs, offsets_xy):
    gridsize = params()['gridsize']
    offsets_flat = flatten_offsets(offsets_xy)
    stacked = tf.repeat(imgs, gridsize**2, axis = 3)
    flat_padded = _channel_to_flat(stacked)
    channels_translated = trim_reconstruction(
        Translation()([flat_padded, offsets_flat]))
    return channels_translated

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
        imgs, offsets = inputs
        return translate(imgs, offsets, interpolation = 'bilinear')

def flatten_offsets(channels):
    return _channel_to_flat(channels)[:, 0, :, 0]

def _reassemble_patches_position_real(imgs, offsets_xy):
    """
    Pass this function as an argument to reassemble_patches by wrapping it, e.g.:
        def reassemble_patches_position_real(imgs, **kwargs):
            return _reassemble_patches_position_real(imgs, coords)
    """
    bigN = get_bigN()
    offsets_flat = flatten_offsets(offsets_xy)
    imgs_flat = _channel_to_flat(imgs)
    imgs_flat_bigN = pad_bigN(imgs_flat)
    imgs_flat_bigN_translated = Translation()([imgs_flat_bigN, -offsets_flat])
    imgs_merged = tf.reduce_sum(
            _flat_to_channel(imgs_flat_bigN_translated, N = bigN),
                axis = 3)[..., None]
    return imgs_merged

gridsize = params()['gridsize']
N = params()['N']
ones = tf.ones((1, N // 2, N // 2, gridsize**2))
ones =   tfkl.ZeroPadding2D((N // 4, N // 4))(ones)
assembled_ones = reassemble_patches_real(ones, False)
norm = assembled_ones + .001

def reassemble_patches(channels, fn_reassemble_real = reassemble_patches_real, average = False):
    """
    Given image patches (shaped such that the channel dimension indexes
    patches within a single solution region), reassemble into an image
    for the entire solution region. Overlaps between patches are
    averaged.
    """
    # TODO dividing out by norm increased the training time quite
    # substantially TODO is that true?
    real = tf.math.real(channels)
    imag = tf.math.imag(channels)
    assembled_real = fn_reassemble_real(real, average = average) / norm
    assembled_imag = fn_reassemble_real(imag, average = average) / norm
    return tf.dtypes.complex(assembled_real, assembled_imag)

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

def meanSquaredLoss(y_true,y_pred):
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

#def symmetrized_loss(target, pred, loss_fn):
#    """
#    Calculate loss function on an image, taking into account that the
#    prediction may be coordinate-inverted relative to the target
#    """
#    abs1 = (target)
#    abs2 = (pred)
#    abs3 = abs2[:, ::-1, ::-1, :]
#    target_sym = (symmetrize_3d(target))
#    a, b, c = loss_fn(abs1, abs2), loss_fn(abs1, abs3), loss_fn(target_sym, pred)
#    return tf.minimum(a,
#                      tf.minimum(b, c))

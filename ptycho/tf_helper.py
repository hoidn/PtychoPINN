import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
from tensorflow.signal import fft
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from skimage.transform import resize as sresize
from tensorflow.signal import fft2d, fftshift
#Keras modules
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, UpSampling2D
from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Lambda
from .params import params, cfg, get_bigN

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

support_threshold = .0
@tf.function
def get_mask(input, support_threshold):
    mask = tf.where(input > support_threshold, tf.ones_like(input),
                    tf.zeros_like(input))
    return mask

def resize(x):
    rmod = do_resize()
    rmod.compile(loss = 'mse')
    return rmod.predict(x)

def do_resize(N):
    N = params()['N']
    transform = tfkl.AveragePooling2D(2)
    return tfk.Sequential([
        tfk.Input(shape = (N, N, 1)),
        transform
    ])

@tf.function
def combine_complex(amp, phi):
    output = tf.cast(amp, tf.complex64) * tf.exp(
        1j * tf.cast(phi, tf.complex64))
    return output

def pad_obj(input, h, w):
    return tfkl.ZeroPadding2D((h // 4, w // 4), name = 'padded_obj')(input)

## TODO nested lambdas?
#@tf.function
#def pad_and_diffract(input, h, w, pad = True):
#    """
#    zero-pad the real-space object and then calculate the far field
#    diffraction amplitude
#    """
#    if pad:
#        input = pad_obj(input, h, w)
#    padded = input
#    assert input.shape[-1] == 1
#    input = (Lambda(lambda resized: (fft2d(
#        #tf.squeeze # this destroys shape information so need to use slicing instead
#        (tf.cast(resized, tf.complex64))[..., 0]
#        ))))(input)
#    input = (Lambda(lambda X: tf.math.real(tf.math.conj(X) * X) / (h * w)))(input)
#    input = (Lambda(lambda psd:
#                          tf.expand_dims(
#                              tf.math.sqrt(
#            fftshift(psd, (-2, -1))
#                                   ), 3),
#        name = 'pred_amplitude'))(input)
#    return padded, input

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
            fftshift(input, (-2, -1))
                                   ), 3)
        ))
    return padded, input

@tf.function
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

@tf.function
def _flat_to_channel(img):
    gridsize = params()['gridsize']
    h = params()['h']
    w = params()['w']
    img = tf.reshape(img, (-1, gridsize**2, h, w))
    img = tf.transpose(img, [0, 2, 3, 1], conjugate=False)
    return img

def _channel_to_flat(img):
    """
    Reshape (b, N, N, gridsize * gridsize) to (-1, N, N, 1)
    """
    _, h, w, c = img.shape
    assert h == w == params()['N']
    img = tf.transpose(img, [0, 3, 1, 2], conjugate=False)
    img = tf.reshape(img, (-1, h, w, 1))
    return img

@tf.function
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

@tf.function
def extract_patches(x, N, offset):
    return tf.compat.v1.extract_image_patches(
        x,
        [1, N, N, 1],
        [1, offset,offset, 1],
        [1, 1, 1, 1],
        padding="VALID"
    )

@tf.function
def extract_patches_inverse(inputs):
    N = params()['N']
    gridsize = params()['gridsize']
    offset = params()['offset']
    target_size = N + (gridsize - 1) * offset
    y, N, offset, average = inputs
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

@tf.function
def reassemble_patches_real(channels, average = True):
    """
    Given image patches (shaped such that the channel dimension indexes
    patches within a single solution region), reassemble into an image
    for the entire solution region. Overlaps between patches are
    averaged.
    """
    real = _channel_to_patches(channels)
    N = params()['N']
    offset = params()['offset']
    return extract_patches_inverse((real, N, offset, average))

gridsize = params()['gridsize']
N = params()['N']
ones = tf.ones((1, N // 2, N // 2, gridsize**2))
ones =   tfkl.ZeroPadding2D((N // 4, N // 4))(ones)
assembled_ones = reassemble_patches_real(ones, False)
norm = assembled_ones + .001

@tf.function
def reassemble_patches(channels, average = False):
    """
    Given image patches (shaped such that the channel dimension indexes
    patches within a single solution region), reassemble into an image
    for the entire solution region. Overlaps between patches are
    averaged.
    """
    # TODO dividing out by norm increased the training time quite
    # substantially
    real = tf.math.real(channels)
    imag = tf.math.imag(channels)
    assembled_real = reassemble_patches_real(real, average = average) / norm
    assembled_imag = reassemble_patches_real(imag, average = average) / norm
    return tf.dtypes.complex(assembled_real, assembled_imag)

#@tf.function
#def reassemble_patches(channels, average = True):
#    """
#    Given image patches (shaped such that the channel dimension indexes
#    patches within a single solution region), reassemble into an image
#    for the entire solution region. Overlaps between patches are
#    averaged.
#    """
#    patches = _channel_to_patches(channels)
#    real = tf.math.real(patches)
#    imag = tf.math.imag(patches)
#    N = params()['N']
#    offset = params()['offset']
#    assembled_real = extract_patches_inverse((real, N, offset, average))
#    assembled_imag = extract_patches_inverse((imag, N, offset, average))
#    return tf.dtypes.complex(assembled_real, assembled_imag)

@tf.function
def extract_nested_patches(img, fmt = 'flat'):
    bigN = get_bigN()
    bigoffset = cfg['bigoffset']
    N = cfg['N']
    offset = params()['offset']
    gridsize = params()['gridsize']
    # First, extract 'big' patches, each of which is a grid of overlapping solution regions
    grid = tf.reshape(
        tf.compat.v1.extract_image_patches(img, [1, bigN, bigN, 1], [1, bigoffset // 2, bigoffset // 2, 1],
                                              [1, 1, 1, 1], padding = 'VALID'),
        (-1, bigN, bigN, 1))
    # Then, extract individual solution regions within each patch
    grid = tf.reshape(
        tf.compat.v1.extract_image_patches(grid, [1, N, N, 1], [1, offset, offset, 1],
                                              [1, 1, 1, 1], padding = 'VALID'),
        (-1, gridsize, gridsize, N, N, 1))
    if fmt == 'flat':
        return _fromgrid(grid)
    elif fmt == 'grid':
        return grid
    elif fmt == 'channel':
        return _grid_to_channel(grid)
    else:
        raise ValueError

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
    pred = tf.keras.layers.AveragePooling2D(padding = 'valid')(pred)
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

    activatedModelVal = feat_model(pred)
    actualModelVal = feat_model(target)
    return meanSquaredLoss(gram_matrix(actualModelVal),gram_matrix(activatedModelVal))

def symmetrized_loss(target, pred, loss_fn):
    """
    Calculate loss function on an image, taking into account that the
    prediction may be coordinate-inverted relative to the target
    """
    abs1 = (target)
    abs2 = (pred)
    abs3 = abs2[:, ::-1, ::-1, :]
    target_sym = (symmetrize_3d(target))
    a, b, c = loss_fn(abs1, abs2), loss_fn(abs1, abs3), loss_fn(target_sym, pred)
    return tf.minimum(a,
                      tf.minimum(b, c))

def amplitude_difference(target, pred):
    """
    Calculate object MAE, taking into account that the prediction may be
    inverted
    """
    abs1 = tf.math.abs(target)
    abs2 = tf.math.abs(pred)
    return symmetrized_loss(target, pred, tf.keras.losses.MeanAbsoluteError())

def symmetrized_perceptual_loss(target, pred):
    return symmetrized_loss(target, pred, perceptual_loss)

def meanSquaredLoss(y_true,y_pred):
    return tf.reduce_mean(tf.keras.losses.MSE(y_true,y_pred))

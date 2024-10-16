import os
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Union, Callable, Any

# Check if there are any GPUs available and set memory growth accordingly
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU found, using CPU instead.")


import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, UpSampling2D
from tensorflow.keras.layers import Lambda
from tensorflow.signal import fft2d, fftshift
import tensorflow_probability as tfp

from .params import params, cfg, get, get_padded_size
#from .logging import debug
from .autotest.debug import debug

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

support_threshold = .0
#@debug
def get_mask(input: tf.Tensor, support_threshold: float) -> tf.Tensor:
    mask = tf.where(input > support_threshold, tf.ones_like(input),
                    tf.zeros_like(input))
    return mask

#@debug
def combine_complex(amp: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
    output = tf.cast(amp, tf.complex64) * tf.exp(
        1j * tf.cast(phi, tf.complex64))
    return output

#@debug
def pad_obj(input: tf.Tensor, h: int, w: int) -> tf.Tensor:
    return tfkl.ZeroPadding2D((h // 4, w // 4), name = 'padded_obj')(input)

#@debug
def pad_and_diffract(input: tf.Tensor, h: int, w: int, pad: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    zero-pad the real-space object and then calculate the far field
    diffraction amplitude.

    Uses sysmmetric FT - L2 norm is conserved
    """
    input = tf.ensure_shape(input, (None, h, w, 1))
    print('input shape', input.shape)
    if pad:
        input = pad_obj(input, h, w)
    padded = input
    assert input.shape[-1] == 1
    input = (((fft2d(
        (tf.cast((input), tf.complex64))[..., 0]
        ))))
    input = (( tf.math.real(tf.math.conj((input)) * input) / (h * w)))
    input = (( tf.expand_dims(
                              tf.math.sqrt(
            fftshift(input, (-2, -1))), 3)
        ))
    return padded, input

#@debug
def _fromgrid(img: tf.Tensor) -> tf.Tensor:
    """
    Reshape (-1, gridsize, gridsize, N, N) to (-1, N, N, 1)
    """
    print("Debug: Entering _fromgrid function")
    N = params()['N']
    return tf.reshape(img, (-1, N, N, 1))

#@debug
def _togrid(img: tf.Tensor, gridsize: Optional[int] = None, N: Optional[int] = None) -> tf.Tensor:
    """
    Reshape (b * gridsize * gridsize, N, N, 1) to (b, gridsize, gridsize, N, N, 1)

    i.e. from flat format to grid format
    """
    if gridsize is None:
        gridsize = params()['gridsize']
    if N is None:
        N = params()['N']
    return tf.reshape(img, (-1, gridsize, gridsize, N, N, 1))

#@debug
def togrid(*imgs: tf.Tensor) -> Tuple[tf.Tensor, ...]:
    """
    Reshape (-1, N, N, 1) to (-1, gridsize, gridsize, N, N)
    """
    return [_togrid(img) for img in imgs]

#@debug
def _grid_to_channel(grid: tf.Tensor) -> tf.Tensor:
    """
    Reshape (-1, gridsize, gridsize, N, N) to (-1, N, N, gridsize * gridsize)
    """
    gridsize = params()['gridsize']
    img = tf.transpose(grid, [0, 3, 4, 1, 2, 5], conjugate=False)
    _, ww, hh = img.shape[:3]
    img = tf.reshape(img, (-1, ww, hh, gridsize**2))
    return img

#@debug
def grid_to_channel(*grids: tf.Tensor) -> Tuple[tf.Tensor, ...]:
    return [_grid_to_channel(g) for g in grids]

#@debug
def _flat_to_channel(img: tf.Tensor, N: Optional[int] = None) -> tf.Tensor:
    gridsize = params()['gridsize']
    if N is None:
        N = params()['N']
    img = tf.reshape(img, (-1, gridsize**2, N, N))
    img = tf.transpose(img, [0, 2, 3, 1], conjugate=False)
    return img

#@debug
def _flat_to_channel_2(img: tf.Tensor) -> tf.Tensor:
    gridsize = params()['gridsize']
    _, N, M, _ = img.shape
    img = tf.reshape(img, (-1, gridsize**2, N, M))
    img = tf.transpose(img, [0, 2, 3, 1], conjugate=False)
    return img

#@debug
def _channel_to_flat(img: tf.Tensor) -> tf.Tensor:
    """
    Reshape (b, N, N, c) to (b * c, N, N, 1)
    """
    shape = tf.shape(img)
    b, h, w, c = shape[0], shape[1], shape[2], shape[3]
    #_, h, w, c = img.shape
    img = tf.transpose(img, [0, 3, 1, 2], conjugate=False)
    img = tf.reshape(img, (-1, h, w, 1))
    return img

#@debug
def _channel_to_patches(channel: tf.Tensor) -> tf.Tensor:
    """
    reshape (-1, N, N, gridsize * gridsize) to (-1, gridsize, gridsize, N**2)
    """
    gridsize = params()['gridsize']
    N = params()['N']
    img = tf.transpose(channel, [0, 3, 1, 2], conjugate=False)
    img = tf.reshape(img, (-1, gridsize, gridsize, N**2))
    return img

#@debug
def channel_to_flat(*imgs: tf.Tensor) -> Tuple[tf.Tensor, ...]:
    return [_channel_to_flat(g) for g in imgs]

#@debug
def extract_patches(x: tf.Tensor, N: int, offset: int) -> tf.Tensor:
    return tf.image.extract_patches(
        x,
        [1, N, N, 1],
        [1, offset,offset, 1],
        [1, 1, 1, 1],
        padding="VALID"
    )

#@debug
def extract_outer(img: tf.Tensor, fmt: str = 'grid',
        bigN: Optional[int] = None, outer_offset: Optional[int] = None) -> tf.Tensor:#,
    """
        Extract big patches (overlapping bigN x bigN regions over an
        entire input img)
    """
    if bigN is None:
        bigN = get('bigN')
    assert img.shape[-1] == 1
    grid = tf.reshape(
        extract_patches(img, bigN, outer_offset // 2),
        (-1, bigN, bigN, 1))
    if fmt == 'flat':
        return _fromgrid(grid)
    elif fmt == 'grid':
        return grid
    elif fmt == 'channel':
        return _grid_to_channel(grid)
    else:
        raise ValueError

#@debug
def extract_inner_grid(grid: tf.Tensor) -> tf.Tensor:
    N = cfg['N']
    offset = params()['offset']
    return extract_patches(grid, N, offset)

#@debug
def extract_nested_patches(img: tf.Tensor, fmt: str = 'flat',
        extract_inner_fn: Callable[[tf.Tensor], tf.Tensor] = extract_inner_grid,
        **kwargs: Any) -> tf.Tensor:
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
    outer_grid = extract_outer(img, fmt = 'grid', **kwargs)
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

#@debug
def mk_extract_inner_position(offsets_xy: tf.Tensor) -> Callable[[tf.Tensor], Tuple[tf.Tensor]]:
    #@debug
    def inner(grid: tf.Tensor) -> Tuple[tf.Tensor]:
        return extract_patches_position(grid, offsets_xy),
    return inner

#@debug
def extract_nested_patches_position(img: tf.Tensor, offsets_xy: tf.Tensor, fmt: str = 'flat',
        **kwargs: Any) -> tf.Tensor:
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
#@debug
def extract_patches_inverse(y: tf.Tensor, N: int, average: bool, gridsize: Optional[int] = None, offset: Optional[int] = None) -> tf.Tensor:
    if gridsize is None:
        gridsize = params()['gridsize']
    if offset is None:
        offset = params()['offset']
    target_size = N + (gridsize - 1) * offset
    b = tf.shape(y)[0]

    _x = tf.zeros((b, target_size, target_size, 1), dtype = y.dtype)
    _y = extract_patches(_x, N, offset)
    if average:
        grad = tf.gradients(_y, _x)[0]
        return tf.gradients(_y, _x, grad_ys=y)[0] / grad
    else:
        return tf.gradients(_y, _x, grad_ys=y)[0]

#@debug
def reassemble_patches_real(channels: tf.Tensor, average: bool = True, **kwargs: Any) -> tf.Tensor:
    """
    Given image patches (shaped such that the channel dimension indexes
    patches within a single solution region), reassemble into an image
    for the entire solution region. Overlaps between patches are
    averaged.
    """
    real = _channel_to_patches(channels)
    N = params()['N']
    return extract_patches_inverse(real, N, average, **kwargs)

#@debug
def pad_patches(imgs: tf.Tensor, padded_size: Optional[int] = None) -> tf.Tensor:
    N = params()['N']
    if padded_size is None:
        padded_size = get_padded_size()
    return tfkl.ZeroPadding2D(((padded_size - N) // 2, (padded_size - N) // 2))(imgs)

#@debug
def pad(imgs: tf.Tensor, size: int) -> tf.Tensor:
    return tfkl.ZeroPadding2D((size, size))(imgs)

#@debug
def trim_reconstruction(x: tf.Tensor, N: Optional[int] = None) -> tf.Tensor:
    """
    Trim from shape (_, M, M, _) to (_, N, N, _), where M >= N

    When dealing with an input with a static shape, assume M = get_padded_size()
    """
    if N is None:
        N = cfg['N']
    shape = x.shape
    if shape[1] is not None:
        assert int(shape[1]) == int(shape[2])
    try:
        clipsize = (int(shape[1]) - N) // 2
    except TypeError:
        clipsize = (get_padded_size() - N) // 2
    return x[:, clipsize: -clipsize,
            clipsize: -clipsize, :]

#@debug
def extract_patches_position(imgs: tf.Tensor, offsets_xy: tf.Tensor, jitter: float = 0.) -> tf.Tensor:
    """
    Expects offsets_xy in channel format.

    imgs must be in flat format with a single image per solution region, i.e.
    (batch size, M, M, 1) where M = N + some padding size.

    Returns shifted images in channel format, cropped symmetrically

    no negative sign
    """
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

#@debug
def center_channels(channels: tf.Tensor, offsets_xy: tf.Tensor) -> tf.Tensor:
    """
    Undo image patch offsets
    """
    ct = Translation()([_channel_to_flat(channels), flatten_offsets(-offsets_xy), 0.])
    channels_centered = _flat_to_channel(ct)
    return channels_centered

#@debug
def is_complex_tensor(tensor: tf.Tensor) -> bool:
    """Check if the tensor is of complex dtype."""
    return tensor.dtype in [tf.complex64, tf.complex128]

#@debug
def complexify_helper(separate: Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]], combine: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]) -> Callable:
    """
    Create a "complexify" function based on the provided separation and combination methods.
    """
    #@debug
    def complexify(fn: Callable[..., tf.Tensor]) -> Callable[..., tf.Tensor]:
        #@debug
        def newf(*args: Any, **kwargs: Any) -> tf.Tensor:
            channels = args[0]
            if is_complex_tensor(channels):
                part1, part2 = separate(channels)
                assembled_part1 = fn(part1, *args[1:], **kwargs)
                assembled_part2 = fn(part2, *args[1:], **kwargs)
                return combine(assembled_part1, assembled_part2)
            else:
                return fn(*args, **kwargs)
        return newf
    return complexify

#@debug
def separate_real_imag(channels: Union[tf.Tensor, np.ndarray]) -> Tuple[Union[tf.Tensor, np.ndarray], Union[tf.Tensor, np.ndarray]]:
    return tf.math.real(channels), tf.math.imag(channels)

#@debug
def combine_real_imag(real: Union[tf.Tensor, np.ndarray], imag: Union[tf.Tensor, np.ndarray]) -> Union[tf.Tensor, np.ndarray]:
    return tf.cast(tf.dtypes.complex(real, imag), tf.complex64)

#@debug
def separate_amp_phase(channels: Union[tf.Tensor, np.ndarray]) -> Tuple[Union[tf.Tensor, np.ndarray], Union[tf.Tensor, np.ndarray]]:
    return tf.math.abs(channels), tf.math.angle(channels)

complexify_function = complexify_helper(separate_real_imag, combine_real_imag)
complexify_amp_phase = complexify_helper(separate_amp_phase, combine_complex)
complexify_sum_amp_phase = complexify_helper(separate_amp_phase, lambda a, b: a + b)
complexify_sum_real_imag = complexify_helper(separate_real_imag, lambda a, b: a + b)


from tensorflow_addons.image import translate as _translate

#from ptycho.misc import debug
@complexify_function
#@debug
def translate(imgs: tf.Tensor, offsets: tf.Tensor, **kwargs: Any) -> tf.Tensor:
    # TODO assert dimensionality of translations is 2; i.e. B, 2
    return _translate(imgs, offsets, **kwargs)

# TODO consolidate this and translate()
class Translation(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super(Translation, self).__init__()
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, float]) -> tf.Tensor:
        imgs, offsets, jitter = inputs
        jitter = tf.random.normal(tf.shape(offsets), stddev = jitter)
        return translate(imgs, offsets + jitter, interpolation = 'bilinear')

#@debug
def flatten_offsets(channels: tf.Tensor) -> tf.Tensor:
    return _channel_to_flat(channels)[:, 0, :, 0]

#@debug
def pad_reconstruction(channels: tf.Tensor) -> tf.Tensor:
    padded_size = get_padded_size()
    imgs_flat = _channel_to_flat(channels)
    return pad_patches(imgs_flat, padded_size)

#@debug
def _reassemble_patches_position_real(imgs: tf.Tensor, offsets_xy: tf.Tensor, agg: bool = True, padded_size: Optional[int] = None,
        **kwargs: Any) -> tf.Tensor:
    """
    Pass this function as an argument to reassemble_patches by wrapping it, e.g.:
        def reassemble_patches_position_real(imgs, **kwargs):
            return _reassemble_patches_position_real(imgs, coords)
    """
    if padded_size is None:
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

#@debug
def mk_centermask(inputs: tf.Tensor, N: int, c: int, kind: str = 'center') -> tf.Tensor:
    b = tf.shape(inputs)[0]
    ones = tf.ones((b, N // 2, N // 2, c), dtype = inputs.dtype)
    ones =   tfkl.ZeroPadding2D((N // 4, N // 4))(ones)
    if kind == 'center':
        return ones
    elif kind == 'border':
        return 1 - ones
    else:
        raise ValueError

#@debug
def mk_norm(channels: tf.Tensor, fn_reassemble_real: Callable[[tf.Tensor], tf.Tensor]) -> tf.Tensor:
    N = params()['N']
    gridsize = params()['gridsize']
    # TODO if probe.big is True, shouldn't the ones fill the full N x N region?
    ones = mk_centermask(channels, N, gridsize**2)
    assembled_ones = fn_reassemble_real(ones, average = False)
    norm = assembled_ones + .001
    return norm

#@debug
def reassemble_patches(channels: tf.Tensor, fn_reassemble_real: Callable[[tf.Tensor], tf.Tensor] = reassemble_patches_real,
        average: bool = False, **kwargs: Any) -> tf.Tensor:
    """
    Given image patches (shaped such that the channel dimension indexes
    patches within a single solution region), reassemble into an image
    for the entire solution region. Overlaps between patches are
    averaged.
    """
    real = tf.math.real(channels)
    imag = tf.math.imag(channels)
    assembled_real = fn_reassemble_real(real, average = average, **kwargs) / mk_norm(real,
        fn_reassemble_real)
    assembled_imag = fn_reassemble_real(imag, average = average, **kwargs) / mk_norm(imag,
        fn_reassemble_real)
    return tf.dtypes.complex(assembled_real, assembled_imag)

#@debug
def shift_and_sum(obj_tensor: np.ndarray, global_offsets: np.ndarray, M: int = 10) -> tf.Tensor:
    from . import tf_helper as hh
    assert len(obj_tensor.shape) == 4
    assert obj_tensor.dtype == np.complex64
    assert len(global_offsets.shape) == 4
    assert global_offsets.dtype == np.float64
    # Extract necessary parameters
    N = params()['N']
    # Select the central part of the object tensor
    obj_tensor = obj_tensor[:, N // 2 - M // 2: N // 2 + M // 2, N // 2 - M // 2: N // 2 + M // 2, :]
    # Calculate the center of mass of global_offsets
    center_of_mass = tf.reduce_mean(tf.cast(global_offsets, tf.float32), axis=0)
    # Adjust global_offsets by subtracting the center of mass
    adjusted_offsets = tf.cast(global_offsets, tf.float32) - center_of_mass
    # Calculate dynamic padding based on maximum adjusted offset
    max_offset = tf.reduce_max(tf.abs(adjusted_offsets))
    dynamic_pad = int(tf.cast(tf.math.ceil(max_offset), tf.int32))
    print('PADDING SIZE:', dynamic_pad)
    
    # Create a canvas to store the shifted and summed object tensors
    result = tf.zeros_like(hh.pad(obj_tensor[0:1], dynamic_pad))
    
    # Iterate over the adjusted offsets and perform shift-and-sum
    for i in range(len(adjusted_offsets)):
        # Apply dynamic padding to the current object tensor
        padded_obj_tensor = hh.pad(obj_tensor[i:i+1], dynamic_pad)
        # Squeeze and cast adjusted offset to 2D float for translation
        offset_2d = tf.cast(tf.squeeze(adjusted_offsets[i]), tf.float32)
        # Translate the padded object tensor
        translated_obj = hh.translate(padded_obj_tensor, offset_2d, interpolation='bilinear')
        # Accumulate the translated object tensor
        result += translated_obj[0]
    
    # TODO: how could we support multiple scans?
    return result[0]

#@debug
def reassemble_whole_object(patches: tf.Tensor, offsets: tf.Tensor, size: int = 226, norm: bool = False) -> tf.Tensor:
    """
    patches: tensor of shape (B, N, N, gridsize**2) containing reconstruction patches

    reassembles the NxN patches into a single size x size x 1 mage, given the
        provided offsets

    This function inverts the offsets, so it's not necessary to multiply by -1
    """
    img = tf.reduce_sum(
        reassemble_patches(patches, fn_reassemble_real=mk_reassemble_position_real(
        offsets, padded_size = size)),
        axis = 0)
    if norm:
        return img / reassemble_whole_object(tf.ones_like(patches), offsets, size = size, norm = False)
    return img

def reassemble_position(obj_tensor: np.ndarray, global_offsets: np.ndarray, M: int = 10) -> tf.Tensor:
    ones = tf.ones_like(obj_tensor)
    return shift_and_sum(obj_tensor, global_offsets, M = M) /\
        (1e-9 + shift_and_sum(ones, global_offsets, M = M))

#@debug
def mk_reassemble_position_real(input_positions: tf.Tensor, **outer_kwargs: Any) -> Callable[[tf.Tensor], tf.Tensor]:
    #@debug
    def reassemble_patches_position_real(imgs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        return _reassemble_patches_position_real(imgs, input_positions,
            **outer_kwargs)
    return reassemble_patches_position_real

#@debug
def preprocess_objects(Y_I: np.ndarray, Y_phi: Optional[np.ndarray] = None,
        offsets_xy: Optional[tf.Tensor] = None, **kwargs: Any) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Extracts normalized object patches from full real-space images, using the
    nested grid format.
    """
    _Y_I_full = Y_I
    if Y_phi is None:
        Y_phi = np.zeros_like(Y_I)

    if offsets_xy is None or tf.math.reduce_all(offsets_xy == 0):
        print('Sampling on regular grid')
        Y_I, Y_phi = \
            [extract_nested_patches(imgs, fmt= 'channel', **kwargs)
                for imgs in [Y_I, Y_phi]]
    else:
        print('Using provided scan point offsets')
        Y_I, Y_phi = \
            [extract_nested_patches_position(imgs, offsets_xy, fmt= 'channel',
                    **kwargs)
                for imgs in [Y_I, Y_phi]]

    assert Y_I.shape[-1] == get('gridsize')**2
    norm_Y_I = tf.math.reduce_max(Y_I, axis = (1, 2, 3))[:, None, None, None]
    norm_Y_I = tf.math.reduce_mean(norm_Y_I)
    Y_I /= norm_Y_I

    Y_I, Y_phi =\
        channel_to_flat(Y_I, Y_phi)
    return Y_I, Y_phi, _Y_I_full / norm_Y_I, norm_Y_I

#@debug
def reassemble_nested_average(output_tensor: tf.Tensor, cropN: Optional[int] = None, M: Optional[int] = None, n_imgs: int = 1,
        offset: int = 4) -> tf.Tensor:
    """
    Stitch reconstruction patches from (first) model output into full
    reconstructed images, averaging the overlaps
    """
    assert len(output_tensor.shape) == 4
    bsize = int(output_tensor.shape[0] / n_imgs)
    output_tensor = output_tensor[:bsize, ...]
    if M is None:
        M = int(np.sqrt(bsize))
    if cropN is None:
        cropN = params.params()['cropN']
    patches = _togrid(trim_reconstruction(output_tensor, cropN), gridsize = M,
        N = cropN)
    patches = tf.reshape(patches, (-1, M, M, cropN**2))
    obj_recon = complexify_function(extract_patches_inverse)(patches, cropN,
        True, gridsize = M, offset = offset)
    return obj_recon


#@debug
def gram_matrix(input_tensor: tf.Tensor) -> tf.Tensor:
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

#@debug
def high_pass_x_y(image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]
    return x_var, y_var

pp = tfk.Sequential([
    Lambda(lambda x: tf.image.grayscale_to_rgb(x)),
])
#@debug
def perceptual_loss(target: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    """
    """
    target = pp(target)
    pred = pp(pred)

    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(N, N, 3))
    vgg.trainable = False

    outputs = [vgg.get_layer('block2_conv2').output]
    feat_model = Model(vgg.input, outputs)
    activatedModelVal = feat_model(pred)
    actualModelVal = feat_model(target)
    return meanSquaredLoss(gram_matrix(actualModelVal),gram_matrix(activatedModelVal))

#@debug
def meanSquaredLoss(y_true: tf.Tensor, y_pred: tf.Tensor, center_target: bool = True) -> tf.Tensor:
    return tf.reduce_mean(tf.keras.losses.MSE(y_true,y_pred))

#@debug
def masked_MAE_loss(target: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
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


@complexify_sum_real_imag
#@debug
def total_variation_complex(obj: tf.Tensor) -> tf.Tensor:
    """ calculate summed total variation of the real and imaginary components
        of a tensor
    """
    x_deltas, y_deltas = high_pass_x_y(obj)
    return tf.reduce_sum(x_deltas**2) + tf.reduce_sum(y_deltas**2)

#@debug
def total_variation(obj: tf.Tensor, amp_only: bool = False) -> tf.Tensor:
    if amp_only:
        obj = Lambda(lambda x: tf.math.abs(x))(obj)
    return total_variation_complex(obj)

@complexify_sum_amp_phase
#@debug
def complex_mae(target: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    mae = tf.keras.metrics.mean_absolute_error
    return mae(target, pred)

#@debug
def masked_mae(target: tf.Tensor, pred: tf.Tensor, **kwargs: Any) -> tf.Tensor:
    N = params()['N']
    mae = tf.keras.metrics.mean_absolute_error
    pred = pred * mk_centermask(pred, N, 1, kind = 'center')
    return mae(target, pred)

#@debug
def realspace_loss(target: tf.Tensor, pred: tf.Tensor, **kwargs: Any) -> tf.Tensor:
    N = params()['N']
    if not get('probe.big'):
        pred = pred * mk_centermask(pred, N, 1, kind = 'center')

    if get('tv_weight') > 0:
        tv_loss = total_variation(pred) * get('tv_weight')
    else:
        tv_loss = 0.

    if get('realspace_mae_weight') > 0:
        mae_loss = complex_mae(target, pred) * get('realspace_mae_weight')
    else:
        mae_loss = 0.
    return tv_loss + mae_loss

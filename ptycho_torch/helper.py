#Type helpers
from typing import Tuple, Optional, Union, Callable, Any

#Additional helper functions ported from original tensorflow lib
import torch
from torch import nn
import torch.nn.functional as F

#Configuration file
cfg = {'N': 64,
       'offset': 4,
       'gridsize': 2,
       'max_position_jitter': 10
    }

#Complex functions
#---------------------
def combine_complex(amp: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    '''
    Converts real number amplitude and phase into single complex number for FFT

    Inputs
    ------
    amp: torch.Tensor
        Amplitude of complex number
    phi: torch.Tensor
        Phase of complex number
    
    Outputs
    -------
    out: torch.Tensor
        Complex number
    '''
    
    out = amp.to(dtype=torch.complex64) * \
        torch.exp(1j * phi.to(dtype=torch.complex64))
    
    return out

def is_complex(x: torch.Tensor) -> bool:
    '''
    Check if tensor is complex dtype.

    Inputs
    ------
    x: torch.Tensor
        Tensor to check

    Outputs
    -------
    out: bool
    '''

    return x.is_complex()

#Patch reassembly
#---------------------
def reassemble_patchess_position_real(inputs: torch.Tensor, offsets_xy: torch.Tensor,
                                      agg: bool = True,
                                      padded_size: Optional[int] = None,
                                      **kwargs: Any) -> torch.Tensor:
    '''
    Given image patches (shaped such that the channel dimension indexes
    patches within a single solution region), reassemble into an image
    for the entire solution region. Overlaps between patches are
    averaged.

    This function is passed as an argument (it is wrapped). This is because it is applied as a Lambda function
    to every single image patch in the dataloader stack.

    Inputs
    ------
    inputs: (n_images, n_patches, N, N)
    offsets_xy: offset patches in x, y
    agg: aggregation boolean
    padded_size: Amount of padding

    Output
    ------
    out: torch.Tensor
        Single tensor of shape (n_images, N, N) where N is summed object size. Everything has been merged here.
    '''

    if padded_size is None:
        padded_size = get_padded_size()
    #Flattening first two dimensions, adding singleton dimension to end
    offsets_flat = torch.flatten(offsets_xy[:, 0 , 0, :], start_dim = 0, end_dim = 1)[:,:,:,None]
    imgs_flat = torch.flatten(inputs, start_dim = 0, end_dim = 1)[:,:,:,None]
    #Pad patches
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




#Masking/norm functions
#---------------------
def mk_centermask(inputs: torch.Tensor, N: int, c: int, kind: str = 'center') -> torch.Tensor:
    '''
    Creates padded object mask to fulfill Nyquist criterion. This way, when we do FFT for
    diffraction image, we don't oversample frequency.
    (I.e. pad image on both sides with N/4 pixels, so total # of
    padded pixels is N/2)).
    The purpose of this mask is to find out how many images are overlapped
    at a specific pixel, so we can normalize the sum of pixel values
    by the number of images on that pixel (e.g. Pixel = 30, images = 3 -> normalized = 10)

    Inputs
    ------
    inputs: torch.Tensor
    N: Image dimension (x and y) in pixels
    c: Number of images per patch (usually 4)
    kind: 'center' or 'border'. Affects how mask is made

    Outputs
    -------
    output: Padded mask
    '''
    n_images = inputs.shape[0]
    count_mask = torch.ones((n_images, c, N//2, N//2),
                              dtype = inputs.dtype)
    count_mask = F.pad(count_mask, (N//4, N//4, N//4, N//4), "constant", 0)

    if kind == 'center':
        return count_mask
    elif kind == 'border':
        return 1 - count_mask
    else:
        raise ValueError

def mk_norm(inputs: torch.Tensor, fn_reassemble_real: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    '''
    Create normalization tensor based on count_mask from mk_centermask
    Adds .001 to deal with divide instablility from norm
    '''
    N = inputs.shape[-1]
    images_per_patch = inputs.shape[1]
    count_mask = mk_centermask(inputs, N, images_per_patch)
    assembled_masks = fn_reassemble_real(count_mask, average = False)
    norm = assembled_masks + 0.001

    return norm

#Pad functions
#---------------------

def pad_patches(input: torch.Tensor, padded_size: Optional[int] = None) -> torch.Tensor:
    '''
    Pad patches to be the same size as the original image
    '''
    if padded_size is None:
        padded_size = get_padded_size()
    pad_dim = (padded_size - input.shape[-1]) // 2

    return F.pad(input, (pad_dim, pad_dim, pad_dim, pad_dim), "constant", 0)

def get_padded_size():
    bigN = get_bigN()
    buffer = cfg['max_position_jitter']

    return bigN + buffer

def get_bigN():
    N = cfg['N']
    gridsize = cfg['gridsize']
    offset = cfg['offset']

    return N + (gridsize - 1) * offset

#Translation functions
#---------------------
def Translation(img, channels, offset, jitter_amt):
    '''
    Translation function with custom complex number support.
    Uses torch.nn.functional.grid_sample to perform subpixel translation.

    Grid_sample takes an input of (N, C, H_in, W_in) and grid of (N, H_out, W_out, 2) to
    output (N, C, H_out, W_out).

    We transform the input solution region to (C, 1, H, W), and the grid to (C, H_out, W_out, 2).
    The grid essentially contains all c possible translations of the input region.

    The output in our case will be (C, 1, H_out, W_out)

    Inputs
    ------
    img: torch.Tensor (C, H, W). Stack of images in a solution region
    offset: torch.Tensor (C, 1, 2). Offset of each image in the solution region
    '''
    c, h, w = img.shape
    #Unsqueeze (C, H, W) -> (C, 1, H, W)
    torch.unsqueeze(img,1)
    #Create 2d grid to sample bilinear interpolation from
    x, y = torch.arange(h)/(h-1), torch.arange(w)/(w-1)
    #Add offset to x, y using broadcasting (H) -> (C, H)
    x_shifted, y_shifted = x + offset[:, 0, 0], y + offset[:, 0, 1]

    #Create grid using manual stacking method C x H x W x 2)
    grid = torch.stack([x_shifted.unsqueeze(-1).expand(c, -1, y_shifted.shape[1]),
                    y_shifted.unsqueeze(1).expand(c, x_shifted.shape[1], -1)],
                    dim = -1)

    #Apply F.grid_sample to translate real and imaginary parts separately
    translated_real = F.grid_sample(img.unsqueeze(1).real, grid, mode = 'bilinear')
    translated_imag = F.grid_sample(img.unsqueeze(1).imag, grid, mode = 'bilinear')

    #Combine real and imag
    translated = torch.view_as_complex(torch.stack((translated_real, translated_imag),
                                                   dim = -1))
    
    return translated

    

#Flattening functions
#---------------------
def flatten_offsets(input: torch.Tensor) -> torch.Tensor:
    return _channel_to_flat(input)[:, 0, 0, :]

def _flat_to_channel(img: torch.Tensor, N: int = None) -> torch.Tensor:
    '''
    Reshapes tensor from flat format to channel format. Useful to batch apply operations such as
    translation on the flat tensor, then reshape back to channel format.

    Inputs
    ------
    img: torch.Tensor (N, H, W)

    Outputs
    -------
    out: torch.Tensor (M, C, H, W)
    '''
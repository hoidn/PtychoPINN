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
    n_channels = inputs.shape[1]
    offsets_flat = torch.flatten(offsets_xy[:, 0 , 0, :], start_dim = 0, end_dim = 1)[:,:,:,None]
    imgs_flat = torch.flatten(inputs, start_dim = 0, end_dim = 1)[:,:,:,None]
    #Pad patches
    imgs_flat_bigN = pad_patches(imgs_flat, padded_size)
    imgs_flat_bigN_translated = Translation()([imgs_flat_bigN, -offsets_flat, 0.])
    if agg:
        imgs_channel = torch.reshape(imgs_flat_bigN_translated,
                                     (-1, n_channels, padded_size, padded_size))
        #Count nonzeros for normalization
        n_nonzero = torch.count_nonzero(imgs_channel, dim = 1)
        imgs_merged = torch.sum(imgs_channel, axis = 1)
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
    img: torch.Tensor (N, H, W). Stack of images in a solution region. dtype complex, cfloat 
    offset: torch.Tensor (N, 1, 2). Offset of each image in the solution region
    '''
    n, h, w = img.shape
    #Create 2d grid to sample bilinear interpolation from
    x, y = torch.arange(h), torch.arange(w)
    #Add offset to x, y using broadcasting (H) -> (C, H)
    #NOTE TO ALBERT: ADD JITTER HERE
    x_shifted, y_shifted = (x + offset[:, 0, 0])/(h-1), (y + offset[:, 0, 1])/(w-1)
    #Create grid using manual stacking method C x H x W x 2)
    #Multiply by 2 and subtract 1 to shift to [-1, 1] range
    grid = torch.stack([x_shifted.unsqueeze(-1).expand(n, -1, y_shifted.shape[1]),
                    y_shifted.unsqueeze(1).expand(n, x_shifted.shape[1], -1)],
                    dim = -1) * 2 - 1

    #Apply F.grid_sample to translate real and imaginary parts separately.
    #grid_sample does not have native complex tensor support
    #Need to unsqueeze img to have it work with grid_sample (check documentation). 
    #In our case, color channels are 1 (singleton) and so we just unsqueeze at 2nd dimension

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

def _flat_to_channel(img: torch.Tensor, channels: int = 4) -> torch.Tensor:
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

    _, H, W = img.shape

    img = torch.reshape(img, (-1, channels, H, W))

    return img
    
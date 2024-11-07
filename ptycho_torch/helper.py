#Type helpers
from typing import Tuple, Optional, Union, Callable, Any

#Additional helper functions ported from original tensorflow lib
import torch
from torch import nn
import torch.nn.functional as F
from ptycho_torch.model import CombineComplex
import numpy as np

#Configurations
from ptycho_torch.config_params import ModelConfig, TrainingConfig, DataConfig

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

    CC = CombineComplex()
       
    return CC(amp, phi)
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
def reassemble_patches_position_real(inputs: torch.Tensor, offsets_xy: torch.Tensor,
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
    inputs: diffraction images from model -> (n_images, n_patches, N, N)
    offsets_xy: offset patches in x, y -> (n_images, n_patches, 1, 2)
    agg: aggregation boolean
    padded_size: Amount of padding

    Output
    ------
    out: torch.Tensor
        Single tensor of shape (n_images, N, N) where N is summed object size. Everything has been merged here.
    '''

    assert inputs.dtype == torch.complex64, 'Input must be complex'

    if padded_size is None:
        padded_size = get_padded_size()

    #Create ones mask for normaliziation, which cuts off half the image b/c Nyquist limit
    norm = norm_mask(inputs)

    #Flattening first two dimensions, adding singleton dimension to end
    n_channels = inputs.shape[1]
    offsets_flat = torch.flatten(offsets_xy, start_dim = 0, end_dim = 1)#[:,:,:,None]
    imgs_flat = torch.flatten(inputs, start_dim = 0, end_dim = 1)#[:,:,:,None]
    norm_flat = torch.flatten(norm, start_dim = 0, end_dim = 1)#[:,:,:,None]

    #Pad patches and translate
    imgs_flat_bigN = pad_patches(imgs_flat, padded_size)
    imgs_flat_bigN_translated = Translation(imgs_flat_bigN, offsets_flat, 0.)

    #Same thing with norm
    norm_flat_bigN = pad_patches(norm_flat, padded_size)
    norm_flat_bigN_translated = Translation(norm_flat_bigN, offsets_flat, 0.)

    if agg:
        #First reshape batch * channel dim tensors to (batch, channel, size, size)
        imgs_channel = torch.reshape(imgs_flat_bigN_translated,
                                     (-1, n_channels, padded_size, padded_size))
        norm_channel = torch.reshape(norm_flat_bigN_translated,
                                     (-1, n_channels, padded_size, padded_size))
        #Then decide which pixels you want to keep. We're getting rid of all pixels outside the N/2 x N/2 window
        #due to nyquist sampling

        #Boolean mask from N//2 sized masking
        boolean_mask = torch.any(norm_channel, dim = 1).to(inputs.device) #(b, N, N)
        #Count number of nonzeros from the ones mask, which has been modified by boolean_mask
        non_zeros = torch.count_nonzero(norm_channel * boolean_mask[:,None,:,:], axis = 1).to(inputs.device)
        #Change value of non_zeros from 0 to 1 in regions without any data. We'll be dividing
        #by the number of non_zeros, so this is to avoid div by 0
        non_zeros[non_zeros == 0] = 1

        #Normalize merged image by number of summed channels
        imgs_merged = (torch.sum(imgs_channel, axis = 1) * boolean_mask) / non_zeros
        return imgs_merged
    else:
        print('no aggregation in patch reassembly')
        return _flat_to_channel(imgs_flat_bigN_translated, N = padded_size)
    
def norm_mask(inputs):
    '''
    Creates normalization mask for patch reassembly. Simply returns a mask of ones which has reduced dimensionality
    '''
    B, C, N, _ = inputs.shape

    #Mask with half-size
    ones = torch.ones(B, C, N//2, N//2).to(inputs.device)
    #Pad rest
    ones = nn.ZeroPad2d(N//4)(ones)

    return ones

    
def extract_channels_from_region(inputs: torch.Tensor,
                                 offsets_xy: torch.Tensor,
                                 jitter_amt: float = 0.0) -> torch.Tensor:
    
    '''
    Extracts averaged objects from the summed M x M solution region.

    Essentially we're translating the solution region in reverse; we reverse the offsets we used to translate
    the original image, so that the image patch of interest (within the solution region) is
    centered at the origin.

    We can then use this modified translated solution region and do a simple crop extraction that's N x N about the origin.

    Inputs
    ------
    inputs: torch.Tensor (batch_size, 1, M, M), M = N + some padding size
    offsets: torch.Tensor (batch_size, C, 1, 2)
    jitter_amt: float
        Assuming zero jitter

    Output
    ------
    output: torch.Tensor (batch_size, C, N, N)
        - Shifted images, cropped symmetrically
    
    '''
    offsets_flat = torch.flatten(offsets_xy, start_dim = 0, end_dim = 1)
    #Check offset and input dimensions
    #List of assertions
    if inputs.shape[0] is not None:
        assert int(inputs.shape[0]) == int(offsets_xy.shape[0])
    assert int(inputs.shape[1]) == 1
    assert int(offsets_xy.shape[3]) == 2

    
    #We need to repeat the solution patch C # of times, so we can perform unique translations
    #for all C image patches.
    #Steps: Repeat -> Flatten
    n_channels = offsets_xy.shape[1]
    stacked_inputs = inputs.expand(-1, n_channels, -1, -1)
    flat_stacked_inputs = torch.flatten(stacked_inputs, start_dim = 0, end_dim = 1)

    #Obtain translation of patches
    #Note, offsets must be inverted in sign compared to Translation in: reassemble_patches_position_real
    translated_patches = Translation(flat_stacked_inputs, -offsets_flat, 0.0)
    cropped_patches = trim_reconstruction(translated_patches)
    cropped_patches = torch.reshape(cropped_patches,
                                    (-1, n_channels, DataConfig().get('N'), DataConfig().get('N')))

    return cropped_patches

    
def trim_reconstruction(inputs: torch.Tensor, N: Optional[int] = None) -> torch.Tensor:
    '''
    Trim from shape (-1, 1, M, M) to (-1, 1, N, N) where M >= N
    Second dimension is 1 because of Translation function always returning 4D tensor
    M is the expanded solution region size
    N is the exact size of the measured diffraction input image (e.g. 64 pixels)

    Assume M = get_padded_size()

    Inputs
    ------
    inputs: torch.Tensor (batch_size, 1, M, M)
    N: int, size of last two dimensions
    '''

    if N is None:
        N = DataConfig().get('N')

    shape = inputs.shape

    #Ensure dimension matching
    if shape[2] is not None:
        assert int(shape[2]) == int(shape[-1])
    try:
        clipsize = (int(shape[2]) - N) // 2
    except TypeError:
        clipsize = (get_padded_size() - N) // 2
    
    #return clipped input
    return inputs[:,:,
                  clipsize:-clipsize,
                  clipsize:-clipsize]



#Masking/norm functions
#---------------------
#UNUSED
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
    count_mask = torch.ones((n_images, c, N, N),
                              dtype = inputs.dtype)
    #count_mask = F.pad(count_mask, (N//4, N//4, N//4, N//4), "constant", 0)

    if kind == 'center':
        return count_mask
    elif kind == 'border':
        return 1 - count_mask
    else:
        raise ValueError

#UNUSED
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

def pad_obj(input: torch.Tensor, h: int, w: int) -> torch.Tensor:
    '''
    Pad object function for Nyquist criterion. An N x M image needs to be padded by N // 4
    pixels on the top and bottom sides, and the M // 4 on the left and right sides
    '''
    return nn.ZeroPad2d((h // 4, h // 4, w // 4, w // 4))(input)

def get_padded_size():
    bigN = get_bigN()
    buffer = ModelConfig().get('max_position_jitter')

    return bigN + buffer

def get_bigN():
    N = DataConfig().get('N')
    gridsize = DataConfig().get('grid_size')
    offset = ModelConfig().get('offset')

    return N + (gridsize[0] - 1) * offset

#Translation functions
#---------------------
def Translation(img, offset, jitter_amt):
    '''
    Translation function with custom complex number support.
    Uses torch.nn.functional.grid_sample to perform subpixel translation.
    Meant to be performed on a flattened set of inputs.

    Grid_sample takes an input of (N, C, H_in, W_in) and grid of (N, H_out, W_out, 2) to
    output (N, C, H_out, W_out). In our case, C is 1 because we already flattened the channels from (N, C, H_in, W_in)
    to (N * C, H_in, W_in). Offset is (N*C, 1, 2)

    We transform the input solution region to (N, 1, H, W), and the grid to (N, H_out, W_out, 2).
    The grid essentially contains all c possible translations of the input region.

    The output in our case will be (N, 1, H_out, W_out)

    Inputs
    ------
    img: torch.Tensor (N, H, W). Stack of images in a solution region. dtype complex, cfloat 
    offset: torch.Tensor (N, 1, 2). Offset of each image in the solution region

    Outputs
    ------
    out: torch.Tensor (N, 1, H_out, W_out)
    '''
    n, h, w = img.shape
    #Create 2d grid to sample bilinear interpolation from
    x, y = torch.arange(h).to(img.device), torch.arange(w).to(img.device)
    #Add offset to x, y using broadcasting (H) -> (C, H)
    jitter_x = torch.normal(torch.zeros(offset[:,:,0].shape).to(img.device), #offset: [n, 1]
                          std=jitter_amt)
    jitter_y = torch.normal(torch.zeros(offset[:,:,1].shape).to(img.device), #offset: [n, 1]
                          std=jitter_amt)
    
    x_shifted, y_shifted = (x + offset[:, :, 0] + jitter_x)/(h-1), \
                           (y + offset[:, :, 1] + jitter_y)/(w-1)
       
    #Create meshgrid using manual stacking method C x H x W x 2)
    #Multiply by 2 and subtract 1 to shift to [-1, 1] range for use by grid_sample
    grid = torch.stack([x_shifted.unsqueeze(-1).expand(n, -1, y_shifted.shape[1]),
                    y_shifted.unsqueeze(1).expand(n, x_shifted.shape[1], -1)],
                    dim = -1) * 2 - 1
    
    #Need to transpose grid due to some weird F.grid_sample behavior to make it align with tensorflow
    grid = torch.transpose(grid, 1, 2)

    #Apply F.grid_sample to translate real and imaginary parts separately.
    #grid_sample does not have native complex tensor support
    #Need to unsqueeze img to have it work with grid_sample (check documentation). 
    #In our case, color channels are 1 (singleton) and so we just unsqueeze at 2nd dimension
    
    translated_real = F.grid_sample(img.unsqueeze(1).real, grid,
                                    mode = 'bilinear', align_corners = True)

    if img.dtype == torch.complex64:
        translated_imag = F.grid_sample(img.unsqueeze(1).imag, grid,
                                        mode = 'bilinear', align_corners = True)
    else:
        translated_imag = torch.zeros_like(translated_real).to(img.device)

    #Combine real and imag
    translated = torch.view_as_complex(torch.stack((translated_real, translated_imag),
                                                   dim = -1))
    
    return translated

#Flattening functions
#---------------------
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
    

#Fourier functions for forward pass

def pad_and_diffract(input: torch.Tensor, pad: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Pads channel images and performs Fourier transform, going from real space (object) to
    reciprocal space (diffraction pattern)

    Input
    --------
    input: torch.Tensor (N, C, H, W). Does not need to be a flattened tensor.
    pad: Boolean. Whether to pad the input before performing the Fourier transform.
    
    '''

    h, w = input.shape[-2], input.shape[-1]

    if pad:
        input = pad_obj(input, h, w)
    padded = input
    #assert input.shape[-1] == 1
    input = torch.fft.fft2(input.to(torch.complex64))
    input = torch.real(torch.conj(input) * input) / (h * w)
    input = torch.sqrt(torch.fft.fftshift(input, dim=(-2, -1)))

    return input, padded

def illuminate_and_diffract(input: torch.Tensor, probe: torch.Tensor, intensity_scale = None) -> torch.Tensor:
    '''
    Performs illumination and diffraction of a single object.
    Supports complex tensors

    Inputs
    --------
    input: torch.Tensor (N, H, W)
        - Channels are missing because we don't expect channels to show up here
        - We are only illuminating a set of position-based patches from the object
    probe: torch.Tensor (N, H, W)
    '''
    #Add batch dimension for single image
    input_amp = torch.abs(input)
    input_phase = torch.angle(input)

    if intensity_scale is None:
        intensity_scale = scale_nphotons(input_amp * torch.abs(probe))

    input_scaled = intensity_scale * input

    #Multiply by probe
    probe_product = input_scaled * probe

    #Get new intensity
    input_amp = torch.abs(probe_product)

    #Perform FFT
    output = torch.fft.fft2(probe_product.to(torch.complex64))
    output = torch.real(torch.conj(output) * output) / (input.shape[-2] * input.shape[-1])
    output = torch.sqrt(torch.fft.fftshift(output, dim=(-2, -1)))

    #Scale by intensity
    output, input_amp, =\
            output / intensity_scale, input_amp / intensity_scale
    
    input_scaled = combine_complex(input_amp, input_phase)
    
    return output, input_scaled, intensity_scale

#Photon scaling functions

def normalize_data(X: torch.Tensor) -> torch.Tensor:
    """
    Scale the photon counts in the diffraction image accounting for zero padding condition
    and the fact that diffraction pixel values are the square root of the intensity, which we're
    looking to scale to

    Returns scaled data and scaling factor

    Inputs
    --------
    input: torch.Tensor (N, H, W)

    Outputs
    --------
    Tensor: torch.Tensor (N, H, W)
    Scaling_factor: float

    """
    N = DataConfig().get('N')
    scaling_factor = torch.sqrt(
            ((N / 2) ** 2) / torch.mean(torch.sum(X**2, dim = (1, 2)))
            )

    return X * scaling_factor, scaling_factor


def scale_nphotons(input: torch.Tensor) -> float:
    """
    Calculate the object amplitude normalization factor that gives the desired
    *expected* number of observed photons, averaged over an entire dataset.

    Returns a single scalar.

    Inputs
    --------
    input: torch.Tensor (N, H, W)
    """
    #Find the mean photons PER image, first by summing all pixels per image, then averaging the sums from all images
    mean_photons = torch.mean(torch.sum(input**2, dim = (1, 2)))
    norm_factor = torch.sqrt(DataConfig().get('nphotons') / mean_photons)

    return norm_factor

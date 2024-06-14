import torch
import torch.nn.functional as F
from typing import Tuple
from .params import params, get_padded_size, cfg

def get_mask(input: torch.Tensor, support_threshold: float) -> torch.Tensor:
    mask = torch.where(input > support_threshold, torch.ones_like(input), torch.zeros_like(input))
    return mask

def combine_complex(amp: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    real = amp * torch.cos(phi)
    imag = amp * torch.sin(phi)
    complex_tensor = torch.view_as_complex(torch.stack((real, imag), dim=-1))
    return complex_tensor

def pad_obj(input: torch.Tensor, h: int, w: int) -> torch.Tensor:
    padding = (w // 4, w // 4, h // 4, h // 4)
    padded_input = F.pad(input, padding, mode='constant', value=0)
    return padded_input

def _fromgrid(img: torch.Tensor) -> torch.Tensor:
    """
    Reshape (-1, gridsize, gridsize, N, N) to (-1, N, N, 1)
    """
    print("Debug: Entering _fromgrid function")
    N = params()['N']
    return torch.reshape(img, (-1, N, N, 1))

def _togrid(img: torch.Tensor, gridsize: int = None, N: int = None) -> torch.Tensor:
    """
    Reshape (b * gridsize * gridsize, N, N, 1) to (b, gridsize, gridsize, N, N, 1)

    i.e., from flat format to grid format
    """
    if gridsize is None:
        gridsize = params()['gridsize']
    if N is None:
        N = params()['N']
    return torch.reshape(img, (-1, gridsize, gridsize, N, N, 1))

def togrid(*imgs: torch.Tensor) -> tuple:
    """
    Reshape (-1, N, N, 1) to (-1, gridsize, gridsize, N, N)
    """
    return tuple(_togrid(img) for img in imgs)

def _grid_to_channel(grid: torch.Tensor) -> torch.Tensor:
    """
    Reshape (-1, gridsize, gridsize, N, N) to (-1, N, N, gridsize * gridsize)
    """
    gridsize = params()['gridsize']
    img = torch.permute(grid, (0, 3, 4, 1, 2, 5))
    _, hh, ww, _, _, _ = img.shape
    img = torch.reshape(img, (-1, hh, ww, gridsize**2))
    return img

def grid_to_channel(*grids: torch.Tensor) -> tuple:
    return tuple(_grid_to_channel(g) for g in grids)

def _flat_to_channel(img: torch.Tensor, N: int = None) -> torch.Tensor:
    gridsize = params()['gridsize']
    if N is None:
        N = params()['N']
    img = torch.reshape(img, (-1, gridsize**2, N, N))
    img = torch.permute(img, (0, 2, 3, 1))
    return img

def _flat_to_channel_2(img: torch.Tensor) -> torch.Tensor:
    gridsize = params()['gridsize']
    _, N, M, _ = img.shape
    img = torch.reshape(img, (-1, gridsize**2, N, M))
    img = torch.permute(img, (0, 2, 3, 1))
    return img

def _channel_to_flat(img: torch.Tensor) -> torch.Tensor:
    """
    Reshape (b, N, N, c) to (b * c, N, N, 1)
    """
    _, h, w, c = img.shape
    img = torch.permute(img, (0, 3, 1, 2))
    img = torch.reshape(img, (-1, h, w, 1))
    return img

def _channel_to_patches(channel: torch.Tensor) -> torch.Tensor:
    """
    Reshape (-1, N, N, gridsize * gridsize) to (-1, gridsize, gridsize, N**2)
    """
    gridsize = params()['gridsize']
    N = params()['N']
    img = torch.permute(channel, (0, 3, 1, 2))
    img = torch.reshape(img, (-1, gridsize, gridsize, N**2))
    return img

def pad_patches(imgs: torch.Tensor, padded_size: int = None) -> torch.Tensor:
    if padded_size is None:
        padded_size = get_padded_size()
    N = cfg['N']
    padding = (padded_size - N) // 2
    return F.pad(imgs, (padding, padding, padding, padding))

def pad(imgs: torch.Tensor, size: int) -> torch.Tensor:
    return F.pad(imgs, (size, size, size, size))

def trim_reconstruction(x: torch.Tensor, N: int = None) -> torch.Tensor:
    """
    Trim from shape (_, M, M, _) to (_, N, N, _), where M >= N

    When dealing with an input with a static shape, assume M = get_padded_size()
    """
    if N is None:
        N = cfg['N']
    shape = x.shape
    if shape[1] is not None:
        assert torch.all(torch.eq(shape[1], shape[2]))
    try:
        clipsize = (shape[1] - N) // 2
    except TypeError:
        clipsize = (get_padded_size() - N) // 2
    return x[:, clipsize:-clipsize, clipsize:-clipsize, :]

def flatten_offsets(channels: torch.Tensor) -> torch.Tensor:
    return _channel_to_flat(channels)[:, 0, :, 0]

def pad_reconstruction(channels: torch.Tensor) -> torch.Tensor:
    padded_size = get_padded_size()
    imgs_flat = _channel_to_flat(channels)
    return pad_patches(imgs_flat, padded_size)

def pad_and_diffract(input: torch.Tensor, h: int, w: int, pad: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    zero-pad the real-space object and then calculate the far field
    diffraction amplitude.

    Uses symmetric FT - L2 norm is conserved
    """
    print('input shape', input.shape)
    if pad:
        input = pad_obj(input, h, w)
    padded = input
    assert input.shape[-1] == 1
    input = torch.fft.fft2(torch.tensor(input[..., 0], dtype=torch.cfloat))
    input = torch.real(torch.conj(input) * input) / (h * w)
    input = torch.unsqueeze(torch.sqrt(torch.fft.fftshift(input, dim=(-2, -1))), -1)
    return padded, input

"""NumPy-based post-processing utilities for stitching ptychographic reconstruction patches.

This module provides CPU-based tools for reassembling small NxN patches into complete 
reconstructed images after the main TensorFlow-based reconstruction pipeline has produced 
its output. It handles overlapping regions, border clipping, and format conversions for 
visualization and saving results.

**Key Distinction**: This module provides NumPy-based post-processing functions, distinct 
from the TensorFlow-based `reassemble_patches()` family in `tf_helper.py` which operates 
during model training/inference. Use this module for final result assembly and visualization.

**Public Interface**:
- `stitch_patches()`: Core stitching function with full parameter control
- `reassemble_patches()`: High-level convenience wrapper

**Config Dictionary Requirements**:
The config parameter must contain these keys:
- `N` (int): Size of individual patches (N x N)
- `gridsize` (int): Number of patches per dimension in the grid
- `offset` (int): Overlap between adjacent patches in pixels
- `nimgs_test` (int): Number of test images for batch size calculation
- `outer_offset_test` (int, optional): Alternative offset for test data

**Data Flow**:
Input patches → Border clipping → Grid reassembly → Output full image(s)
- Handles complex-valued patches with flexible part extraction ('amp', 'phase', 'complex')
- Supports batch processing for multiple reconstructions
- Manages coordinate transformations and normalization

**Usage Patterns**:
- Called by training scripts for progress visualization
- Used by inference pipelines for final result assembly  
- Integrated into workflow components for automated processing

**Dependencies**: NumPy only (no TensorFlow dependencies for CPU-based processing)
"""
import math

import numpy as np


def stitch_raster_patches(
    patches,
    *,
    outer_offset: int,
    normalization: float = 1.0,
) -> np.ndarray:
    """Crop and tile a complete square raster of complex object patches.

    Input rows must already be in canonical row-major raster order.  Callers
    that load grouped or shuffled data are responsible for restoring that
    order from authenticated scan identity before invoking this pure NumPy
    assembly helper.
    """

    array = np.asarray(patches)
    if array.ndim != 3 or array.shape[0] == 0 or array.shape[1] != array.shape[2]:
        raise ValueError("patches must have nonempty shape (M, N, N)")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("patches must contain only finite numeric values")
    side = math.isqrt(array.shape[0])
    if side * side != array.shape[0]:
        raise ValueError("raster patch count must be a perfect square")
    if (
        isinstance(outer_offset, (bool, np.bool_))
        or not isinstance(outer_offset, (int, np.integer))
        or int(outer_offset) <= 0
        or int(outer_offset) % 2
    ):
        raise ValueError("outer_offset must be a positive even integer")
    outer_offset = int(outer_offset)
    N = int(array.shape[1])
    if outer_offset > 2 * N:
        raise ValueError("outer_offset produces an invalid patch crop")
    normalization = float(normalization)
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("normalization must be positive and finite")

    border_size = (N - outer_offset / 2.0) / 2.0
    border_left = int(np.ceil(border_size))
    border_right = int(np.floor(border_size))
    end = N - border_right
    if border_left < 0 or end <= border_left:
        raise ValueError("outer_offset leaves an empty raster tile")
    scaled = array * normalization
    if not np.isfinite(scaled).all():
        raise ValueError("normalization produced nonfinite raster patches")
    cropped = scaled[:, border_left:end, border_left:end]
    tile_height, tile_width = cropped.shape[1:]
    tiled = (
        cropped.reshape(side, side, tile_height, tile_width)
        .transpose(0, 2, 1, 3)
        .reshape(side * tile_height, side * tile_width)
    )
    return np.ascontiguousarray(tiled)

def stitch_patches(patches, config, *, 
                  norm_Y_I: float = 1.0,
                  norm: bool = True,
                  part: str = 'amp') -> np.ndarray:
    """
    Stitch NxN patches into full images.
    
    Args:
        patches: numpy array or tensorflow tensor of image patches to stitch
        config: Configuration dictionary containing patch parameters
        norm_Y_I: Normalization factor (default: 1.0)
        norm: Whether to apply normalization (default: True)
        part: Which part to extract - 'amp', 'phase', or 'complex' (default: 'amp')
        
    Returns:
        np.ndarray: Stitched image(s) with shape (batch, height, width, 1)
    """
    # Get N from config at the start
    N = config['N']
    def get_clip_sizes(outer_offset):
        """Calculate border sizes for clipping overlapping regions."""
        N = config['N']
        gridsize = config['gridsize']
        offset = config['offset']
        bordersize = (N - outer_offset / 2) / 2
        borderleft = int(np.ceil(bordersize))
        borderright = int(np.floor(bordersize))
        clipsize = (bordersize + ((gridsize - 1) * offset) // 2)
        clipleft = int(np.ceil(clipsize))
        clipright = int(np.floor(clipsize))
        return borderleft, borderright, clipleft, clipright
    
    # Convert tensorflow tensor to numpy if needed
    if hasattr(patches, 'numpy'):
        patches = patches.numpy()
    
    # For gridsize=1, offset might be None since there's no overlap
    outer_offset = config.get('outer_offset_test', config.get('offset', 0))
    if outer_offset is None:
        outer_offset = 0
    
    # Calculate number of segments using numpy's size
    nsegments = int(np.sqrt((patches.size / config['nimgs_test']) / (config['N']**2)))
    
    # Select extraction function
    if part == 'amp':
        getpart = np.absolute
    elif part == 'phase':
        getpart = np.angle
    elif part == 'complex':
        getpart = lambda x: x
    else:
        raise ValueError("part must be 'amp', 'phase', or 'complex'")
    
    # Extract and normalize if requested
    if norm:
        img_recon = np.reshape((norm_Y_I * getpart(patches)), 
                              (-1, nsegments, nsegments, N, N, 1))
    else:
        img_recon = np.reshape(getpart(patches), 
                              (-1, nsegments, nsegments, N, N, 1))
    
    # Clip borders
    borderleft, borderright, clipleft, clipright = get_clip_sizes(outer_offset)
    img_recon = img_recon[:, :, :, borderleft:-borderright, borderleft:-borderright, :]
    
    # Rearrange and reshape to final form
    tmp = img_recon.transpose(0, 1, 3, 2, 4, 5)
    stitched = tmp.reshape(-1, np.prod(tmp.shape[1:3]), np.prod(tmp.shape[1:3]), 1)
    
    return stitched

def reassemble_patches(patches, config, *, norm_Y_I=1., part='amp', norm=False):
    """
    High-level convenience function for stitching patches using config parameters.
    
    Args:
        patches: Patches to reassemble
        config: Configuration dictionary containing patch parameters
        norm_Y_I: Normalization factor (default: 1.0)
        part: Which part to extract (default: 'amp')
        norm: Whether to normalize (default: False)
    """
    return stitch_patches(
        patches,
        config,
        norm_Y_I=norm_Y_I,
        norm=norm,
        part=part
    )

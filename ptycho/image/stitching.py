import numpy as np

def stitch_patches(patches, *, 
                  N: int,
                  gridsize: int,
                  offset: int,
                  nimgs: int,
                  outer_offset: int = None,
                  norm_Y_I: float = 1.0,
                  norm: bool = True,
                  part: str = 'amp') -> np.ndarray:
    """
    Stitch NxN patches into full images.
    
    Args:
        patches: numpy array or tensorflow tensor of image patches to stitch
        N: Size of each square patch
        gridsize: Grid size for patch arrangement  
        offset: Spacing between patches
        nimgs: Number of images
        outer_offset: Offset between patches
        norm_Y_I: Normalization factor (default: 1.0)
        norm: Whether to apply normalization (default: True)
        part: Which part to extract - 'amp', 'phase', or 'complex' (default: 'amp')
        
    Returns:
        np.ndarray: Stitched image(s) with shape (batch, height, width, 1)
    """
    def get_clip_sizes(outer_offset):
        """Calculate border sizes for clipping overlapping regions."""
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
    
    outer_offset = outer_offset if outer_offset is not None else offset
    
    # Calculate number of segments using numpy's size
    nsegments = int(np.sqrt((patches.size / nimgs) / (N**2)))
    
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
        config: Configuration object containing patch parameters
        norm_Y_I: Normalization factor (default: 1.0)
        part: Which part to extract (default: 'amp')
        norm: Whether to normalize (default: False)
    """
    return stitch_patches(
        patches,
        N=config.N,
        gridsize=config.gridsize,
        offset=config.offset,
        nimgs=config.nimgs_test,  # Support legacy config naming
        outer_offset=getattr(config, 'outer_offset_test', None),  # Support legacy config naming
        norm_Y_I=norm_Y_I,
        norm=norm,
        part=part
    )
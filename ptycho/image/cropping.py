"""
Image cropping utilities for coordinate-based alignment.

This module provides utilities to crop images based on scan coordinates,
ensuring consistent alignment between reconstructed and ground truth objects.

Note: The current implementation uses a simplified margin calculation (half the
probe radius) which works well for typical ptychographic reconstructions but
may need adjustment for different reconstruction algorithms or scan patterns.
For precise alignment with specific reconstruction methods, consider using
the reconstruction bounds directly rather than calculating from scan coordinates.
"""

import numpy as np
from typing import Tuple


def get_scan_area_bbox(scan_coords_yx: np.ndarray, 
                      probe_radius: float, 
                      object_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box of the scanned area based on scan coordinates.
    
    This function determines the rectangular region that encompasses all scan
    positions plus a reduced margin (half the probe radius). The margin is 
    reduced because scan coordinates represent probe centers and reconstruction
    algorithms already account for probe overlap.
    
    Args:
        scan_coords_yx: Array of scan coordinates with shape (n_positions, 2)
                       where each row is [y, x] coordinates
        probe_radius: Radius of the probe in pixels (half the probe width)
        object_shape: Shape of the full object as (height, width)
        
    Returns:
        Tuple of (start_row, end_row, start_col, end_col) defining the bounding box
        
    Example:
        >>> coords = np.array([[10, 20], [30, 40], [50, 60]])
        >>> bbox = get_scan_area_bbox(coords, probe_radius=5, object_shape=(100, 100))
        >>> start_row, end_row, start_col, end_col = bbox
    """
    if len(scan_coords_yx) == 0:
        raise ValueError("scan_coords_yx cannot be empty")
    
    if scan_coords_yx.shape[1] != 2:
        raise ValueError("scan_coords_yx must have shape (n_positions, 2)")
    
    # Get the min/max coordinates
    min_y, min_x = scan_coords_yx.min(axis=0)
    max_y, max_x = scan_coords_yx.max(axis=0)
    
    # Add a smaller margin - the probe radius is too large since scan coordinates 
    # represent probe centers and stitching algorithms already handle overlap
    # Use a smaller margin to match the actual reconstruction bounds
    margin = int(np.ceil(probe_radius * 0.5))  # Use half the probe radius
    
    start_row = max(0, int(min_y) - margin)
    end_row = min(object_shape[0], int(max_y) + margin + 1)
    start_col = max(0, int(min_x) - margin)
    end_col = min(object_shape[1], int(max_x) + margin + 1)
    
    return start_row, end_row, start_col, end_col


def crop_to_scan_area(image: np.ndarray, 
                     scan_coords_yx: np.ndarray, 
                     probe_radius: float) -> np.ndarray:
    """
    Crop an image to the area covered by scan coordinates.
    
    This high-level function crops an image to contain only the region that
    was illuminated during scanning, based on the scan coordinates and probe size.
    
    Args:
        image: Input image to crop (2D or 3D array)
        scan_coords_yx: Array of scan coordinates with shape (n_positions, 2)
                       where each row is [y, x] coordinates  
        probe_radius: Radius of the probe in pixels
        
    Returns:
        Cropped image containing only the scanned region
        
    Example:
        >>> full_object = np.random.random((200, 200))
        >>> coords = np.array([[50, 60], [70, 80], [90, 100]])
        >>> cropped = crop_to_scan_area(full_object, coords, probe_radius=10)
    """
    if image.ndim < 2:
        raise ValueError("Image must be at least 2D")
    
    # Get the object shape from the first two dimensions
    object_shape = image.shape[:2]
    
    # Calculate bounding box
    start_row, end_row, start_col, end_col = get_scan_area_bbox(
        scan_coords_yx, probe_radius, object_shape
    )
    
    # Crop the image
    if image.ndim == 2:
        return image[start_row:end_row, start_col:end_col]
    elif image.ndim == 3:
        return image[start_row:end_row, start_col:end_col, :]
    else:
        # Handle higher dimensional arrays
        return image[start_row:end_row, start_col:end_col, ...]
"""
Reconstruction alignment and cropping utilities for evaluation pipelines.

This module provides specialized image cropping functions for aligning ptychographic
reconstructions with ground truth objects during evaluation. The core function
`align_for_evaluation` performs a critical role in model comparison workflows by
ensuring reconstructed objects are properly cropped and aligned with ground truth
data before metric calculation.

**Primary Use Case**: Model comparison and evaluation workflows where reconstructions
need to be aligned with ground truth for fair quantitative comparison (SSIM, FRC, etc.).

**Key Functions**:
    - `align_for_evaluation`: Two-stage alignment for reconstruction-to-ground-truth comparison
    - `crop_to_scan_area`: Generic scanning area cropping based on coordinates
    - `get_scan_area_bbox`: Bounding box calculation from scan coordinate arrays

**Integration Pattern**:
    ```python
    # Typical model comparison workflow integration
    from ptycho.tf_helper import reassemble_position
    from ptycho.image.cropping import align_for_evaluation
    
    # After reconstruction assembly
    pinn_recon = reassemble_position(pinn_patches, coords, stitch_size)
    baseline_recon = reassemble_position(baseline_patches, coords, stitch_size)
    
    # Align both reconstructions with ground truth
    pinn_aligned, gt_for_pinn = align_for_evaluation(
        pinn_recon, ground_truth, scan_coords_yx, stitch_size)
    baseline_aligned, gt_for_baseline = align_for_evaluation(
        baseline_recon, ground_truth, scan_coords_yx, stitch_size)
    
    # Now both are ready for fair metric comparison
    pinn_ssim = calculate_ssim(pinn_aligned, gt_for_pinn) 
    baseline_ssim = calculate_ssim(baseline_aligned, gt_for_baseline)
    ```

**Data Formats**:
    - Input images: 2D complex numpy arrays (amplitude + phase)
    - Coordinates: `(n_positions, 2)` arrays with `[y, x]` format
    - Output: Identically-shaped complex arrays ready for metric calculation
"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


# --- HELPER FUNCTION ---
def _center_crop(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crops a 2D array to a target size."""
    h, w = img.shape
    if h == target_h and w == target_w:
        return img
    
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    
    logger.info(f"Center-cropping from ({h}, {w}) to ({target_h}, {target_w})")
    return img[start_h : start_h + target_h, start_w : start_w + target_w]


# --- NEW AUTHORITATIVE FUNCTION ---
def align_for_evaluation(
    reconstruction_image: np.ndarray,
    ground_truth_image: np.ndarray,
    scan_coords_yx: np.ndarray,
    stitch_patch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns a reconstructed image with a ground truth object for evaluation.

    This function performs a two-stage alignment:
    1. It crops the ground truth to the bounding box defined by the scan
       coordinates and the patch size used for stitching (`stitch_patch_size`).
    2. It performs a final center-crop on both images to ensure they have
       identical dimensions, correcting for any minor off-by-one errors.

    Args:
        reconstruction_image: The reconstructed complex object, typically from
                              `tf_helper.reassemble_position`.
        ground_truth_image: The full, uncropped ground truth complex object.
        scan_coords_yx: Array of scan coordinates with shape (n_positions, 2),
                        where each row is [y, x] coordinates.
        stitch_patch_size: The size `M` of the central patch region used during
                           the `reassemble_position` stitching process.

    Returns:
        A tuple of (aligned_reconstruction, aligned_ground_truth), where both
        are 2D complex numpy arrays with identical shapes, ready for metric
        calculation.
    """
    logger.info("--- Aligning reconstruction with ground truth for evaluation ---")
    
    # 1. Squeeze inputs to 2D complex arrays
    recon_2d = np.squeeze(reconstruction_image)
    gt_2d = np.squeeze(ground_truth_image)
    
    if scan_coords_yx.shape[1] != 2:
        raise ValueError("scan_coords_yx must have shape (n_positions, 2)")

    # 2. Calculate the bounding box of the reconstruction on the ground truth canvas
    effective_radius = stitch_patch_size // 2
    min_y, min_x = scan_coords_yx.min(axis=0)
    max_y, max_x = scan_coords_yx.max(axis=0)

    start_row = max(0, int(min_y) - effective_radius)
    end_row = min(gt_2d.shape[0], int(max_y) + effective_radius)
    start_col = max(0, int(min_x) - effective_radius)
    end_col = min(gt_2d.shape[1], int(max_x) + effective_radius)
    
    logger.info(f"Calculated ground truth crop region: rows [{start_row}:{end_row}], cols [{start_col}:{end_col}]")

    # 3. Crop the ground truth to the calculated region
    gt_cropped = gt_2d[start_row:end_row, start_col:end_col]
    logger.info(f"Initial shapes: Recon={recon_2d.shape}, Cropped GT={gt_cropped.shape}")

    # 4. Perform final alignment by center-cropping to the smallest common shape
    target_h = min(recon_2d.shape[0], gt_cropped.shape[0])
    target_w = min(recon_2d.shape[1], gt_cropped.shape[1])

    aligned_recon = _center_crop(recon_2d, target_h, target_w)
    aligned_gt = _center_crop(gt_cropped, target_h, target_w)
    
    logger.info(f"Final aligned shape: ({target_h}, {target_w})")
    logger.info("--- Alignment complete ---")
    
    return aligned_recon, aligned_gt


def align_for_evaluation_with_registration(
    reconstruction_image: np.ndarray,
    ground_truth_image: np.ndarray,
    scan_coords_yx: np.ndarray,
    stitch_patch_size: int,
    *,
    upsample_factor: int = 50,
    border_crop: int = 2,
    skip_registration: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float] | None]:
    """
    Align reconstruction to ground truth using scan coordinates + optional registration.

    This combines the coordinate-based cropping from align_for_evaluation with the
    fine-scale registration used in compare_models.py (phase cross-correlation
    followed by border cropping).
    """
    aligned_recon, aligned_gt = align_for_evaluation(
        reconstruction_image=reconstruction_image,
        ground_truth_image=ground_truth_image,
        scan_coords_yx=scan_coords_yx,
        stitch_patch_size=stitch_patch_size,
    )
    if skip_registration:
        return aligned_recon, aligned_gt, None

    from ptycho.image.registration import apply_shift_and_crop, find_translation_offset

    offset = find_translation_offset(
        aligned_recon,
        aligned_gt,
        upsample_factor=upsample_factor,
    )
    aligned_recon, aligned_gt = apply_shift_and_crop(
        aligned_recon,
        aligned_gt,
        offset,
        border_crop=border_crop,
    )
    return aligned_recon, aligned_gt, offset


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
    
    This is a generic cropping function that crops an image to contain only the 
    region that was illuminated during scanning, based on the scan coordinates 
    and probe size.
    
    Note: For precise evaluation alignment of reconstructions with ground truth,
    use `align_for_evaluation()` instead, which handles reconstruction-specific 
    alignment requirements.
    
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

"""Inference and stitching pipeline for grid-based ptychography study.

Handles model inference and patch stitching for both baseline and PtychoPINN
models with proper handling of gridsize=1 data.

References:
    - ptycho/image/stitching.py - Patch stitching utilities
    - ptycho/train_pinn.py::eval() - PINN inference
"""
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ptycho import params as p


@dataclass
class InferenceResult:
    """Container for inference results."""
    pred_amp: np.ndarray       # Predicted amplitude patches
    pred_phase: np.ndarray     # Predicted phase patches
    stitched_amp: Optional[np.ndarray] = None   # Stitched amplitude image
    stitched_phase: Optional[np.ndarray] = None # Stitched phase image
    stitched_complex: Optional[np.ndarray] = None  # Stitched complex object
    config: Dict[str, Any] = None


def run_baseline_inference(
    model: tf.keras.Model,
    X_test: np.ndarray,
    intensity_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference with baseline model.

    Args:
        model: Trained baseline model
        X_test: Test diffraction patterns (batch, N, N, 1)
        intensity_scale: Intensity scaling factor

    Returns:
        (pred_amp, pred_phase) arrays
    """
    # Ensure single channel
    if X_test.ndim == 3:
        X_test = X_test[..., np.newaxis]

    # Baseline model outputs [amplitude, phase]
    pred_amp, pred_phase = model.predict(X_test, verbose=1)

    return pred_amp, pred_phase


def run_pinn_inference(
    model: tf.keras.Model,
    X_test: np.ndarray,
    intensity_scale: float,
    coords_test: Optional[np.ndarray] = None,
    gridsize: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference with PtychoPINN model.

    Args:
        model: Trained PtychoPINN model
        X_test: Test diffraction patterns (batch, N, N, gridsize^2)
        intensity_scale: Intensity scaling factor
        coords_test: Coordinate information (optional for gridsize=1)
        gridsize: Grid size used in training

    Returns:
        (pred_amp, pred_phase) arrays
    """
    # Ensure correct channel count
    if X_test.ndim == 3:
        X_test = X_test[..., np.newaxis]

    # Create zero coords if not provided (for gridsize=1)
    if coords_test is None:
        coords_test = np.zeros((X_test.shape[0], 1, 2, gridsize**2), dtype=np.float32)

    # Scale input
    X_scaled = X_test * intensity_scale

    # PINN model outputs [reconstructed_obj, pred_amp, reconstructed_obj_cdi]
    outputs = model.predict([X_scaled, coords_test], verbose=1)

    if len(outputs) == 3:
        reconstructed_obj, pred_amp, _ = outputs
    else:
        reconstructed_obj = outputs[0]
        pred_amp = np.abs(reconstructed_obj)

    # Extract amplitude and phase from complex reconstruction
    pred_amp_out = np.abs(reconstructed_obj)
    pred_phase_out = np.angle(reconstructed_obj)

    return pred_amp_out, pred_phase_out


def stitch_patches_simple(
    patches: np.ndarray,
    N: int,
    outer_offset: int,
    size: int,
    part: str = 'amp'
) -> np.ndarray:
    """
    Simple patch stitching for gridsize=1.

    For gridsize=1, patches are laid out on a regular grid.
    This function reassembles them by taking the central region
    of each patch to avoid border artifacts.

    Args:
        patches: Patch array (batch, N, N, 1)
        N: Patch size
        outer_offset: Grid step size
        size: Original object size
        part: 'amp', 'phase', or 'complex'

    Returns:
        Stitched image (height, width, 1)
    """
    if patches.ndim == 3:
        patches = patches[..., np.newaxis]

    # Calculate grid dimensions
    n_patches_per_dim = int(np.sqrt(patches.shape[0]))
    if n_patches_per_dim ** 2 != patches.shape[0]:
        # Not a perfect square - try to infer from geometry
        positions_per_dim = (size - N) // outer_offset + 1
        n_patches_per_dim = positions_per_dim

    # Calculate border to clip
    # For outer_offset=12 and N=128, border = (128 - 12/2) / 2 = 61
    border = int((N - outer_offset / 2) / 2)
    central_size = N - 2 * border  # Size of central region to keep

    if central_size <= 0:
        # Fallback: use a smaller border
        border = N // 4
        central_size = N - 2 * border
        print(f"Warning: Adjusted border to {border}, central_size={central_size}")

    # Extract the appropriate part
    if part == 'amp':
        data = np.abs(patches)
    elif part == 'phase':
        data = np.angle(patches)
    elif part == 'complex':
        data = patches
    else:
        raise ValueError(f"Unknown part: {part}")

    # Output size
    out_height = n_patches_per_dim * central_size
    out_width = n_patches_per_dim * central_size

    # Allocate output
    if part == 'complex':
        stitched = np.zeros((out_height, out_width, 1), dtype=np.complex64)
    else:
        stitched = np.zeros((out_height, out_width, 1), dtype=np.float32)

    # Stitch patches
    patch_idx = 0
    for i in range(n_patches_per_dim):
        for j in range(n_patches_per_dim):
            if patch_idx >= patches.shape[0]:
                break

            # Extract central region
            central = data[patch_idx, border:border+central_size, border:border+central_size, :]

            # Place in output
            y_start = i * central_size
            x_start = j * central_size
            stitched[y_start:y_start+central_size, x_start:x_start+central_size, :] = central

            patch_idx += 1

    return stitched


def run_inference_and_stitch(
    model: tf.keras.Model,
    X_test: np.ndarray,
    config: Dict[str, Any],
    model_type: str = 'baseline',
    intensity_scale: float = 1.0,
    coords_test: Optional[np.ndarray] = None,
) -> InferenceResult:
    """
    Run inference and stitch patches into full images.

    Args:
        model: Trained model
        X_test: Test diffraction patterns
        config: Configuration dictionary with N, outer_offset, size
        model_type: 'baseline' or 'pinn'
        intensity_scale: Intensity scaling factor
        coords_test: Coordinate information (for PINN)

    Returns:
        InferenceResult with predictions and stitched images
    """
    N = config['N']
    outer_offset = config.get('outer_offset_test', config.get('outer_offset', 12))
    size = config.get('size', 500)
    gridsize = config.get('gridsize', 1)

    print(f"Running {model_type} inference...")
    print(f"  Input shape: {X_test.shape}")
    print(f"  N={N}, outer_offset={outer_offset}, size={size}")

    # Run inference
    if model_type == 'baseline':
        pred_amp, pred_phase = run_baseline_inference(model, X_test, intensity_scale)
    else:  # pinn
        pred_amp, pred_phase = run_pinn_inference(
            model, X_test, intensity_scale, coords_test, gridsize
        )

    print(f"  Prediction shapes: amp={pred_amp.shape}, phase={pred_phase.shape}")

    # Stitch patches
    print("Stitching patches...")

    # For single-object test, we need to identify patches from same object
    # With n_test_objects=2, each object produces (size-N)//outer_offset+1)^2 patches
    patches_per_object = ((size - N) // outer_offset + 1) ** 2

    # Stitch first object only for visualization
    pred_amp_obj1 = pred_amp[:patches_per_object]
    pred_phase_obj1 = pred_phase[:patches_per_object]

    stitched_amp = stitch_patches_simple(
        pred_amp_obj1, N, outer_offset, size, part='amp'
    )
    stitched_phase = stitch_patches_simple(
        pred_phase_obj1, N, outer_offset, size, part='phase'
    )

    # Create complex stitched image
    stitched_complex = stitched_amp * np.exp(1j * stitched_phase)

    print(f"  Stitched shape: {stitched_amp.shape}")

    return InferenceResult(
        pred_amp=pred_amp,
        pred_phase=pred_phase,
        stitched_amp=stitched_amp,
        stitched_phase=stitched_phase,
        stitched_complex=stitched_complex,
        config=config,
    )


def stitch_ground_truth(
    YY_full: np.ndarray,
    config: Dict[str, Any],
    object_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and prepare ground truth for comparison.

    Args:
        YY_full: Full ground truth objects (n_objects, size, size)
        config: Configuration dictionary
        object_idx: Which object to extract

    Returns:
        (gt_amp, gt_phase) arrays
    """
    if YY_full.ndim == 2:
        gt_obj = YY_full
    else:
        gt_obj = YY_full[object_idx]

    gt_amp = np.abs(gt_obj)[..., np.newaxis]
    gt_phase = np.angle(gt_obj)[..., np.newaxis]

    return gt_amp, gt_phase


def align_for_comparison(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align prediction and ground truth for fair comparison.

    Handles size differences by center-cropping to the smaller size.

    Args:
        pred: Prediction array (H1, W1, 1)
        gt: Ground truth array (H2, W2, 1)

    Returns:
        (aligned_pred, aligned_gt) with same shape
    """
    pred_h, pred_w = pred.shape[:2]
    gt_h, gt_w = gt.shape[:2]

    # Use smaller dimensions
    min_h = min(pred_h, gt_h)
    min_w = min(pred_w, gt_w)

    # Center crop both
    pred_start_h = (pred_h - min_h) // 2
    pred_start_w = (pred_w - min_w) // 2
    gt_start_h = (gt_h - min_h) // 2
    gt_start_w = (gt_w - min_w) // 2

    aligned_pred = pred[pred_start_h:pred_start_h+min_h, pred_start_w:pred_start_w+min_w]
    aligned_gt = gt[gt_start_h:gt_start_h+min_h, gt_start_w:gt_start_w+min_w]

    return aligned_pred, aligned_gt


if __name__ == "__main__":
    # Test inference pipeline
    from probe_utils import get_probe_for_N
    from grid_data_generator import generate_train_test_data
    from train_models import train_baseline

    print("=== Inference Pipeline Test ===\n")

    # Generate small dataset
    probe_64 = get_probe_for_N(64)
    train_data, test_data = generate_train_test_data(
        probe=probe_64,
        n_train_objects=1,
        n_test_objects=1,
        outer_offset=12,
    )

    # Train baseline briefly
    print("\n--- Training baseline for inference test ---")
    baseline_result = train_baseline(
        X_train=train_data.X[:100],
        Y_I_train=train_data.Y_I[:100],
        Y_phi_train=train_data.Y_phi[:100],
        nepochs=1,
        batch_size=16,
    )

    # Run inference
    print("\n--- Running inference ---")
    config = {
        'N': 64,
        'outer_offset': 12,
        'outer_offset_test': 12,
        'size': 500,
        'gridsize': 1,
    }

    inference_result = run_inference_and_stitch(
        model=baseline_result.model,
        X_test=test_data.X[:100],
        config=config,
        model_type='baseline',
    )

    print(f"\nStitched amplitude shape: {inference_result.stitched_amp.shape}")
    print(f"Stitched phase shape: {inference_result.stitched_phase.shape}")
    print("\n--- Inference test complete ---")

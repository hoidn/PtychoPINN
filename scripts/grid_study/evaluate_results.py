"""Evaluation metrics and visualization for grid-based ptychography study.

Provides SSIM, MS-SSIM, and other metrics for comparing baseline vs PtychoPINN
reconstructions against ground truth.

References:
    - ptycho/evaluation.py - Core evaluation functions
"""
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    mae_amp: float
    mae_phase: float
    mse_amp: float
    mse_phase: float
    ssim_amp: float
    ssim_phase: float
    ms_ssim_amp: float
    ms_ssim_phase: float
    psnr_amp: float
    psnr_phase: float
    label: str = ""


def normalize_amplitude(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Normalize prediction amplitude to match target mean."""
    scale = np.mean(target) / (np.mean(pred) + 1e-10)
    return pred * scale


def align_phase(pred: np.ndarray, target: np.ndarray, method: str = 'plane') -> Tuple[np.ndarray, np.ndarray]:
    """
    Align phase images for fair comparison.

    Args:
        pred: Predicted phase
        target: Ground truth phase
        method: 'plane' (fit and remove plane) or 'mean' (subtract mean)

    Returns:
        (aligned_pred, aligned_target)
    """
    if method == 'mean':
        pred_aligned = pred - np.mean(pred)
        target_aligned = target - np.mean(target)
    elif method == 'plane':
        pred_aligned = _fit_and_remove_plane(pred)
        target_aligned = _fit_and_remove_plane(target)
    else:
        raise ValueError(f"Unknown method: {method}")

    return pred_aligned, target_aligned


def _fit_and_remove_plane(phase_img: np.ndarray) -> np.ndarray:
    """Fit and remove a plane from phase image."""
    # Squeeze to 2D if needed
    phase_2d = np.squeeze(phase_img)
    h, w = phase_2d.shape

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Flatten for linear system
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    phase_flat = phase_2d.flatten()

    # Set up linear system A * coeffs = phase for plane z = a*x + b*y + c
    A = np.column_stack([x_flat, y_flat, np.ones(len(x_flat))])

    # Solve for plane coefficients
    coeffs, _, _, _ = np.linalg.lstsq(A, phase_flat, rcond=None)

    # Compute fitted plane
    fitted_plane = coeffs[0] * x_coords + coeffs[1] * y_coords + coeffs[2]

    # Remove the fitted plane
    result = phase_2d - fitted_plane

    # Restore original shape
    return result.reshape(phase_img.shape)


def calculate_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return float(np.mean(np.abs(pred - target)))


def calculate_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return float(np.mean((pred - target) ** 2))


def calculate_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = calculate_mse(pred, target)
    if mse < 1e-10:
        return float('inf')
    max_val = np.max(target)
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def calculate_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Structural Similarity Index."""
    from skimage.metrics import structural_similarity

    # Ensure 2D
    pred_2d = np.squeeze(pred)
    target_2d = np.squeeze(target)

    data_range = float(np.max(target_2d) - np.min(target_2d))
    return float(structural_similarity(pred_2d, target_2d, data_range=data_range))


def calculate_ms_ssim(pred: np.ndarray, target: np.ndarray, levels: int = 5, sigma: float = 0.0) -> float:
    """
    Calculate Multi-Scale SSIM.

    Args:
        pred: Predicted image
        target: Ground truth image
        levels: Number of scale levels
        sigma: Gaussian smoothing sigma (0 = no smoothing)

    Returns:
        MS-SSIM value
    """
    from skimage.metrics import structural_similarity
    from scipy.ndimage import gaussian_filter

    # Ensure 2D
    pred_2d = np.squeeze(pred).astype(np.float64)
    target_2d = np.squeeze(target).astype(np.float64)

    # Apply Gaussian smoothing if requested
    if sigma > 0:
        pred_2d = gaussian_filter(pred_2d, sigma)
        target_2d = gaussian_filter(target_2d, sigma)

    # Standard MS-SSIM weights
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = min(levels, len(weights))

    data_range = max(pred_2d.max() - pred_2d.min(), target_2d.max() - target_2d.min())

    ms_ssim_val = 1.0

    for level in range(levels):
        # Calculate SSIM at current scale
        ssim_val = structural_similarity(pred_2d, target_2d, data_range=data_range)

        # Handle edge cases
        if np.isnan(ssim_val):
            return 0.0
        elif ssim_val < 0:
            ssim_val = 0.0001

        ms_ssim_val *= (ssim_val ** weights[level])

        if np.isnan(ms_ssim_val):
            return 0.0

        # Downsample for next level
        if level < levels - 1:
            pred_2d = pred_2d[::2, ::2]
            target_2d = target_2d[::2, ::2]

            if min(pred_2d.shape) < 7:
                break

    return float(ms_ssim_val) if not np.isnan(ms_ssim_val) else 0.0


def evaluate_reconstruction(
    pred_amp: np.ndarray,
    pred_phase: np.ndarray,
    gt_amp: np.ndarray,
    gt_phase: np.ndarray,
    label: str = "",
    phase_align_method: str = 'plane',
    ms_ssim_sigma: float = 1.0,
) -> EvaluationMetrics:
    """
    Evaluate reconstruction quality against ground truth.

    Args:
        pred_amp: Predicted amplitude
        pred_phase: Predicted phase
        gt_amp: Ground truth amplitude
        gt_phase: Ground truth phase
        label: Label for this evaluation (e.g., 'baseline', 'pinn')
        phase_align_method: 'plane' or 'mean' for phase alignment
        ms_ssim_sigma: Gaussian smoothing for MS-SSIM amplitude

    Returns:
        EvaluationMetrics dataclass
    """
    print(f"\nEvaluating {label}...")

    # Normalize amplitude
    pred_amp_norm = normalize_amplitude(np.squeeze(pred_amp), np.squeeze(gt_amp))
    gt_amp_sq = np.squeeze(gt_amp)

    # Align phase
    pred_phase_aligned, gt_phase_aligned = align_phase(
        np.squeeze(pred_phase), np.squeeze(gt_phase), method=phase_align_method
    )

    # Calculate amplitude metrics
    mae_amp = calculate_mae(pred_amp_norm, gt_amp_sq)
    mse_amp = calculate_mse(pred_amp_norm, gt_amp_sq)
    psnr_amp = calculate_psnr(pred_amp_norm, gt_amp_sq)
    ssim_amp = calculate_ssim(pred_amp_norm, gt_amp_sq)
    ms_ssim_amp = calculate_ms_ssim(pred_amp_norm, gt_amp_sq, sigma=ms_ssim_sigma)

    # Calculate phase metrics (scale to [0, 1] for SSIM)
    phase_pred_scaled = (pred_phase_aligned + np.pi) / (2 * np.pi)
    phase_gt_scaled = (gt_phase_aligned + np.pi) / (2 * np.pi)

    mae_phase = calculate_mae(pred_phase_aligned, gt_phase_aligned)
    mse_phase = calculate_mse(pred_phase_aligned, gt_phase_aligned)
    psnr_phase = calculate_psnr(phase_pred_scaled, phase_gt_scaled)
    ssim_phase = calculate_ssim(phase_pred_scaled, phase_gt_scaled)
    ms_ssim_phase = calculate_ms_ssim(phase_pred_scaled, phase_gt_scaled)

    metrics = EvaluationMetrics(
        mae_amp=mae_amp,
        mae_phase=mae_phase,
        mse_amp=mse_amp,
        mse_phase=mse_phase,
        ssim_amp=ssim_amp,
        ssim_phase=ssim_phase,
        ms_ssim_amp=ms_ssim_amp,
        ms_ssim_phase=ms_ssim_phase,
        psnr_amp=psnr_amp,
        psnr_phase=psnr_phase,
        label=label,
    )

    print(f"  Amplitude - MAE: {mae_amp:.6f}, SSIM: {ssim_amp:.4f}, MS-SSIM: {ms_ssim_amp:.4f}")
    print(f"  Phase     - MAE: {mae_phase:.6f}, SSIM: {ssim_phase:.4f}, MS-SSIM: {ms_ssim_phase:.4f}")

    return metrics


def compare_models(
    baseline_metrics: EvaluationMetrics,
    pinn_metrics: EvaluationMetrics,
) -> Dict[str, Any]:
    """
    Compare baseline vs PtychoPINN metrics.

    Returns:
        Dictionary with comparison statistics
    """
    comparison = {
        'amplitude': {
            'mae_improvement': (baseline_metrics.mae_amp - pinn_metrics.mae_amp) / baseline_metrics.mae_amp * 100,
            'ssim_improvement': (pinn_metrics.ssim_amp - baseline_metrics.ssim_amp) / baseline_metrics.ssim_amp * 100,
            'ms_ssim_improvement': (pinn_metrics.ms_ssim_amp - baseline_metrics.ms_ssim_amp) / baseline_metrics.ms_ssim_amp * 100,
        },
        'phase': {
            'mae_improvement': (baseline_metrics.mae_phase - pinn_metrics.mae_phase) / baseline_metrics.mae_phase * 100,
            'ssim_improvement': (pinn_metrics.ssim_phase - baseline_metrics.ssim_phase) / baseline_metrics.ssim_phase * 100,
            'ms_ssim_improvement': (pinn_metrics.ms_ssim_phase - baseline_metrics.ms_ssim_phase) / baseline_metrics.ms_ssim_phase * 100,
        },
        'baseline': baseline_metrics,
        'pinn': pinn_metrics,
    }

    print("\n" + "=" * 60)
    print("Model Comparison (Positive = PINN better)")
    print("=" * 60)
    print(f"Amplitude MAE improvement:    {comparison['amplitude']['mae_improvement']:+.1f}%")
    print(f"Amplitude SSIM improvement:   {comparison['amplitude']['ssim_improvement']:+.1f}%")
    print(f"Amplitude MS-SSIM improvement:{comparison['amplitude']['ms_ssim_improvement']:+.1f}%")
    print(f"Phase MAE improvement:        {comparison['phase']['mae_improvement']:+.1f}%")
    print(f"Phase SSIM improvement:       {comparison['phase']['ssim_improvement']:+.1f}%")
    print(f"Phase MS-SSIM improvement:    {comparison['phase']['ms_ssim_improvement']:+.1f}%")
    print("=" * 60)

    return comparison


def save_comparison_plots(
    pred_amp: np.ndarray,
    pred_phase: np.ndarray,
    gt_amp: np.ndarray,
    gt_phase: np.ndarray,
    output_path: Path,
    label: str = "",
) -> None:
    """
    Save comparison visualization plots.

    Args:
        pred_amp: Predicted amplitude
        pred_phase: Predicted phase
        gt_amp: Ground truth amplitude
        gt_phase: Ground truth phase
        output_path: Directory to save plots
        label: Label for filenames
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Squeeze arrays
    pred_amp = np.squeeze(pred_amp)
    pred_phase = np.squeeze(pred_phase)
    gt_amp = np.squeeze(gt_amp)
    gt_phase = np.squeeze(gt_phase)

    # Normalize amplitude for display
    pred_amp_norm = normalize_amplitude(pred_amp, gt_amp)

    # Align phase
    pred_phase_aligned, gt_phase_aligned = align_phase(pred_phase, gt_phase, method='plane')

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Amplitude row
    vmax_amp = max(gt_amp.max(), pred_amp_norm.max())
    axes[0, 0].imshow(gt_amp, cmap='gray', vmin=0, vmax=vmax_amp)
    axes[0, 0].set_title('Ground Truth Amplitude')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pred_amp_norm, cmap='gray', vmin=0, vmax=vmax_amp)
    axes[0, 1].set_title(f'{label} Predicted Amplitude')
    axes[0, 1].axis('off')

    amp_diff = np.abs(gt_amp - pred_amp_norm)
    im = axes[0, 2].imshow(amp_diff, cmap='hot')
    axes[0, 2].set_title('Amplitude Difference')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])

    # Phase row
    vmin_phase = min(gt_phase_aligned.min(), pred_phase_aligned.min())
    vmax_phase = max(gt_phase_aligned.max(), pred_phase_aligned.max())
    axes[1, 0].imshow(gt_phase_aligned, cmap='hsv', vmin=vmin_phase, vmax=vmax_phase)
    axes[1, 0].set_title('Ground Truth Phase (aligned)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pred_phase_aligned, cmap='hsv', vmin=vmin_phase, vmax=vmax_phase)
    axes[1, 1].set_title(f'{label} Predicted Phase (aligned)')
    axes[1, 1].axis('off')

    phase_diff = np.abs(gt_phase_aligned - pred_phase_aligned)
    im = axes[1, 2].imshow(phase_diff, cmap='hot')
    axes[1, 2].set_title('Phase Difference')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(output_path / f'{label}_comparison.png', dpi=150)
    plt.close()

    print(f"Saved comparison plot: {output_path / f'{label}_comparison.png'}")


def save_metrics_csv(
    metrics_list: list,
    output_path: Path,
    filename: str = "metrics.csv",
) -> None:
    """
    Save metrics to CSV file.

    Args:
        metrics_list: List of EvaluationMetrics
        output_path: Directory to save CSV
        filename: CSV filename
    """
    import csv

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'label', 'mae_amp', 'mae_phase', 'mse_amp', 'mse_phase',
        'ssim_amp', 'ssim_phase', 'ms_ssim_amp', 'ms_ssim_phase',
        'psnr_amp', 'psnr_phase'
    ]

    csv_path = output_path / filename
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow({
                'label': m.label,
                'mae_amp': m.mae_amp,
                'mae_phase': m.mae_phase,
                'mse_amp': m.mse_amp,
                'mse_phase': m.mse_phase,
                'ssim_amp': m.ssim_amp,
                'ssim_phase': m.ssim_phase,
                'ms_ssim_amp': m.ms_ssim_amp,
                'ms_ssim_phase': m.ms_ssim_phase,
                'psnr_amp': m.psnr_amp,
                'psnr_phase': m.psnr_phase,
            })

    print(f"Saved metrics CSV: {csv_path}")


if __name__ == "__main__":
    # Test evaluation with synthetic data
    print("=== Evaluation Test ===\n")

    # Create synthetic test data
    np.random.seed(42)
    gt_amp = np.random.rand(64, 64).astype(np.float32)
    gt_phase = (np.random.rand(64, 64) - 0.5) * np.pi  # [-pi/2, pi/2]

    # Create noisy predictions
    pred_amp_baseline = gt_amp + np.random.randn(64, 64) * 0.1
    pred_phase_baseline = gt_phase + np.random.randn(64, 64) * 0.2

    pred_amp_pinn = gt_amp + np.random.randn(64, 64) * 0.05  # Less noise
    pred_phase_pinn = gt_phase + np.random.randn(64, 64) * 0.1

    # Evaluate baseline
    baseline_metrics = evaluate_reconstruction(
        pred_amp_baseline, pred_phase_baseline,
        gt_amp, gt_phase,
        label='baseline'
    )

    # Evaluate PINN
    pinn_metrics = evaluate_reconstruction(
        pred_amp_pinn, pred_phase_pinn,
        gt_amp, gt_phase,
        label='pinn'
    )

    # Compare
    comparison = compare_models(baseline_metrics, pinn_metrics)

    print("\n--- Evaluation test complete ---")

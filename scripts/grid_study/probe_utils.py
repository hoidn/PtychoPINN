"""Probe extraction and upscaling utilities.

Handles loading experimental probes from NPZ files and upscaling with
energy normalization to preserve physical validity.

Critical: When upscaling, the probe's total energy must be normalized to
match the original. Bicubic interpolation preserves per-pixel amplitude
but increases total energy proportional to the area increase.
"""
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
from typing import Tuple, Optional

# Default probe source
DEFAULT_PROBE_NPZ = Path(__file__).parent.parent.parent / "ptycho" / "datasets" / "Run1084_recon3_postPC_shrunk_3.npz"


def extract_probe_from_npz(npz_path: Optional[Path] = None) -> np.ndarray:
    """
    Extract probe from NPZ file.

    Args:
        npz_path: Path to NPZ file containing 'probeGuess' key.
                  Defaults to Run1084_recon3_postPC_shrunk_3.npz

    Returns:
        Complex probe array (typically 64x64 complex128)
    """
    if npz_path is None:
        npz_path = DEFAULT_PROBE_NPZ

    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Probe NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    if 'probeGuess' not in data:
        available = list(data.keys())
        raise KeyError(f"'probeGuess' not found in {npz_path}. Available keys: {available}")

    probe = data['probeGuess']

    # Validate probe
    if probe.ndim != 2:
        raise ValueError(f"Expected 2D probe, got shape {probe.shape}")
    if probe.shape[0] != probe.shape[1]:
        raise ValueError(f"Expected square probe, got shape {probe.shape}")

    return probe.astype(np.complex128)


def compute_probe_energy(probe: np.ndarray) -> float:
    """Compute total energy (sum of |probe|^2)."""
    return float(np.sum(np.abs(probe) ** 2))


def upscale_probe(probe: np.ndarray, target_size: int, normalize_energy: bool = True) -> np.ndarray:
    """
    Upscale probe to target size using bicubic interpolation.

    Args:
        probe: Source probe array (complex)
        target_size: Target size (square)
        normalize_energy: If True, normalize to preserve total energy.
                         CRITICAL: Must be True for physical validity.

    Returns:
        Upscaled probe array
    """
    source_size = probe.shape[0]

    if target_size == source_size:
        return probe.copy()

    if target_size < source_size:
        raise ValueError(f"Downscaling not supported: {source_size} -> {target_size}")

    scale_factor = target_size / source_size

    # Upscale real and imaginary parts separately (bicubic interpolation)
    upscaled_real = zoom(probe.real, scale_factor, order=3)
    upscaled_imag = zoom(probe.imag, scale_factor, order=3)
    upscaled = upscaled_real + 1j * upscaled_imag

    # Ensure exact target size (zoom may have rounding errors)
    if upscaled.shape[0] != target_size:
        upscaled = upscaled[:target_size, :target_size]

    if normalize_energy:
        # CRITICAL: Preserve total energy
        # Upscaling increases pixel count, so total energy increases ~4x for 2x upscale
        original_energy = compute_probe_energy(probe)
        upscaled_energy = compute_probe_energy(upscaled)

        if upscaled_energy > 0:
            normalization_factor = np.sqrt(original_energy / upscaled_energy)
            upscaled = upscaled * normalization_factor

    return upscaled.astype(np.complex128)


def get_probe_for_N(N: int, npz_path: Optional[Path] = None) -> np.ndarray:
    """
    Get probe of specified size, upscaling if necessary.

    Args:
        N: Target probe size (64 or 128)
        npz_path: Path to source probe NPZ

    Returns:
        Probe array of shape (N, N)
    """
    probe = extract_probe_from_npz(npz_path)
    source_size = probe.shape[0]

    if N == source_size:
        print(f"Using native {source_size}x{source_size} probe")
        return probe
    elif N > source_size:
        print(f"Upscaling probe from {source_size}x{source_size} to {N}x{N} with energy normalization")
        return upscale_probe(probe, N, normalize_energy=True)
    else:
        raise ValueError(f"Downscaling not supported: source={source_size}, target={N}")


def validate_probe(probe: np.ndarray, expected_size: int) -> Tuple[bool, str]:
    """
    Validate probe for use in simulation.

    Returns:
        (is_valid, message)
    """
    issues = []

    if probe.shape != (expected_size, expected_size):
        issues.append(f"Shape mismatch: expected ({expected_size}, {expected_size}), got {probe.shape}")

    if not np.iscomplexobj(probe):
        issues.append("Probe must be complex-valued")

    if np.any(np.isnan(probe)):
        issues.append("Probe contains NaN values")

    if np.any(np.isinf(probe)):
        issues.append("Probe contains infinite values")

    # Check for reasonable amplitude range
    amp = np.abs(probe)
    if amp.max() < 1e-10:
        issues.append("Probe amplitude is effectively zero")

    # Check edge energy (should be small for well-contained probe)
    edge_energy = (np.sum(np.abs(probe[0, :]) ** 2) +
                   np.sum(np.abs(probe[-1, :]) ** 2) +
                   np.sum(np.abs(probe[:, 0]) ** 2) +
                   np.sum(np.abs(probe[:, -1]) ** 2))
    total_energy = compute_probe_energy(probe)
    edge_fraction = edge_energy / total_energy if total_energy > 0 else 0

    if edge_fraction > 0.05:
        issues.append(f"High edge energy ({edge_fraction:.1%}) - probe may not be well-contained")

    if issues:
        return False, "; ".join(issues)

    return True, f"Probe valid: {expected_size}x{expected_size}, energy={total_energy:.2f}"


if __name__ == "__main__":
    # Test probe extraction and upscaling
    print("=== Probe Utils Test ===\n")

    # Extract native probe
    probe_64 = extract_probe_from_npz()
    print(f"Native probe: shape={probe_64.shape}, dtype={probe_64.dtype}")
    print(f"  Amplitude range: [{np.abs(probe_64).min():.6f}, {np.abs(probe_64).max():.6f}]")
    print(f"  Total energy: {compute_probe_energy(probe_64):.2f}")

    valid, msg = validate_probe(probe_64, 64)
    print(f"  Validation: {msg}\n")

    # Upscale to 128x128
    probe_128 = upscale_probe(probe_64, 128, normalize_energy=True)
    print(f"Upscaled probe: shape={probe_128.shape}")
    print(f"  Amplitude range: [{np.abs(probe_128).min():.6f}, {np.abs(probe_128).max():.6f}]")
    print(f"  Total energy: {compute_probe_energy(probe_128):.2f} (should match 64x64)")

    valid, msg = validate_probe(probe_128, 128)
    print(f"  Validation: {msg}\n")

    # Test convenience function
    for N in [64, 128]:
        probe = get_probe_for_N(N)
        print(f"get_probe_for_N({N}): shape={probe.shape}, energy={compute_probe_energy(probe):.2f}")

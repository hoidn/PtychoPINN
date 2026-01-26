#!/usr/bin/env python3
"""
grid_resolution_study.py - Grid-Sampled Resolution Comparison Study

This script compares PtychoPINN vs Baseline reconstruction quality at different
diffraction pattern resolutions (N=64, N=128) using grid-sampled synthetic data.

Goals:
1. Generate grid-sampled synthetic data at specified resolutions
2. Train PtychoPINN and Baseline models on identical datasets
3. Evaluate reconstruction quality metrics at each resolution
4. Produce comparative visualizations and analysis

Architecture:
    - Uses grid-based sampling (gridsize=1 or 2)
    - Follows data contracts (DATA-001) and normalization (NORMALIZATION-001)
    - Modern configuration system with update_legacy_dict (CONFIG-001)

Usage:
    python scripts/studies/grid_resolution_study.py [options]

Example:
    python scripts/studies/grid_resolution_study.py \
        --output-dir grid_res_outputs \
        --nepochs 50 \
        --resolutions 64 128 \
        --gridsize 2 \
        --n-train 500 \
        --n-test 50

References:
    - Replaces deprecated ptycho_lines.ipynb notebook
    - See scripts/studies/README.md for study workflow guidance
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
from scipy.ndimage import zoom
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Integration test dataset constant
INTEGRATION_TEST_NPZ = project_root / 'ptycho/datasets/Run1084_recon3_postPC_shrunk_3.npz'


# =============================================================================
# Phase 1: Probe Extraction & Upscaling
# =============================================================================

def extract_and_scale_probe(source_npz: Path, target_N: int) -> np.ndarray:
    """Extract probe from NPZ and scale to target resolution.

    Uses cubic interpolation on real/imag parts separately to preserve
    phase information during upscaling.

    Contract: PROBE-SCALE-001
    - Input: source NPZ with 'probeGuess' key, complex array
    - Output: complex64 array of shape (target_N, target_N)
    - Scaling preserves relative phase structure

    Args:
        source_npz: Path to NPZ file containing 'probeGuess'
        target_N: Target resolution (e.g., 64 or 128)

    Returns:
        Complex probe array scaled to (target_N, target_N)

    Raises:
        KeyError: If 'probeGuess' not found in NPZ
        ValueError: If probe is not square
    """
    data = np.load(source_npz)
    if 'probeGuess' not in data:
        raise KeyError(f"'probeGuess' not found in {source_npz}. Available keys: {list(data.keys())}")

    probe = data['probeGuess']
    if probe.shape[0] != probe.shape[1]:
        raise ValueError(f"Probe must be square, got shape {probe.shape}")

    source_N = probe.shape[0]
    logger.info(f"Extracted probe from {source_npz}: shape={probe.shape}, dtype={probe.dtype}")

    if target_N == source_N:
        logger.info(f"No scaling needed, returning probe as complex64")
        return probe.astype(np.complex64)

    # Scale real and imaginary parts separately with cubic interpolation
    scale_factor = target_N / source_N
    logger.info(f"Scaling probe from {source_N}x{source_N} to {target_N}x{target_N} (factor={scale_factor:.2f})")

    real_scaled = zoom(probe.real, scale_factor, order=3)
    imag_scaled = zoom(probe.imag, scale_factor, order=3)
    probe_scaled = real_scaled + 1j * imag_scaled

    logger.info(f"Scaled probe: shape={probe_scaled.shape}, dtype will be complex64")
    return probe_scaled.astype(np.complex64)


def _test_extract_and_scale_probe() -> bool:
    """Inline test for extract_and_scale_probe function.

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("=" * 60)
    logger.info("Testing extract_and_scale_probe()")
    logger.info("=" * 60)

    all_passed = True

    # Test 1: Extract at native resolution (64x64)
    logger.info("\nTest 1: Extract at native resolution (64x64)")
    try:
        probe_64 = extract_and_scale_probe(INTEGRATION_TEST_NPZ, 64)
        assert probe_64.shape == (64, 64), f"Expected (64, 64), got {probe_64.shape}"
        assert probe_64.dtype == np.complex64, f"Expected complex64, got {probe_64.dtype}"
        assert np.iscomplexobj(probe_64), "Expected complex array"
        logger.info(f"  PASS: shape={probe_64.shape}, dtype={probe_64.dtype}")
        logger.info(f"  Stats: |probe| mean={np.abs(probe_64).mean():.4f}, max={np.abs(probe_64).max():.4f}")
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        all_passed = False

    # Test 2: Upscale to 128x128
    logger.info("\nTest 2: Upscale to 128x128")
    try:
        probe_128 = extract_and_scale_probe(INTEGRATION_TEST_NPZ, 128)
        assert probe_128.shape == (128, 128), f"Expected (128, 128), got {probe_128.shape}"
        assert probe_128.dtype == np.complex64, f"Expected complex64, got {probe_128.dtype}"
        assert np.iscomplexobj(probe_128), "Expected complex array"
        logger.info(f"  PASS: shape={probe_128.shape}, dtype={probe_128.dtype}")
        logger.info(f"  Stats: |probe| mean={np.abs(probe_128).mean():.4f}, max={np.abs(probe_128).max():.4f}")
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        all_passed = False

    # Test 3: Verify phase structure preserved (center region should have similar phase pattern)
    logger.info("\nTest 3: Phase structure preservation check")
    try:
        probe_64 = extract_and_scale_probe(INTEGRATION_TEST_NPZ, 64)
        probe_128 = extract_and_scale_probe(INTEGRATION_TEST_NPZ, 128)

        # Compare center phases - downsample 128 back to 64 and compare
        probe_128_downsampled = zoom(probe_128.real, 0.5, order=3) + 1j * zoom(probe_128.imag, 0.5, order=3)

        # Phase correlation in high-intensity regions
        mask = np.abs(probe_64) > 0.1 * np.abs(probe_64).max()
        if mask.sum() > 10:
            phase_orig = np.angle(probe_64[mask])
            phase_roundtrip = np.angle(probe_128_downsampled[mask])
            phase_diff = np.abs(phase_orig - phase_roundtrip)
            # Wrap phase difference to [-pi, pi]
            phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)
            mean_phase_diff = np.mean(phase_diff)
            logger.info(f"  Mean phase difference in high-intensity region: {mean_phase_diff:.4f} rad")
            # Allow some interpolation error but should be < 0.5 rad
            assert mean_phase_diff < 0.5, f"Phase difference too large: {mean_phase_diff:.4f}"
            logger.info(f"  PASS: Phase structure reasonably preserved")
        else:
            logger.info(f"  SKIP: Not enough high-intensity pixels for phase check")
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        all_passed = False

    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("All probe extraction tests PASSED")
    else:
        logger.error("Some probe extraction tests FAILED")
    logger.info("=" * 60)

    return all_passed


# =============================================================================
# Phase 2: Grid Data Simulation
# =============================================================================

# Import simulation dependencies (deferred to avoid slow TF import on --help)
_sim_imports_done = False

def _ensure_sim_imports():
    """Lazy import of simulation dependencies."""
    global _sim_imports_done
    if _sim_imports_done:
        return
    global p, data_preprocessing, memoize_disk_and_memory
    from ptycho.misc import memoize_disk_and_memory
    from ptycho import params as p
    from ptycho import data_preprocessing
    _sim_imports_done = True


def simulate_grid_data(
    probe: np.ndarray,
    N: int,
    gridsize: int = 2,
    n_train: int = 500,
    n_test: int = 50,
    nphotons: float = 1e7,
) -> Tuple:
    """Generate grid-sampled data following ptycho_lines.ipynb pattern.

    Uses data_preprocessing.generate_data() which properly handles:
    - Coordinate tracking (coords_nominal, coords_true)
    - Intensity scale derivation
    - Ground truth clipping for stitched comparison

    Contract: DATA-GEN-001
    - Uses 'lines' data source pattern
    - Object size ~3x N (notebook convention)
    - Returns memoizable tuple for caching

    Args:
        probe: Complex probe array (N, N)
        N: Patch size (64 or 128)
        gridsize: Grid grouping (1 or 2)
        n_train: Number of training images
        n_test: Number of test images
        nphotons: Photon count

    Returns:
        Tuple of (ptycho_dataset, YY_ground_truth, norm_Y_I_test)
    """
    _ensure_sim_imports()

    # Object size - smaller than notebook convention to fit in memory
    # Notebook used ~3x N but that's too large for test datasets
    object_size = 2 * N  # 128 for N=64, 256 for N=128

    # Scale outer_offset proportionally to N
    outer_offset_train = max(4, N // 16)  # 4 for N=64, 8 for N=128
    outer_offset_test = max(10, N // 6)   # 10 for N=64, ~21 for N=128

    # Set params.cfg following notebook init() pattern
    p.cfg['N'] = N
    p.cfg['gridsize'] = gridsize
    p.cfg['offset'] = 4
    p.cfg['outer_offset_train'] = outer_offset_train
    p.cfg['outer_offset_test'] = outer_offset_test
    p.cfg['nimgs_train'] = n_train
    p.cfg['nimgs_test'] = n_test
    p.cfg['nphotons'] = nphotons
    p.cfg['size'] = object_size
    p.cfg['data_source'] = 'lines'
    p.cfg['set_phi'] = False
    p.cfg['max_position_jitter'] = 3
    p.cfg['sim_jitter_scale'] = 0.0
    p.cfg['object.big'] = True
    p.cfg['probe.trainable'] = False
    p.cfg['intensity_scale.trainable'] = True

    # Set probe in params.cfg as TF tensor (required by data_preprocessing.generate_data)
    import tensorflow as tf
    probe_tf = tf.convert_to_tensor(probe, tf.complex64)[..., None]
    p.cfg['probe'] = probe_tf

    logger.info(f"Simulating grid data: N={N}, gridsize={gridsize}, "
                f"n_train={n_train}, n_test={n_test}, nphotons={nphotons:.0e}")
    logger.info(f"Object size: {object_size}, outer_offset: train={outer_offset_train}, test={outer_offset_test}")

    # Generate data via notebook code path
    result = data_preprocessing.generate_data(probeGuess=probe)

    X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, \
        YY_ground_truth, ptycho_dataset, YY_test_full, norm_Y_I_test = result

    logger.info(f"Generated: X_train={X_train.shape}, X_test={X_test.shape}")
    if YY_ground_truth is not None:
        logger.info(f"Ground truth shape: {YY_ground_truth.shape}")

    return ptycho_dataset, YY_ground_truth, norm_Y_I_test


def _test_simulate_grid_data() -> bool:
    """Inline test for grid data simulation.

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("=" * 60)
    logger.info("Testing simulate_grid_data()")
    logger.info("=" * 60)

    all_passed = True

    # Test 1: Generate small dataset with gridsize=2
    logger.info("\nTest 1: Generate small dataset with gridsize=2, N=64")
    try:
        probe = extract_and_scale_probe(INTEGRATION_TEST_NPZ, 64)
        dataset, ground_truth, norm_Y_I = simulate_grid_data(
            probe, N=64, gridsize=2, n_train=10, n_test=4, nphotons=1e7
        )

        assert dataset.train_data is not None, "train_data should not be None"
        assert dataset.test_data is not None, "test_data should not be None"
        assert ground_truth is not None, "ground_truth should not be None"
        assert norm_Y_I > 0, f"norm_Y_I should be positive, got {norm_Y_I}"

        # Check shapes
        assert dataset.train_data.X.shape[1:3] == (64, 64), \
            f"Expected X shape (*, 64, 64, *), got {dataset.train_data.X.shape}"
        assert dataset.test_data.X.shape[1:3] == (64, 64), \
            f"Expected X shape (*, 64, 64, *), got {dataset.test_data.X.shape}"

        # For gridsize=2, last dimension should be 4
        assert dataset.train_data.X.shape[-1] == 4, \
            f"Expected 4 channels for gridsize=2, got {dataset.train_data.X.shape[-1]}"

        logger.info(f"  PASS: dataset.train_data.X.shape={dataset.train_data.X.shape}")
        logger.info(f"  PASS: dataset.test_data.X.shape={dataset.test_data.X.shape}")
        logger.info(f"  PASS: ground_truth.shape={ground_truth.shape}")
        logger.info(f"  PASS: norm_Y_I={norm_Y_I:.4f}")
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 2: Generate with gridsize=1
    logger.info("\nTest 2: Generate small dataset with gridsize=1, N=64")
    try:
        probe = extract_and_scale_probe(INTEGRATION_TEST_NPZ, 64)
        dataset, ground_truth, norm_Y_I = simulate_grid_data(
            probe, N=64, gridsize=1, n_train=10, n_test=4, nphotons=1e7
        )

        # For gridsize=1, last dimension should be 1
        assert dataset.train_data.X.shape[-1] == 1, \
            f"Expected 1 channel for gridsize=1, got {dataset.train_data.X.shape[-1]}"

        logger.info(f"  PASS: dataset.train_data.X.shape={dataset.train_data.X.shape}")
        logger.info(f"  PASS: gridsize=1 generates single-channel data")
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("All grid data simulation tests PASSED")
    else:
        logger.error("Some grid data simulation tests FAILED")
    logger.info("=" * 60)

    return all_passed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Grid Resolution Study: Compare PtychoPINN vs Baseline at different resolutions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic study with default settings
  python scripts/studies/grid_resolution_study.py

  # Custom study with high resolution and more training data
  python scripts/studies/grid_resolution_study.py \\
      --resolutions 64 128 256 \\
      --n-train 1000 \\
      --nepochs 100 \\
      --output-dir high_res_study

  # Quick test with minimal training
  python scripts/studies/grid_resolution_study.py \\
      --resolutions 64 \\
      --n-train 100 \\
      --n-test 20 \\
      --nepochs 10
        """
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('grid_resolution_study_outputs'),
        help='Output directory for results (default: grid_resolution_study_outputs)'
    )
    parser.add_argument(
        '--nepochs',
        type=int,
        default=30,
        help='Number of training epochs (default: 30)'
    )
    parser.add_argument(
        '--gridsize',
        type=int,
        default=2,
        choices=[1, 2],
        help='Grid size for sampling (1 or 2, default: 2)'
    )
    parser.add_argument(
        '--nphotons',
        type=float,
        default=1e7,
        help='Photon count for simulation (default: 1e7)'
    )
    parser.add_argument(
        '--n-train',
        type=int,
        default=500,
        help='Number of training images (default: 500)'
    )
    parser.add_argument(
        '--n-test',
        type=int,
        default=50,
        help='Number of test images (default: 50)'
    )
    parser.add_argument(
        '--resolutions',
        type=int,
        nargs='+',
        default=[64, 128],
        help='Resolutions (N values) to test (default: 64 128)'
    )

    return parser.parse_args()


def main():
    """Main orchestration function for the grid resolution study."""
    args = parse_args()

    # Test mode: run inline tests when output-dir is "__test__"
    if str(args.output_dir) == '__test__':
        logger.info("Running inline tests...")
        success = True
        success = _test_extract_and_scale_probe() and success
        success = _test_simulate_grid_data() and success
        sys.exit(0 if success else 1)

    logger.info("=" * 80)
    logger.info("Grid Resolution Study")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Training epochs: {args.nepochs}")
    logger.info(f"Grid size: {args.gridsize}")
    logger.info(f"Photon count: {args.nphotons:.1e}")
    logger.info(f"Training images: {args.n_train}")
    logger.info(f"Test images: {args.n_test}")
    logger.info(f"Resolutions to test: {args.resolutions}")
    logger.info(f"Integration test dataset: {INTEGRATION_TEST_NPZ}")
    logger.info("=" * 80)

    # TODO: Implement workflow
    # Phase 1: Data generation for each resolution
    # Phase 2: Train PtychoPINN and Baseline for each resolution
    # Phase 3: Evaluate and compare results
    # Phase 4: Generate visualizations

    logger.warning("Implementation pending - skeleton only")


if __name__ == '__main__':
    main()

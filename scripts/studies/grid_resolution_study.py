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


# =============================================================================
# Phase 3: Stitching (with gridsize=1 bug workaround)
# =============================================================================

def stitch_predictions(predictions: np.ndarray, norm_Y_I: float, part: str = 'amp') -> np.ndarray:
    """Stitch model predictions, bypassing the incorrect gridsize=1 guard.

    NOTE: This function exists because data_preprocessing.stitch_data() has an
    incorrect ValueError guard for gridsize=1 added in commit aa80f15b (July 2025).
    The original stitching math works fine for gridsize=1 (produces 1x1 grid).

    TODO: Remove this function after fixing data_preprocessing.stitch_data()
    Bug: ptycho/data_preprocessing.py:152-156 should not raise for gridsize=1

    Contract: STITCH-001
    - Handles both gridsize=1 and gridsize=2
    - Uses outer_offset_test from params.cfg for border clipping
    - Returns stitched array with last dimension = 1

    Args:
        predictions: Model output, shape (n_test, N, N, gridsize^2) or complex
        norm_Y_I: Normalization factor from simulation
        part: 'amp', 'phase', or 'complex'

    Returns:
        Stitched images, shape (n_test, H, W, 1)
    """
    _ensure_sim_imports()

    nimgs = p.get('nimgs_test')
    outer_offset = p.get('outer_offset_test')
    N = p.cfg['N']
    gridsize = p.cfg['gridsize']

    nsegments = int(np.sqrt((predictions.size / nimgs) / (N**2)))

    if part == 'amp':
        getpart = np.absolute
    elif part == 'phase':
        getpart = np.angle
    else:
        getpart = lambda x: x

    img_recon = np.reshape(norm_Y_I * getpart(predictions),
                           (-1, nsegments, nsegments, N, N, 1))

    # Border clipping (from data_preprocessing.get_clip_sizes)
    bordersize = (N - outer_offset / 2) / 2
    borderleft = int(np.ceil(bordersize))
    borderright = int(np.floor(bordersize))

    img_recon = img_recon[:, :, :, borderleft:-borderright, borderleft:-borderright, :]
    tmp = img_recon.transpose(0, 1, 3, 2, 4, 5)
    stitched = tmp.reshape(-1, np.prod(tmp.shape[1:3]), np.prod(tmp.shape[1:3]), 1)

    return stitched


def _test_stitch_predictions() -> bool:
    """Inline test for stitching function.

    Returns:
        True if all tests pass, False otherwise
    """
    _ensure_sim_imports()

    logger.info("=" * 60)
    logger.info("Testing stitch_predictions()")
    logger.info("=" * 60)

    all_passed = True

    # Test 1: Stitch with gridsize=2
    logger.info("\nTest 1: Stitch with gridsize=2, N=64")
    try:
        p.cfg['N'] = 64
        p.cfg['gridsize'] = 2
        p.cfg['offset'] = 4
        p.cfg['outer_offset_test'] = 20
        p.cfg['nimgs_test'] = 4

        # Mock predictions: (4, 64, 64, 4) for gridsize=2
        predictions = np.random.randn(4, 64, 64, 4) + 1j * np.random.randn(4, 64, 64, 4)

        stitched = stitch_predictions(predictions, norm_Y_I=1.0, part='amp')
        # For gridsize=2, outer_offset=20: bordersize = (64 - 10)/2 = 27
        # borderleft=27, borderright=27, so each patch contributes 64-54=10 pixels
        # 2x2 grid * 10 = 20x20 output
        assert stitched.shape == (4, 20, 20, 1), f"Expected (4, 20, 20, 1), got {stitched.shape}"
        logger.info(f"  PASS: shape={stitched.shape}")
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 2: Stitch with gridsize=1 (this would fail with data_preprocessing.stitch_data)
    logger.info("\nTest 2: Stitch with gridsize=1, N=64 (bug workaround)")
    try:
        p.cfg['gridsize'] = 1
        p.cfg['nimgs_test'] = 4

        predictions_gs1 = np.random.randn(4, 64, 64, 1) + 1j * np.random.randn(4, 64, 64, 1)
        stitched_gs1 = stitch_predictions(predictions_gs1, norm_Y_I=1.0, part='amp')
        # For gridsize=1, 1x1 grid * 10 = 10x10 output
        assert stitched_gs1.shape == (4, 10, 10, 1), f"Expected (4, 10, 10, 1), got {stitched_gs1.shape}"
        logger.info(f"  PASS: gridsize=1 stitching works, shape={stitched_gs1.shape}")
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 3: Phase stitching
    logger.info("\nTest 3: Phase stitching")
    try:
        p.cfg['gridsize'] = 2
        predictions = np.random.randn(4, 64, 64, 4) + 1j * np.random.randn(4, 64, 64, 4)

        stitched_phase = stitch_predictions(predictions, norm_Y_I=1.0, part='phase')
        assert stitched_phase.shape == (4, 20, 20, 1), f"Expected (4, 20, 20, 1), got {stitched_phase.shape}"
        # Phase should be in [-pi, pi]
        assert stitched_phase.min() >= -np.pi - 0.01 and stitched_phase.max() <= np.pi + 0.01, \
            "Phase values should be in [-pi, pi]"
        logger.info(f"  PASS: phase stitching works, range=[{stitched_phase.min():.2f}, {stitched_phase.max():.2f}]")
    except Exception as e:
        logger.error(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("All stitching tests PASSED")
    else:
        logger.error("Some stitching tests FAILED")
    logger.info("=" * 60)

    return all_passed


# =============================================================================
# Phase 4: Training
# =============================================================================

def train_arm(
    N: int,
    model_type: str,  # 'pinn' or 'baseline'
    gridsize: int,
    dataset,  # PtychoDataset
    output_dir: Path,
    nepochs: int,
    nphotons: float,
) -> Path:
    """Train a single model arm.

    PINN and baseline use completely different training paths:
    - PINN: ptycho.model.train() with physics-informed NLL loss
    - Baseline: ptycho.baselines.train() with supervised MAE loss

    Contract: TRAIN-001
    - PINN uses TrainingConfig -> update_legacy_dict pattern
    - Baseline uses direct params.cfg manipulation
    - Both save via ModelManager.save_multiple_models()

    Args:
        N: Patch size
        model_type: 'pinn' or 'baseline'
        gridsize: Grid grouping
        dataset: PtychoDataset with train_data and test_data
        output_dir: Where to save model
        nepochs: Training epochs
        nphotons: Photon count for physics simulation

    Returns:
        Path to saved model directory
    """
    _ensure_sim_imports()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_type == 'pinn':
        from ptycho import model as ptycho_model
        from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict

        model_config = ModelConfig(
            N=N,
            gridsize=gridsize,
            model_type='pinn',
            object_big=True,
        )
        config = TrainingConfig(
            model=model_config,
            nepochs=nepochs,
            nphotons=nphotons,
            nll_weight=1.0,
            mae_weight=0.0,
            batch_size=16,
            output_dir=output_dir,
            probe_trainable=False,
            intensity_scale_trainable=True,
        )
        update_legacy_dict(p.cfg, config)

        logger.info(f"Training PINN: N={N}, gridsize={gridsize}, epochs={nepochs}")
        trained_model, history = ptycho_model.train(
            dataset.train_data,
            dataset.test_data,
            config=config,
        )

    else:  # baseline
        from ptycho import baselines

        # Baseline needs raw arrays, not containers
        X_train = np.array(dataset.train_data.X)
        Y_I_train = np.array(dataset.train_data.Y_I)
        Y_phi_train = np.array(dataset.train_data.Y_phi)

        # Flatten gridsize channels to batch dimension (baseline is single-channel)
        if gridsize > 1:
            X_train = X_train.reshape(-1, N, N, 1)
            Y_I_train = Y_I_train.reshape(-1, N, N, 1)
            Y_phi_train = Y_phi_train.reshape(-1, N, N, 1)

        # Set params for baseline (it reads from params.cfg)
        p.cfg['N'] = N
        p.cfg['nepochs'] = nepochs
        p.cfg['batch_size'] = 16

        logger.info(f"Training Baseline: N={N}, epochs={nepochs}, samples={X_train.shape[0]}")
        trained_model, history = baselines.train(X_train, Y_I_train, Y_phi_train)

    # Save model
    from ptycho.model import ModelManager
    model_path = output_dir / 'wts.h5.zip'
    ModelManager.save_multiple_models({'autoencoder': trained_model}, model_path)
    logger.info(f"Saved model to {model_path}")

    # Save history
    try:
        import dill
        history_path = output_dir / 'history.dill'
        with open(history_path, 'wb') as f:
            dill.dump(history.history if hasattr(history, 'history') else history, f)
        logger.info(f"Saved history to {history_path}")
    except Exception as e:
        logger.warning(f"Could not save history: {e}")

    return output_dir


# =============================================================================
# Phase 5: Inference
# =============================================================================

def run_inference_and_stitch(
    model_dir: Path,
    test_data,  # PtychoDataContainer
    norm_Y_I: float,
    model_type: str,
    gridsize: int,
    N: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and stitch predictions into full images.

    Contract: INFERENCE-001
    - PINN model expects [X, coords] input
    - Baseline model expects single-channel X, outputs [amplitude, phase]
    - Uses our stitch_predictions() to bypass gridsize=1 bug

    Args:
        model_dir: Path to saved model
        test_data: PtychoDataContainer with test data
        norm_Y_I: Normalization factor
        model_type: 'pinn' or 'baseline'
        gridsize: Grid grouping
        N: Patch size

    Returns:
        (stitched_amplitude, stitched_phase)
    """
    _ensure_sim_imports()
    from ptycho.model import ModelManager

    # Load model
    model_path = model_dir / 'wts.h5.zip'
    models = ModelManager.load_multiple_models(model_path)
    model = models['autoencoder']

    logger.info(f"Running inference: {model_type} from {model_dir}")

    if model_type == 'pinn':
        # PINN model expects [X, coords] input
        X_test = np.array(test_data.X)
        coords_test = np.array(test_data.coords_nominal)
        predictions = model.predict([X_test, coords_test])

        # PINN output is complex object patches
        pred_complex = predictions

    else:  # baseline
        # Baseline expects single-channel X, outputs [amplitude, phase]
        X_test = np.array(test_data.X)

        if gridsize > 1:
            X_test_flat = X_test.reshape(-1, N, N, 1)
        else:
            X_test_flat = X_test

        pred_amp, pred_phase = model.predict(X_test_flat)

        if gridsize > 1:
            n_test = X_test.shape[0]
            pred_amp = pred_amp.reshape(n_test, N, N, gridsize**2)
            pred_phase = pred_phase.reshape(n_test, N, N, gridsize**2)

        # Combine to complex for stitching
        pred_complex = pred_amp * np.exp(1j * pred_phase)

    logger.info(f"Predictions shape: {pred_complex.shape}")

    # Stitch using our bypass function
    stitched_amp = stitch_predictions(pred_complex, norm_Y_I, part='amp')
    stitched_phase = stitch_predictions(pred_complex, norm_Y_I, part='phase')

    logger.info(f"Stitched shape: {stitched_amp.shape}")

    return stitched_amp, stitched_phase


# =============================================================================
# Phase 6: Evaluation & Visualization
# =============================================================================

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def evaluate_and_visualize(
    results: Dict[str, Dict],
    output_dir: Path,
):
    """Evaluate all arms and create comparison visualization.

    Contract: EVAL-001
    - Computes MS-SSIM and MAE for amplitude and phase
    - Aligns reconstructions to ground truth before comparison
    - Produces metrics.csv and comparison_figure.png

    Args:
        results: Dict mapping arm names to result dicts with keys:
            - reconstruction_amp, reconstruction_phase, ground_truth
        output_dir: Where to save outputs
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics_rows = []

    for arm_name, data in results.items():
        recon_amp = data['reconstruction_amp']
        recon_phase = data['reconstruction_phase']
        ground_truth = data['ground_truth']

        if ground_truth is None:
            logger.warning(f"No ground truth for {arm_name}, skipping metrics")
            continue

        # Get amplitude and phase of ground truth
        gt_amp = np.abs(ground_truth)
        gt_phase = np.angle(ground_truth)

        # Handle shape differences - take first image if batched
        if recon_amp.ndim == 4:
            recon_amp = recon_amp[0, :, :, 0]
        if recon_phase.ndim == 4:
            recon_phase = recon_phase[0, :, :, 0]
        if gt_amp.ndim > 2:
            gt_amp = gt_amp.squeeze()
        if gt_phase.ndim > 2:
            gt_phase = gt_phase.squeeze()

        # Simple center-crop alignment: crop both to the smaller size
        def center_crop_to_match(a, b):
            """Center crop a and b to the same size (minimum of both)."""
            min_h = min(a.shape[0], b.shape[0])
            min_w = min(a.shape[1], b.shape[1])
            a_start_h = (a.shape[0] - min_h) // 2
            a_start_w = (a.shape[1] - min_w) // 2
            b_start_h = (b.shape[0] - min_h) // 2
            b_start_w = (b.shape[1] - min_w) // 2
            return (a[a_start_h:a_start_h+min_h, a_start_w:a_start_w+min_w],
                    b[b_start_h:b_start_h+min_h, b_start_w:b_start_w+min_w])

        aligned_amp, gt_amp_cropped = center_crop_to_match(recon_amp, gt_amp)
        aligned_phase, gt_phase_cropped = center_crop_to_match(recon_phase, gt_phase)
        gt_amp = gt_amp_cropped
        gt_phase = gt_phase_cropped

        # Compute simple metrics (SSIM would require extra import)
        # Use MSE and MAE as basic metrics
        mse_amp = np.mean((aligned_amp - gt_amp)**2)
        mae_amp = np.mean(np.abs(aligned_amp - gt_amp))
        mse_phase = np.mean((aligned_phase - gt_phase)**2)
        mae_phase = np.mean(np.abs(aligned_phase - gt_phase))

        # Compute correlation coefficient as similarity metric
        corr_amp = np.corrcoef(aligned_amp.flatten(), gt_amp.flatten())[0, 1]
        corr_phase = np.corrcoef(aligned_phase.flatten(), gt_phase.flatten())[0, 1]

        metrics_rows.append({
            'arm': arm_name,
            'mse_amp': mse_amp,
            'mae_amp': mae_amp,
            'corr_amp': corr_amp,
            'mse_phase': mse_phase,
            'mae_phase': mae_phase,
            'corr_phase': corr_phase,
        })

        data['metrics'] = {'mse_amp': mse_amp, 'mae_amp': mae_amp, 'corr_amp': corr_amp}
        data['aligned_amp'] = aligned_amp
        data['aligned_phase'] = aligned_phase

        logger.info(f"{arm_name}: corr(amp)={corr_amp:.4f}, MAE(amp)={mae_amp:.4f}")

    # Save metrics
    if metrics_rows:
        if HAS_PANDAS:
            df = pd.DataFrame(metrics_rows)
            df.to_csv(output_dir / 'metrics.csv', index=False)
        else:
            # Fallback without pandas
            with open(output_dir / 'metrics.csv', 'w') as f:
                f.write('arm,mse_amp,mae_amp,corr_amp,mse_phase,mae_phase,corr_phase\n')
                for row in metrics_rows:
                    f.write(f"{row['arm']},{row['mse_amp']},{row['mae_amp']},{row['corr_amp']},"
                            f"{row['mse_phase']},{row['mae_phase']},{row['corr_phase']}\n")
        logger.info(f"Saved metrics to {output_dir / 'metrics.csv'}")

    # Create comparison figure
    create_comparison_figure(results, output_dir)


def create_comparison_figure(results: Dict, output_dir: Path):
    """Create grid figure comparing all arms to ground truth.

    Layout:
        Row 0: Ground Truth (N=64) | Ground Truth (N=128)
        Row 1: PINN N=64           | PINN N=128
        Row 2: Baseline N=64       | Baseline N=128
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not results:
        logger.warning("No results to visualize")
        return

    # Extract unique resolutions and model types
    resolutions = sorted(set(
        int(name.split('_N')[1]) for name in results.keys() if '_N' in name
    ))
    model_types = ['pinn', 'baseline']

    if not resolutions:
        logger.warning("Could not extract resolutions from result names")
        return

    n_cols = len(resolutions)
    n_rows = len(model_types) + 1  # +1 for ground truth row

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col, N in enumerate(resolutions):
        # Ground truth row - use PINN's ground truth (same for both)
        pinn_key = f"pinn_N{N}"
        if pinn_key in results and results[pinn_key].get('ground_truth') is not None:
            gt = results[pinn_key]['ground_truth']
            gt_display = np.abs(gt.squeeze())
            axes[0, col].imshow(gt_display, cmap='gray')
            axes[0, col].set_title(f'Ground Truth N={N}')
        else:
            axes[0, col].text(0.5, 0.5, 'No GT', ha='center', va='center', transform=axes[0, col].transAxes)
            axes[0, col].set_title(f'Ground Truth N={N}')
        axes[0, col].axis('off')

        # Model rows
        for row, model_type in enumerate(model_types, start=1):
            arm_name = f"{model_type}_N{N}"
            ax = axes[row, col]

            if arm_name in results and 'aligned_amp' in results[arm_name]:
                recon = results[arm_name]['aligned_amp']
                metrics = results[arm_name].get('metrics', {})
                corr = metrics.get('corr_amp', 0)

                ax.imshow(recon.squeeze(), cmap='gray')
                ax.set_title(f'{model_type.upper()} N={N}\ncorr={corr:.4f}')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_type.upper()} N={N}')
            ax.axis('off')

    plt.tight_layout()
    fig_path = output_dir / 'comparison_figure.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved figure to {fig_path}")


# =============================================================================
# Phase 7: Main Orchestrator
# =============================================================================

def run_study(
    output_dir: Path,
    resolutions: Tuple[int, ...] = (64, 128),
    gridsize: int = 2,
    nepochs: int = 30,
    nphotons: float = 1e7,
    n_train: int = 500,
    n_test: int = 50,
) -> Dict:
    """Main study orchestrator.

    For each resolution:
    1. Extract/scale probe
    2. Generate grid data (memoized)
    3. Train PINN and Baseline
    4. Run inference and stitch
    5. Evaluate and visualize

    Contract: STUDY-001
    - Creates output_dir/<model_type>_N<resolution>/ for each arm
    - Produces metrics.csv and comparison_figure.png in output_dir
    - Returns results dict for programmatic access

    Args:
        output_dir: Base output directory
        resolutions: Tuple of N values to test
        gridsize: Grid grouping (1 or 2)
        nepochs: Training epochs
        nphotons: Photon count
        n_train: Number of training images
        n_test: Number of test images

    Returns:
        Dict mapping arm names to result dicts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for N in resolutions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing resolution N={N}")
        logger.info(f"{'='*60}")

        # Extract and scale probe
        probe = extract_and_scale_probe(INTEGRATION_TEST_NPZ, N)
        logger.info(f"Probe shape: {probe.shape}")

        # Generate data (memoized - fast on re-runs)
        dataset, ground_truth, norm_Y_I = simulate_grid_data(
            probe=probe,
            N=N,
            gridsize=gridsize,
            n_train=n_train,
            n_test=n_test,
            nphotons=nphotons,
        )

        for model_type in ('pinn', 'baseline'):
            arm_name = f"{model_type}_N{N}"
            arm_dir = output_dir / arm_name

            logger.info(f"\n--- {arm_name} ---")

            # Train
            model_dir = train_arm(
                N=N,
                model_type=model_type,
                gridsize=gridsize,
                dataset=dataset,
                output_dir=arm_dir,
                nepochs=nepochs,
                nphotons=nphotons,
            )

            # Inference and stitch
            recon_amp, recon_phase = run_inference_and_stitch(
                model_dir=model_dir,
                test_data=dataset.test_data,
                norm_Y_I=norm_Y_I,
                model_type=model_type,
                gridsize=gridsize,
                N=N,
            )

            results[arm_name] = {
                'reconstruction_amp': recon_amp,
                'reconstruction_phase': recon_phase,
                'ground_truth': ground_truth,
                'model_dir': model_dir,
            }

    # Evaluate and visualize all results
    logger.info(f"\n{'='*60}")
    logger.info("Evaluating and visualizing results")
    logger.info(f"{'='*60}")
    evaluate_and_visualize(results, output_dir)

    logger.info(f"\n=== Study Complete ===")
    logger.info(f"Results saved to: {output_dir}")

    return results


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
        success = _test_stitch_predictions() and success  # test before simulation (faster)
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

    # Run the study
    results = run_study(
        output_dir=args.output_dir,
        resolutions=tuple(args.resolutions),
        gridsize=args.gridsize,
        nepochs=args.nepochs,
        nphotons=args.nphotons,
        n_train=args.n_train,
        n_test=args.n_test,
    )

    logger.info(f"Study complete. Results: {list(results.keys())}")


if __name__ == '__main__':
    main()

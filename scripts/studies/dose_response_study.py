#!/usr/bin/env python3
"""
dose_response_study.py - Synthetic Dose Response & Loss Comparison Study

This script implements STUDY-SYNTH-DOSE-COMPARISON-001, comparing PtychoPINN
reconstruction quality under High Dose (1e9 photons) vs. Low Dose (1e4 photons)
conditions using identical scan trajectories.

Goals:
1. Compare reconstruction quality under High vs Low dose conditions
2. Evaluate robustness of Poisson NLL loss vs MAE loss across dose regimes
3. Produce a publication-ready 6-panel figure visualizing results
4. Demonstrate a "Pure Python" workflow using library imports directly

Architecture:
    - Uses Modern Coordinate-Based System (CONVENTION-001)
    - Calls update_legacy_dict before legacy module usage (CONFIG-001)
    - Follows data contracts (DATA-001) and normalization (NORMALIZATION-001)

Usage:
    python scripts/studies/dose_response_study.py [--output-dir DIR] [--nepochs N]

References:
    - Implementation plan: plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/implementation.md
    - Spec: docs/specs/spec-ptycho-core.md (Physics/Normalization)
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# PtychoPINN imports - modern config system
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict  # noqa: E402
from ptycho import params as p  # noqa: E402

# PtychoPINN imports - simulation and training
from scripts.simulation.synthetic_helpers import make_probe, simulate_nongrid_raw_data  # noqa: E402

# Lazy imports for training (deferred until after params.cfg is set)


# =============================================================================
# Phase A: Orchestration & Data Fabric
# =============================================================================

def verify_params_setup(config: TrainingConfig) -> None:
    """
    A0: Verify that update_legacy_dict properly populates params.cfg.

    This is the nucleus/test-first gate for Phase A.
    """
    # Critical: update params.cfg before any legacy module usage (CONFIG-001)
    update_legacy_dict(p.cfg, config)

    logger.info("=== Verifying params.cfg population ===")
    logger.info(f"params.cfg['N'] = {p.cfg.get('N')} (expected: {config.model.N})")
    logger.info(f"params.cfg['gridsize'] = {p.cfg.get('gridsize')} (expected: {config.model.gridsize})")
    logger.info(f"params.cfg['nphotons'] = {p.cfg.get('nphotons')} (expected: {config.nphotons})")

    # Verify critical params are set correctly
    assert p.cfg.get('N') == config.model.N, f"N mismatch: {p.cfg.get('N')} != {config.model.N}"
    assert p.cfg.get('gridsize') == config.model.gridsize, "gridsize mismatch"
    assert p.cfg.get('nphotons') == config.nphotons, "nphotons mismatch"

    logger.info("=== params.cfg verification PASSED ===")


def generate_ground_truth(N: int = 64, object_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """
    A1: Generate ground truth object and probe in memory.

    Uses an amplitude-only object (zero phase) for controlled comparison,
    isolating dose/loss effects from phase-wrapping ambiguities.

    Args:
        N: Probe size (patch size)
        object_size: Full object size

    Returns:
        Tuple of (objectGuess, probeGuess) as complex64 arrays
    """
    logger.info("=== Generating Ground Truth ===")

    # Generate object using lines pattern (amplitude-only)
    # mk_lines_img is in ptycho.diffsim, returns (N, N, 3) with Gaussian filtering
    from ptycho.diffsim import mk_lines_img

    # Generate lines pattern - match notebook (ptycho_lines.ipynb) approach:
    # sim_object_image generates at 2*size with 400 lines, then crops to size
    # Scale nlines to maintain similar density: notebook uses 400 lines for 784 pixels
    full_size = 2 * object_size
    nlines = int(400 * full_size / 784)  # Scale lines to match notebook density
    obj_full = mk_lines_img(N=full_size, nlines=nlines)

    # Crop to object_size (center crop like sim_object_image does)
    crop_start = object_size // 2
    crop_end = full_size - object_size // 2
    obj_full = obj_full[crop_start:crop_end, crop_start:crop_end, :]

    # Convert to complex (amplitude-only: real-valued, zero phase)
    # Normalize to reasonable amplitude range [0.5, 1.5]
    obj_amp = obj_full[:, :, 0]  # Take first channel
    obj_amp = obj_amp / obj_amp.max()  # Normalize to [0, 1]
    obj_amp = 0.5 + obj_amp  # Shift to [0.5, 1.5]

    objectGuess = obj_amp.astype(np.complex64)

    # Verify amplitude-only constraint
    assert np.allclose(objectGuess.imag, 0), "Object should be amplitude-only (zero phase)"

    logger.info(f"Object shape: {objectGuess.shape}")
    logger.info(f"Object amplitude range: [{np.abs(objectGuess).min():.3f}, {np.abs(objectGuess).max():.3f}]")

    probeGuess = make_probe(N, mode="idealized", scale=4.0)

    logger.info(f"Probe shape: {probeGuess.shape}")
    logger.info(f"Probe amplitude range: [{np.abs(probeGuess).min():.3f}, {np.abs(probeGuess).max():.3f}]")

    return objectGuess, probeGuess


def make_config(
    nphotons: float,
    loss_type: str,
    output_subdir: str,
    base_output_dir: Path,
    N: int = 64,
    gridsize: int = 2,
    n_images: int = 2000,
    nepochs: int = 50,
    enable_oversampling: bool = False
) -> TrainingConfig:
    """
    B1: Create TrainingConfig for a specific experimental arm.

    Args:
        nphotons: Photon count (1e9 for high dose, 1e4 for low dose)
        loss_type: 'nll' for Poisson NLL, 'mae' for MAE loss
        output_subdir: Subdirectory name for this arm's outputs
        base_output_dir: Base output directory
        N: Patch size
        gridsize: Grid size for grouping
        n_images: Number of raw images to simulate
        nepochs: Number of training epochs
        enable_oversampling: Enable K choose C oversampling for test data

    Returns:
        Configured TrainingConfig
    """
    nll_weight = 1.0 if loss_type == 'nll' else 0.0
    mae_weight = 1.0 if loss_type == 'mae' else 0.0

    model_config = ModelConfig(
        N=N,
        gridsize=gridsize,
        model_type='pinn'
    )

    config = TrainingConfig(
        model=model_config,
        n_groups=n_images,  # n_groups controls raw positions generated
        nphotons=nphotons,
        nll_weight=nll_weight,
        mae_weight=mae_weight,
        nepochs=nepochs,
        batch_size=16,
        output_dir=base_output_dir / output_subdir,
        neighbor_count=7,  # Higher K for oversampling
        enable_oversampling=enable_oversampling,  # Allow oversampling when needed
        probe_trainable=False,
        intensity_scale_trainable=True
    )

    return config


def simulate_datasets(
    objectGuess: np.ndarray,
    probeGuess: np.ndarray,
    base_output_dir: Path,
    n_train: int = 2000,
    n_test: int = 128,
    N: int = 64,
    gridsize: int = 2,
    nepochs: int = 50
) -> Dict[str, Dict[str, Any]]:
    """
    A2: Generate all 4 experimental datasets (High/Low x Train/Test).

    Uses the same seed for train sets to ensure identical scan trajectories,
    isolating photon statistics as the only variable.

    Note: Simulation is done with gridsize=1 (individual patterns), and grouping
    happens downstream in create_ptycho_data_container during training. This is
    because RawData.from_simulation only supports gridsize=1.

    Returns:
        Dictionary mapping arm names to {config, train_data, test_data}
    """
    logger.info("=== Simulating Datasets ===")

    buffer = 15.0  # Minimum distance from object edges

    arms = {
        'high_nll': {'nphotons': 1e9, 'loss_type': 'nll'},
        'high_mae': {'nphotons': 1e9, 'loss_type': 'mae'},
        'low_nll': {'nphotons': 1e4, 'loss_type': 'nll'},
        'low_mae': {'nphotons': 1e4, 'loss_type': 'mae'},
    }

    datasets = {}

    for arm_name, arm_params in arms.items():
        logger.info(f"\n--- Simulating {arm_name} ---")

        # Create training config for this arm (with target gridsize)
        config = make_config(
            nphotons=arm_params['nphotons'],
            loss_type=arm_params['loss_type'],
            output_subdir=arm_name,
            base_output_dir=base_output_dir,
            N=N,
            gridsize=gridsize,
            n_images=n_train,
            nepochs=nepochs,
            enable_oversampling=False
        )

        # Generate training data with seed=1 for consistent trajectories
        train_data = simulate_nongrid_raw_data(
            objectGuess,
            probeGuess,
            N=N,
            n_images=n_train,
            nphotons=arm_params['nphotons'],
            seed=1,
            buffer=buffer,
        )

        # Generate test data with seed=2
        test_data = simulate_nongrid_raw_data(
            objectGuess,
            probeGuess,
            N=N,
            n_images=n_test,
            nphotons=arm_params['nphotons'],
            seed=2,
            buffer=buffer,
        )

        datasets[arm_name] = {
            'config': config,  # Store the actual training config (with gridsize=2)
            'train_data': train_data,
            'test_data': test_data,
            'nphotons': arm_params['nphotons'],
            'loss_type': arm_params['loss_type']
        }

        logger.info(f"Train data shape: {train_data.diff3d.shape}")
        logger.info(f"Test data shape: {test_data.diff3d.shape}")

    return datasets


# =============================================================================
# Grid Mode Simulation (ALIGN-DOSE-STUDY-GRID-001)
# =============================================================================

def simulate_datasets_grid_mode(
    probeGuess: np.ndarray,
    base_output_dir: Path,
    nepochs: int = 50,
    nimgs_train: int = 2,
    nimgs_test: int = 2
) -> Dict[str, Dict[str, Any]]:
    """
    Grid-based simulation matching notebooks/dose_dependence.ipynb.

    This uses the legacy mk_simdata() path with fixed grid extraction,
    enabling reproducibility validation against the notebook results.

    Args:
        probeGuess: Probe function as complex64 array
        base_output_dir: Base output directory for artifacts
        nepochs: Number of training epochs
        nimgs_train: Number of training images (default: 2 per notebook)
        nimgs_test: Number of test images (default: 2 per notebook)

    Returns:
        Dictionary mapping arm names to {config, train_data, test_data}

    References:
        - Implementation plan: plans/active/ALIGN-DOSE-STUDY-GRID-001/implementation.md
        - Notebook: notebooks/dose_dependence.ipynb
        - CONVENTION-001: Grid mode explicitly uses legacy system
    """
    from ptycho.diffsim import mk_simdata

    logger.info("=== Grid Mode Simulation (Legacy mk_simdata) ===")

    # Grid mode parameters - simplified for compatibility
    # Using gridsize=1 to avoid model architecture complexity with gridsize>1
    # Reduced from notebook defaults (N=128) to avoid OOM
    GRID_N = 64
    GRID_SIZE = 2  # Match reference notebook (2x2 patch groups)
    GRID_OFFSET = 4  # Required for patch extraction
    GRID_OUTER_OFFSET_TRAIN = 8
    GRID_OUTER_OFFSET_TEST = 20
    GRID_OBJECT_SIZE = 196  # Scaled down from 392 to match N=64
    GRID_MAX_JITTER = 4  # Must be EVEN for symmetric padding (odd causes N+1 output)

    arms = {
        'high_nll': {'nphotons': 1e9, 'loss_type': 'nll'},
        'high_mae': {'nphotons': 1e9, 'loss_type': 'mae'},
        'low_nll': {'nphotons': 1e4, 'loss_type': 'nll'},
        'low_mae': {'nphotons': 1e4, 'loss_type': 'mae'},
    }

    datasets = {}

    for arm_name, arm_params in arms.items():
        logger.info(f"\n--- Grid Mode: Simulating {arm_name} ---")

        nphotons = arm_params['nphotons']

        # CONFIG-001: Set params.cfg before mk_simdata() call
        # This is the legacy system - explicit params setup required
        p.cfg['N'] = GRID_N
        p.cfg['gridsize'] = GRID_SIZE
        p.cfg['offset'] = GRID_OFFSET
        p.cfg['outer_offset_train'] = GRID_OUTER_OFFSET_TRAIN
        p.cfg['outer_offset_test'] = GRID_OUTER_OFFSET_TEST
        p.cfg['nphotons'] = nphotons
        p.cfg['size'] = GRID_OBJECT_SIZE
        p.cfg['data_source'] = 'lines'
        p.cfg['max_position_jitter'] = GRID_MAX_JITTER
        p.cfg['sim_jitter_scale'] = 0.0
        p.cfg['set_phi'] = False

        logger.info(f"Grid params: N={GRID_N}, gridsize={GRID_SIZE}, "
                    f"size={GRID_OBJECT_SIZE}, nphotons={nphotons:.0e}")

        # Generate training data
        # mk_simdata returns: X, Y_I, Y_phi, intensity_scale, YY_full, norm_Y_I, coords
        # coords is a tuple ((coords, true_coords)) - we need to extract and reshape
        train_result = mk_simdata(
            n=nimgs_train,
            size=GRID_OBJECT_SIZE,
            probe=probeGuess,
            outer_offset=GRID_OUTER_OFFSET_TRAIN,
            which='train'
        )
        X_train, Y_I_train, Y_phi_train, intensity_scale_train, YY_full_train, norm_Y_I_train, coords_train_raw = train_result

        # Generate test data
        test_result = mk_simdata(
            n=nimgs_test,
            size=GRID_OBJECT_SIZE,
            probe=probeGuess,
            outer_offset=GRID_OUTER_OFFSET_TEST,
            intensity_scale=intensity_scale_train,  # Use same scale for consistency
            which='test'
        )
        X_test, Y_I_test, Y_phi_test, _, YY_full_test, norm_Y_I_test, coords_test_raw = test_result

        # Create PtychoDataContainer directly from mk_simdata outputs
        from ptycho.loader import PtychoDataContainer

        # Convert probe to correct shape if needed
        probe_tensor = probeGuess.astype(np.complex64)

        # mk_simdata returns coords as array of shape (2, B, 1, 2, C) where:
        # - coords[0] = nominal coords (B, 1, 2, C)
        # - coords[1] = true coords (B, 1, 2, C)
        # The model needs these for position-aware reconstruction/stitching
        coords_train = coords_train_raw[0].astype(np.float32)  # Use nominal coords
        coords_test = coords_test_raw[0].astype(np.float32)

        train_container = PtychoDataContainer(
            X=X_train,
            Y_I=Y_I_train,
            Y_phi=Y_phi_train if Y_phi_train is not None else np.zeros_like(Y_I_train),
            norm_Y_I=norm_Y_I_train,
            YY_full=YY_full_train,
            coords_nominal=coords_train,
            coords_true=coords_train,  # Grid mode has no jitter, so nominal = true
            nn_indices=None,
            global_offsets=None,
            local_offsets=None,
            probeGuess=probe_tensor
        )

        test_container = PtychoDataContainer(
            X=X_test,
            Y_I=Y_I_test,
            Y_phi=Y_phi_test if Y_phi_test is not None else np.zeros_like(Y_I_test),
            norm_Y_I=norm_Y_I_test,
            YY_full=YY_full_test,
            coords_nominal=coords_test,
            coords_true=coords_test,
            nn_indices=None,
            global_offsets=None,
            local_offsets=None,
            probeGuess=probe_tensor
        )

        # Create config for this arm (for training parameters)
        nll_weight = 1.0 if arm_params['loss_type'] == 'nll' else 0.0
        mae_weight = 1.0 if arm_params['loss_type'] == 'mae' else 0.0

        model_config = ModelConfig(
            N=GRID_N,
            gridsize=GRID_SIZE,
            model_type='pinn'
        )

        config = TrainingConfig(
            model=model_config,
            n_groups=nimgs_train * (GRID_SIZE ** 2),  # Total groups after gridding
            nphotons=nphotons,
            nll_weight=nll_weight,
            mae_weight=mae_weight,
            nepochs=nepochs,
            batch_size=16,
            output_dir=base_output_dir / arm_name,
            probe_trainable=False,
            intensity_scale_trainable=True
        )

        datasets[arm_name] = {
            'config': config,
            'train_container': train_container,
            'test_container': test_container,
            'nphotons': nphotons,
            'loss_type': arm_params['loss_type'],
            'intensity_scale': intensity_scale_train,
            'grid_mode': True,
            'nimgs_test_param': nimgs_test  # Store for stitching (number of full images)
        }

        logger.info(f"Train data shape: X={X_train.shape}")
        logger.info(f"Test data shape: X={X_test.shape}")

    return datasets


def sanity_check_datasets(datasets: Dict[str, Dict[str, Any]]) -> None:
    """
    A3: Verify dataset shapes and intensity scaling.

    Expected: ~10^5 factor difference in mean intensity between High and Low dose.
    """
    logger.info("\n=== Sanity Check: Dataset Statistics ===")
    logger.info(f"{'Arm':<15} {'nphotons':>12} {'Train Shape':>20} {'Mean Intensity':>18}")
    logger.info("-" * 70)

    intensities = {}
    for arm_name, arm_data in datasets.items():
        train_data = arm_data['train_data']
        mean_intensity = train_data.diff3d.mean()
        intensities[arm_name] = mean_intensity

        logger.info(
            f"{arm_name:<15} {arm_data['nphotons']:>12.0e} "
            f"{str(train_data.diff3d.shape):>20} {mean_intensity:>18.2e}"
        )

    # Verify dose scaling
    high_intensity = intensities['high_nll']
    low_intensity = intensities['low_nll']
    ratio = high_intensity / low_intensity

    logger.info(f"\nIntensity ratio (High/Low): {ratio:.2e}")
    logger.info("Expected ratio: ~1.0 (diffraction is normalized; nphotons affects noise statistics)")

    # Guard against unexpected scaling drift
    if ratio < 0.5 or ratio > 2.0:
        logger.warning(f"Intensity ratio {ratio:.2e} is outside expected normalized range!")
    else:
        logger.info("=== Intensity scaling verification PASSED ===")


def sanity_check_datasets_grid_mode(datasets: Dict[str, Dict[str, Any]]) -> None:
    """
    Sanity check for grid mode datasets.

    Grid mode datasets have containers directly, not RawData objects.
    """
    logger.info("\n=== Sanity Check: Grid Mode Dataset Statistics ===")
    logger.info(f"{'Arm':<15} {'nphotons':>12} {'Train Shape':>25} {'Mean Intensity':>18}")
    logger.info("-" * 75)

    intensities = {}
    for arm_name, arm_data in datasets.items():
        train_container = arm_data['train_container']
        # Access X data (lazy loading)
        X_data = train_container._X_np if hasattr(train_container, '_X_np') else train_container.X
        if hasattr(X_data, 'numpy'):
            X_data = X_data.numpy()
        mean_intensity = X_data.mean()
        intensities[arm_name] = mean_intensity

        logger.info(
            f"{arm_name:<15} {arm_data['nphotons']:>12.0e} "
            f"{str(X_data.shape):>25} {mean_intensity:>18.2e}"
        )

    # Verify dose scaling
    high_intensity = intensities['high_nll']
    low_intensity = intensities['low_nll']
    ratio = high_intensity / low_intensity

    logger.info(f"\nIntensity ratio (High/Low): {ratio:.2e}")
    logger.info("Expected ratio: ~316 (sqrt of 1e9/1e4) for amplitude data")

    if ratio < 100:
        logger.warning(f"Intensity ratio {ratio:.2e} is lower than expected!")
    else:
        logger.info("=== Grid mode intensity scaling verification PASSED ===")


# =============================================================================
# Phase B: Training & Inference Loop
# =============================================================================

def train_all_arms_grid_mode(
    datasets: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Train all arms using grid mode containers directly.

    Grid mode provides PtychoDataContainer objects directly, bypassing RawData.
    This matches the notebook workflow where mk_simdata outputs are used directly.

    Returns:
        Updated datasets dict with training results added
    """
    from ptycho.model_manager import ModelManager
    from ptycho import model as ptycho_model

    logger.info("\n=== Training All Arms (Grid Mode) ===")

    results = {}
    for arm_name, arm_data in datasets.items():
        logger.info(f"\n--- Training {arm_name} (Grid Mode) ---")

        config = arm_data['config']
        train_container = arm_data['train_container']
        test_container = arm_data['test_container']

        # CONFIG-001: update params.cfg before training
        update_legacy_dict(p.cfg, config)

        # Set intensity_scale from simulation
        p.cfg['intensity_scale'] = arm_data.get('intensity_scale', 1.0)

        # Set probe in params.cfg before model creation (matches reference train_pinn.py)
        # This ensures ProbeIllumination layer uses the correct probe
        from ptycho import probe as probe_module
        probe_module.set_probe_guess(None, train_container.probe)

        logger.info(f"[DEBUG] Grid mode: N={config.model.N}, gridsize={config.model.gridsize}")

        # Ensure output directory exists
        config.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Use create_compiled_model which properly sets jit_compile=False
            # This avoids XLA FakeParam errors from tf.cond in reassembly layers
            from ptycho.model import create_compiled_model
            from ptycho import tf_helper as hh

            # create_compiled_model returns (autoencoder, diffraction_to_obj) with proper compilation
            autoencoder, diffraction_to_obj = create_compiled_model(
                config.model.gridsize, config.model.N
            )

            # Prepare training inputs/outputs - must match prepare_inputs/prepare_outputs
            # from ptycho.model for consistent training behavior
            intensity_scale = p.cfg.get('intensity_scale', 1.0)
            X_train = train_container.X
            Y_I_train = train_container.Y_I
            coords_train = train_container.coords_nominal

            # Inputs: scale X by intensity_scale (per prepare_inputs)
            inputs = [X_train * intensity_scale, coords_train]

            # Outputs (per prepare_outputs):
            # 1. Y_I centered by coords, first channel only (for realspace_loss)
            # 2. Scaled diffraction amplitude (for MAE loss)
            # 3. Squared scaled diffraction (for NLL loss)
            Y_I_centered = hh.center_channels(Y_I_train, coords_train)[:, :, :, :1]
            scaled_X = intensity_scale * X_train
            squared_scaled_X = scaled_X ** 2
            outputs = [Y_I_centered, scaled_X, squared_scaled_X]

            # Train the model
            logger.info(f"Training with {len(train_container)} groups, {config.nepochs} epochs")
            logger.info(f"intensity_scale={intensity_scale:.2e}")

            # Use Keras model.fit directly for grid mode (simpler path)
            history = autoencoder.fit(
                inputs,
                outputs,
                epochs=config.nepochs,
                batch_size=config.batch_size,
                verbose=1
            )

            # Copy weights from autoencoder to diffraction_to_obj
            # They share the same encoder weights
            diffraction_to_obj.set_weights(autoencoder.get_weights()[:len(diffraction_to_obj.get_weights())])

            # Update module singletons for model_manager.save() (SINGLETON-SAVE-001)
            ptycho_model.autoencoder = autoencoder
            ptycho_model.diffraction_to_obj = diffraction_to_obj

            # Run inference on test data - use autoencoder.predict() like inference.py does
            # autoencoder returns 3 outputs: [reconstructed_obj, pred_amp, pred_intensity]
            X_test = test_container.X
            coords_test = test_container.coords_nominal

            # Use batch_size=len(X_test) to process all at once, avoiding per-batch aggregation
            reconstructed_obj, pred_amp, pred_intensity = autoencoder.predict(
                [X_test * intensity_scale, coords_test],
                batch_size=len(X_test)
            )
            logger.info(f"[DEBUG] reconstructed_obj shape: {reconstructed_obj.shape}")
            logger.info(f"[DEBUG] X_test shape: {X_test.shape}")

            # Stitch using reassemble_with_config like inference.py does
            from ptycho.inference import reassemble_with_config
            stitch_config = {
                'N': p.cfg['N'],
                'gridsize': p.cfg['gridsize'],
                'offset': p.cfg['offset'],
                'nimgs_test': arm_data.get('nimgs_test_param', 2),
                'outer_offset_test': p.cfg['outer_offset_test']
            }

            stitched_amp = reassemble_with_config(reconstructed_obj, stitch_config, norm_Y_I=1.0, part='amp')
            stitched_phase = reassemble_with_config(reconstructed_obj, stitch_config, norm_Y_I=1.0, part='phase')

            if stitched_amp is not None and stitched_phase is not None:
                reconstruction = {
                    'amplitude': np.squeeze(stitched_amp),
                    'phase': np.squeeze(stitched_phase),
                    'patches': reconstructed_obj
                }
                logger.info(f"Stitched reconstruction shape: {stitched_amp.shape}")
            else:
                logger.warning(f"Stitching failed for {arm_name}, using mean patches")
                reconstruction = {
                    'amplitude': np.mean(np.abs(reconstructed_obj), axis=0).squeeze(),
                    'phase': np.mean(np.angle(reconstructed_obj), axis=0).squeeze(),
                    'patches': reconstructed_obj
                }

            # Save model
            from ptycho.model import IntensityScaler
            from ptycho.custom_layers import (
                FlatToChannelLayer, ScaleLayer, InvScaleLayer,
                ActivationLayer, SquareLayer, TrimReconstructionLayer,
                PadAndDiffractLayer
            )

            intensity_scale = p.cfg.get('intensity_scale', 1.0)
            models_to_save = {
                'autoencoder': autoencoder,
                'diffraction_to_obj': diffraction_to_obj
            }
            model_path = str(config.output_dir / 'wts.h5')
            custom_objects = {
                'IntensityScaler': IntensityScaler,
                'FlatToChannelLayer': FlatToChannelLayer,
                'ScaleLayer': ScaleLayer,
                'InvScaleLayer': InvScaleLayer,
                'ActivationLayer': ActivationLayer,
                'SquareLayer': SquareLayer,
                'TrimReconstructionLayer': TrimReconstructionLayer,
                'PadAndDiffractLayer': PadAndDiffractLayer,
            }
            ModelManager.save_multiple_models(
                models_to_save, model_path, custom_objects, intensity_scale
            )
            logger.info(f"Model saved to: {model_path}.zip")

            results[arm_name] = {
                **arm_data,
                'train_results': {
                    'model_instance': autoencoder,
                    'history': {
                        'train_loss': history.history.get('loss', [])
                    },
                    'reconstructed_obj': reconstructed_obj,
                    'test_container': test_container
                },
                'reconstruction': reconstruction,  # Stitched reconstruction for visualization
                'success': True
            }

            # Log training summary
            if history.history.get('loss'):
                final_loss = history.history['loss'][-1]
                logger.info(f"Final training loss: {final_loss}")

        except Exception as e:
            logger.error(f"Training failed for {arm_name}: {e}")
            import traceback
            traceback.print_exc()
            results[arm_name] = {
                **arm_data,
                'train_results': None,
                'success': False,
                'error': str(e)
            }

    return results


def train_all_arms(
    datasets: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    B2: Execute training for all experimental arms.

    Returns:
        Updated datasets dict with training results added
    """
    # Import training components (after params.cfg is set)
    from ptycho.workflows.backend_selector import train_cdi_model_with_backend
    from ptycho import model_manager

    logger.info("\n=== Training All Arms (Nongrid Mode) ===")

    results = {}
    for arm_name, arm_data in datasets.items():
        logger.info(f"\n--- Training {arm_name} ---")

        config = arm_data['config']
        train_data = arm_data['train_data']

        # Critical: update params.cfg before each training run (CONFIG-001)
        update_legacy_dict(p.cfg, config)

        # DEBUG: Verify gridsize is correctly set before model import
        logger.info(f"[DEBUG] After update_legacy_dict: params.cfg['gridsize'] = {p.cfg.get('gridsize')} (expected: {config.model.gridsize})")

        # XLA translate is enabled by default (use_xla_translate=True in params.py)
        # This provides better performance and avoids the non-XLA shape mismatch bug (TF-NON-XLA-SHAPE-001)

        # Ensure output directory exists
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Train the model
        try:
            train_results = train_cdi_model_with_backend(train_data, None, config)

            # Save the model bundle for inference (includes custom layers).
            model_manager.save(str(config.output_dir))
            logger.info("Model saved to: %s/wts.h5.zip", config.output_dir)

            results[arm_name] = {
                **arm_data,
                'train_results': train_results,
                'success': True
            }

            # Log training summary
            history = train_results.get('history', {})
            if 'train_loss' in history:
                final_loss = history['train_loss'][-1] if history['train_loss'] else 'N/A'
                logger.info(f"Final training loss: {final_loss}")

        except Exception as e:
            logger.error(f"Training failed for {arm_name}: {e}")
            import traceback
            traceback.print_exc()
            results[arm_name] = {
                **arm_data,
                'train_results': None,
                'success': False,
                'error': str(e)
            }

    return results


def run_inference(
    results: Dict[str, Dict[str, Any]],
    test_on_train: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    B3: Run inference on test sets for all trained models.

    Uses reconstructed_obj from training results (computed by train_cdi_model).
    Falls back to model loading if available.

    Args:
        results: Training results dict
        test_on_train: If True, also run inference on training data

    Returns:
        Updated results dict with reconstructions added
    """
    from ptycho import tf_helper as hh
    from ptycho.config.config import InferenceConfig, ModelConfig
    from ptycho.workflows.backend_selector import load_inference_bundle_with_backend
    from ptycho import loader
    from ptycho import nbutils

    logger.info("\n=== Running Inference ===")

    for arm_name, arm_data in results.items():
        if not arm_data.get('success', False):
            logger.warning(f"Skipping inference for {arm_name} (training failed)")
            continue

        logger.info(f"\n--- Inference for {arm_name} ---")

        config = arm_data['config']
        # Update params.cfg (CONFIG-001)
        update_legacy_dict(p.cfg, config)

        try:
            test_data = arm_data.get('test_data')
            if test_data is None:
                logger.warning(f"No test_data available for {arm_name}")
                results[arm_name]['reconstruction'] = None
                continue

            model_dir = Path(config.output_dir)
            infer_config = InferenceConfig(
                model=ModelConfig(N=config.model.N, gridsize=config.model.gridsize),
                model_path=model_dir,
                test_data_file=model_dir / "dummy.npz",
                n_groups=None,
                neighbor_count=config.neighbor_count,
                backend=config.backend,
            )
            model, params_dict = load_inference_bundle_with_backend(model_dir, infer_config)

            raw_count = len(test_data.diff3d)
            if config.n_groups is None:
                group_count = raw_count
            else:
                group_count = min(raw_count, config.n_groups)

            grouped = test_data.generate_grouped_data(
                params_dict.get("N", config.model.N),
                K=config.neighbor_count,
                nsamples=group_count,
                gridsize=params_dict.get("gridsize", config.model.gridsize),
                enable_oversampling=False,
            )
            container = loader.load(lambda: grouped, test_data.probeGuess, which=None, create_split=False)

            obj_tensor_full, global_offsets = nbutils.reconstruct_image(
                container,
                diffraction_to_obj=model,
            )
            recon = hh.reassemble_position(obj_tensor_full, global_offsets, M=20)
            recon_amp = np.abs(recon)
            recon_phase = np.angle(recon)

            results[arm_name]['reconstruction'] = {
                'amplitude': recon_amp,
                'phase': recon_phase,
                'patches': obj_tensor_full
            }
            logger.info(f"Reconstruction shape: {recon_amp.shape}")

        except Exception as e:
            logger.error(f"Inference failed for {arm_name}: {e}")
            import traceback
            traceback.print_exc()
            results[arm_name]['reconstruction'] = None

    return results


# =============================================================================
# Phase C: Visualization & Delivery
# =============================================================================

def generate_six_panel_figure(
    results: Dict[str, Dict[str, Any]],
    output_path: Path,
    crop_size: int = 128
) -> None:
    """
    C1: Generate the 6-panel publication figure.

    Layout:
        Row 1: High Dose - Diffraction (log), MAE Recon, NLL Recon
        Row 2: Low Dose - Diffraction (log), MAE Recon, NLL Recon
    """
    logger.info("\n=== Generating 6-Panel Figure ===")

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)

    # Get reconstructions and diffraction patterns
    high_mae = results.get('high_mae', {})
    high_nll = results.get('high_nll', {})
    low_mae = results.get('low_mae', {})
    low_nll = results.get('low_nll', {})

    def get_diffraction_sample(arm_data):
        """Get a representative diffraction pattern (log scale)."""
        # Try nongrid mode path first (test_data.diff3d)
        if arm_data and 'test_data' in arm_data:
            diff = arm_data['test_data'].diff3d[0]  # First pattern
            return np.log10(diff + 1)  # Log scale
        # Try grid mode path (test_container.X or train_results.test_container.X)
        if arm_data:
            test_container = arm_data.get('test_container')
            if test_container is None and 'train_results' in arm_data:
                test_container = arm_data['train_results'].get('test_container')
            if test_container is not None:
                # X is diffraction data, shape (B, N, N, C)
                diff = np.array(test_container.X[0])  # First pattern
                if diff.ndim == 3:
                    diff = diff[:, :, 0]  # Take first channel
                return np.log10(diff + 1)  # Log scale
        return None

    def get_reconstruction(arm_data):
        """Get reconstruction amplitude - try full recon, then patches."""
        if arm_data and arm_data.get('reconstruction'):
            recon = arm_data['reconstruction']
            # Try full reconstruction first
            if recon.get('amplitude') is not None:
                return recon['amplitude']
            # Fall back to showing a sample patch
            if recon.get('patches') is not None:
                patches = recon['patches']
                # Show mean of first few patches
                if len(patches) > 0:
                    sample = np.abs(patches[0])  # First patch
                    if sample.ndim > 2:
                        sample = np.squeeze(sample)
                    return sample
        # Try to get from train_results directly
        if arm_data and arm_data.get('train_results'):
            recon_obj = arm_data['train_results'].get('reconstructed_obj')
            if recon_obj is not None and len(recon_obj) > 0:
                sample = np.abs(recon_obj[0])
                if sample.ndim > 2:
                    sample = np.squeeze(sample)
                return sample
        return None

    def crop_center(img, size):
        """Crop to central region."""
        if img is None:
            return None
        h, w = img.shape[:2]
        start_h = max(0, (h - size) // 2)
        start_w = max(0, (w - size) // 2)
        return img[start_h:start_h+size, start_w:start_w+size]

    # Row 1: High Dose
    panels = [
        (0, 0, get_diffraction_sample(high_nll), 'High Dose\nDiffraction (log)', 'viridis'),
        (0, 1, crop_center(get_reconstruction(high_mae), crop_size), 'High Dose\nMAE Reconstruction', 'gray'),
        (0, 2, crop_center(get_reconstruction(high_nll), crop_size), 'High Dose\nNLL Reconstruction', 'gray'),
        # Row 2: Low Dose
        (1, 0, get_diffraction_sample(low_nll), 'Low Dose\nDiffraction (log)', 'viridis'),
        (1, 1, crop_center(get_reconstruction(low_mae), crop_size), 'Low Dose\nMAE Reconstruction', 'gray'),
        (1, 2, crop_center(get_reconstruction(low_nll), crop_size), 'Low Dose\nNLL Reconstruction', 'gray'),
    ]

    for row, col, data, title, cmap in panels:
        ax = fig.add_subplot(gs[row, col])

        if data is not None:
            # Squeeze extra dimensions
            data = np.squeeze(data)
            im = ax.imshow(data, cmap=cmap)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    # Add overall title
    fig.suptitle(
        'Dose Response Study: PtychoPINN Reconstruction Quality\n'
        'Comparing Poisson NLL vs MAE Loss under High/Low Photon Conditions',
        fontsize=14, fontweight='bold', y=0.98
    )

    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Figure saved to: {output_path}")


def save_training_history(
    results: Dict[str, Dict[str, Any]],
    output_path: Path
) -> None:
    """Save training history to JSON for convergence analysis."""
    import json

    history_data = {}
    for arm_name, arm_data in results.items():
        if arm_data.get('train_results') and 'history' in arm_data['train_results']:
            history = arm_data['train_results']['history']
            # Convert numpy arrays to lists for JSON serialization
            history_data[arm_name] = {
                k: [float(v) for v in vals] if hasattr(vals, '__iter__') else float(vals)
                for k, vals in history.items()
            }

    with open(output_path, 'w') as f:
        json.dump(history_data, f, indent=2)

    logger.info(f"Training history saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Synthetic Dose Response & Loss Comparison Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--output-dir', type=Path, default=Path('tmp/dose_study'),
        help='Base output directory for study artifacts (default: tmp/dose_study)'
    )
    parser.add_argument(
        '--nepochs', type=int, default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--n-train', type=int, default=2000,
        help='Number of training images (default: 2000)'
    )
    parser.add_argument(
        '--n-test', type=int, default=128,
        help='Number of test images (default: 128)'
    )
    parser.add_argument(
        '--N', type=int, default=64,
        help='Patch size (default: 64)'
    )
    parser.add_argument(
        '--gridsize', type=int, default=2,
        help='Grid size for grouping (default: 2)'
    )
    parser.add_argument(
        '--skip-training', action='store_true',
        help='Skip training and use existing models'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--grid-mode', action='store_true',
        help='Use legacy grid-based simulation (notebook-compatible). '
             'Matches notebooks/dose_dependence.ipynb behavior with N=128, '
             'gridsize=2, fixed grid extraction. See ALIGN-DOSE-STUDY-GRID-001.'
    )
    parser.add_argument(
        '--nimgs-train', type=int, default=2,
        help='Number of training images for grid mode (default: 2)'
    )
    parser.add_argument(
        '--nimgs-test', type=int, default=2,
        help='Number of test images for grid mode (default: 2)'
    )

    return parser.parse_args()


def main():
    """Main execution flow for the dose response study."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("STUDY-SYNTH-DOSE-COMPARISON-001: Dose Response Study")
    if args.grid_mode:
        logger.info("MODE: Grid-based simulation (ALIGN-DOSE-STUDY-GRID-001)")
    else:
        logger.info("MODE: Nongrid random coordinate simulation")
    logger.info("=" * 60)

    # Setup output directories
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path('plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports')
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Phase A: Setup and Data Generation
    logger.info("\n" + "=" * 40)
    logger.info("PHASE A: Orchestration & Data Fabric")
    logger.info("=" * 40)

    if args.grid_mode:
        # Grid mode: Use legacy mk_simdata() path
        # Simplified: using gridsize=1 to avoid model architecture complexity
        logger.info("Grid mode: Using N=64, gridsize=1, size=196 (simplified from notebook)")

        # Generate probe for grid mode (N=64)
        from ptycho import probe as probe_module
        p.cfg['default_probe_scale'] = 4.0
        p.cfg['N'] = 64  # Grid mode uses N=64 (reduced from 128 to avoid OOM)
        probeGuess = probe_module.get_default_probe(N=64, fmt='np').astype(np.complex64)

        # Simulate datasets using grid mode
        datasets = simulate_datasets_grid_mode(
            probeGuess=probeGuess,
            base_output_dir=output_dir,
            nepochs=args.nepochs,
            nimgs_train=args.nimgs_train,
            nimgs_test=args.nimgs_test
        )

        # Grid mode sanity check
        sanity_check_datasets_grid_mode(datasets)

    else:
        # Nongrid mode: Original behavior
        # A0: Create initial config and verify params setup
        initial_config = make_config(
            nphotons=1e9,
            loss_type='nll',
            output_subdir='initial',
            base_output_dir=output_dir,
            N=args.N,
            gridsize=args.gridsize,
            n_images=args.n_train,
            nepochs=args.nepochs
        )
        verify_params_setup(initial_config)

        # A1: Generate ground truth
        objectGuess, probeGuess = generate_ground_truth(N=args.N, object_size=128)

        # A2: Simulate all datasets
        datasets = simulate_datasets(
            objectGuess=objectGuess,
            probeGuess=probeGuess,
            base_output_dir=output_dir,
            n_train=args.n_train,
            n_test=args.n_test,
            N=args.N,
            gridsize=args.gridsize,
            nepochs=args.nepochs
        )

        # A3: Sanity check
        sanity_check_datasets(datasets)

    # Phase B: Training & Inference
    if not args.skip_training:
        logger.info("\n" + "=" * 40)
        logger.info("PHASE B: Training & Inference Loop")
        logger.info("=" * 40)

        if args.grid_mode:
            # Grid mode training (uses containers directly)
            results = train_all_arms_grid_mode(datasets)
            # Inference already done in train_all_arms_grid_mode
        else:
            # Nongrid mode training
            results = train_all_arms(datasets)
            # B3: Run inference
            results = run_inference(results)
    else:
        logger.info("\n[SKIP] Training skipped (--skip-training flag)")
        results = datasets

    # Phase C: Visualization & Delivery
    logger.info("\n" + "=" * 40)
    logger.info("PHASE C: Visualization & Delivery")
    logger.info("=" * 40)

    # C1: Generate 6-panel figure
    figure_path = reports_dir / 'dose_comparison.png'
    generate_six_panel_figure(results, figure_path)

    # Save training history
    if not args.skip_training:
        history_path = reports_dir / 'training_history.json'
        save_training_history(results, history_path)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("STUDY COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Figure saved to: {figure_path}")

    # Report success/failure for each arm
    for arm_name, arm_data in results.items():
        status = "SUCCESS" if arm_data.get('success', False) else "PENDING/FAILED"
        logger.info(f"  {arm_name}: {status}")

    logger.info("\nTo view results: open %s", figure_path)


if __name__ == '__main__':
    main()

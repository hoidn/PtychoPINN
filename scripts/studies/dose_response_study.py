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
from typing import Dict, Any, Optional, Tuple
from dataclasses import replace

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
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params as p

# PtychoPINN imports - simulation and training
from ptycho.nongrid_simulation import generate_simulated_data
from ptycho.loader import RawData
from ptycho import probe

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
    assert p.cfg.get('gridsize') == config.model.gridsize, f"gridsize mismatch"
    assert p.cfg.get('nphotons') == config.nphotons, f"nphotons mismatch"

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

    # Generate lines pattern
    obj_full = mk_lines_img(N=object_size, nlines=400)

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

    # Generate probe (requires params.cfg['default_probe_scale'] to be set)
    # Ensure default_probe_scale is set
    if p.cfg.get('default_probe_scale') is None:
        p.cfg['default_probe_scale'] = 4.0

    probeGuess = probe.get_default_probe(N=N, fmt='np').astype(np.complex64)

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
        # Enable oversampling for test data when gridsize > 1
        config = make_config(
            nphotons=arm_params['nphotons'],
            loss_type=arm_params['loss_type'],
            output_subdir=arm_name,
            base_output_dir=base_output_dir,
            N=N,
            gridsize=gridsize,
            n_images=n_train,
            nepochs=nepochs,
            enable_oversampling=True  # Required for gridsize > 1 with small test sets
        )

        # For simulation, we need gridsize=1 (from_simulation limitation)
        # Create a simulation-specific config with gridsize=1
        sim_model_config = ModelConfig(N=N, gridsize=1, model_type='pinn')
        sim_config_train = TrainingConfig(
            model=sim_model_config,
            n_groups=n_train,
            nphotons=arm_params['nphotons'],
            nll_weight=config.nll_weight,
            mae_weight=config.mae_weight,
            nepochs=1,  # Not used for simulation
            output_dir=config.output_dir
        )

        # Update params.cfg for simulation (CONFIG-001)
        update_legacy_dict(p.cfg, sim_config_train)

        # Generate training data with seed=1 for consistent trajectories
        np.random.seed(1)  # Same seed for all training sets
        train_data = generate_simulated_data(
            config=sim_config_train,
            objectGuess=objectGuess,
            probeGuess=probeGuess,
            buffer=buffer,
            return_patches=False
        )

        # Generate test data with seed=2
        sim_config_test = replace(sim_config_train, n_groups=n_test)
        np.random.seed(2)  # Different seed for test sets
        test_data = generate_simulated_data(
            config=sim_config_test,
            objectGuess=objectGuess,
            probeGuess=probeGuess,
            buffer=buffer,
            return_patches=False
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
    logger.info(f"Expected ratio: ~1e5 (from nphotons ratio 1e9/1e4)")

    # Relaxed check: ratio should be > 1e3 (accounting for sqrt in amplitudes)
    if ratio < 1e3:
        logger.warning(f"Intensity ratio {ratio:.2e} is lower than expected!")
    else:
        logger.info("=== Intensity scaling verification PASSED ===")


# =============================================================================
# Phase B: Training & Inference Loop
# =============================================================================

def train_all_arms(
    datasets: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    B2: Execute training for all experimental arms.

    Returns:
        Updated datasets dict with training results added
    """
    # Import training components (after params.cfg is set)
    from ptycho.workflows.components import train_cdi_model
    from ptycho.model_manager import ModelManager

    logger.info("\n=== Training All Arms ===")

    results = {}
    for arm_name, arm_data in datasets.items():
        logger.info(f"\n--- Training {arm_name} ---")

        config = arm_data['config']
        train_data = arm_data['train_data']
        test_data = arm_data['test_data']

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
            train_results = train_cdi_model(train_data, test_data, config)

            # Save the model weights to wts.h5.zip for later inference
            model_instance = train_results.get('model_instance')
            if model_instance is not None:
                # Get intensity_scale from params
                intensity_scale = p.cfg.get('intensity_scale', 1.0)

                # Import model module here (after params are fully configured with probe)
                from ptycho.model import create_model_with_gridsize, IntensityScaler
                from ptycho.custom_layers import (
                    FlatToChannelLayer, ScaleLayer, InvScaleLayer,
                    ActivationLayer, SquareLayer, TrimReconstructionLayer,
                    PadAndDiffractLayer
                )

                # Create diffraction_to_obj model from the autoencoder
                # The autoencoder has 3 outputs, diffraction_to_obj has 1 (trimmed_obj)
                _, diffraction_to_obj = create_model_with_gridsize(
                    config.model.gridsize, config.model.N
                )
                # Copy weights from trained autoencoder to diffraction_to_obj
                # Both models share the same encoder weights
                diffraction_to_obj.set_weights(model_instance.get_weights()[:len(diffraction_to_obj.get_weights())])

                models_to_save = {
                    'autoencoder': model_instance,
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

    logger.info("\n=== Running Inference ===")

    for arm_name, arm_data in results.items():
        if not arm_data.get('success', False):
            logger.warning(f"Skipping inference for {arm_name} (training failed)")
            continue

        logger.info(f"\n--- Inference for {arm_name} ---")

        config = arm_data['config']
        train_results = arm_data.get('train_results', {})

        # Update params.cfg (CONFIG-001)
        update_legacy_dict(p.cfg, config)

        try:
            # Use reconstructed_obj from training results (already computed)
            recon_obj = train_results.get('reconstructed_obj')
            test_container = train_results.get('test_container')

            if recon_obj is not None and test_container is not None:
                logger.info(f"Using reconstructed_obj from training, shape: {recon_obj.shape}")

                # Stitch reconstruction
                try:
                    recon = hh.reassemble_position(
                        recon_obj,
                        test_container.global_offsets,
                        M=20
                    )
                    recon_amp = np.abs(recon)
                    recon_phase = np.angle(recon)

                    results[arm_name]['reconstruction'] = {
                        'amplitude': recon_amp,
                        'phase': recon_phase,
                        'patches': recon_obj
                    }
                    logger.info(f"Reconstruction shape: {recon_amp.shape}")
                except Exception as stitch_e:
                    logger.warning(f"Stitching failed for {arm_name}: {stitch_e}")
                    # Store individual patches as fallback
                    # Take mean amplitude of all patches as a simple visualization
                    mean_patch_amp = np.mean(np.abs(recon_obj), axis=0)
                    if mean_patch_amp.ndim > 2:
                        mean_patch_amp = np.squeeze(mean_patch_amp)
                    results[arm_name]['reconstruction'] = {
                        'amplitude': mean_patch_amp,
                        'phase': np.mean(np.angle(recon_obj), axis=0).squeeze(),
                        'patches': recon_obj
                    }
                    logger.info(f"Using mean patch visualization, shape: {mean_patch_amp.shape}")
            else:
                logger.warning(f"No reconstructed_obj for {arm_name}")
                results[arm_name]['reconstruction'] = None

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
        if arm_data and 'test_data' in arm_data:
            diff = arm_data['test_data'].diff3d[0]  # First pattern
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

    return parser.parse_args()


def main():
    """Main execution flow for the dose response study."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("STUDY-SYNTH-DOSE-COMPARISON-001: Dose Response Study")
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

        # B2: Train all arms
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

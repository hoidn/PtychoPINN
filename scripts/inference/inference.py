#!/usr/bin/env python
# coding: utf-8
# MAYBE only generate the comparison plot when ground truth object is provided
# MAYBE save output to npz file, not just image

"""
Inference script for ptychography reconstruction.

This script loads a trained model and test data, performs inference,
and saves the reconstructed image comparison and optionally a probe visualization.

Usage:
    python inference_script.py --model_prefix <model_prefix> --test_data <test_data_file> [--output_path <output_path>]
                               [--visualize_probe] [--K <K>] [--nsamples <nsamples>]

Arguments:
    --model_prefix: Path prefix for the saved model and its configuration
    --test_data: Path to the .npz file containing test data
    --output_path: Path prefix for saving output files and images (default: './')
    --visualize_probe: Flag to generate and save probe visualization
    --K: Number of nearest neighbors for grouped data generation (default: 7)
    --nsamples: Number of samples for grouped data generation (default: 1)
"""

from __future__ import annotations

from typing import Optional, Tuple
import argparse
import logging
import os
import sys
import time
import math
import json
import signal
import warnings
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptycho.raw_data import RawData
from ptycho.workflows.config_cli import load_data
from ptycho.workflows.backend_selector import load_inference_bundle_with_backend
from ptycho.config import resolve_inference_config
from ptycho.config.config import InferenceConfig, load_yaml_config
from ptycho.config.legacy_state import scoped_legacy_params

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler('inference.log')
                    ])
logger = logging.getLogger(__name__)

# Redirect print statements to logger
print = logger.info

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    print(f"Received signal {signum}. Initiating graceful shutdown...")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ptychography Inference Script")
    parser.add_argument("--model_path", type=str, default=argparse.SUPPRESS,
                       help="Path to the saved model")
    parser.add_argument("--test_data", type=str, default=argparse.SUPPRESS,
                       help="Path to the test data file")
    parser.add_argument("--config", type=str, required=False, default=None,
                       help="Optional path to YAML configuration file to override defaults")
    parser.add_argument("--output_dir", type=str, default=argparse.SUPPRESS,
                       help="Directory for saving output files and images")
    parser.add_argument("--debug", action="store_true", default=argparse.SUPPRESS,
                       help="Enable debug mode")
    parser.add_argument("--comparison_plot", action="store_true",
                       help="Generate original comparison plot (only if ground truth is available)")
    parser.add_argument("--inference_groups", type=int, required=False,
                       default=argparse.SUPPRESS,
                       help="Number of groups to process.")
    parser.add_argument("--n_groups", type=int, required=False,
                       default=argparse.SUPPRESS,
                       help="DEPRECATED: Use --inference-groups instead.")
    parser.add_argument("--n_images", type=int, required=False,
                       default=argparse.SUPPRESS,
                       help="DEPRECATED: Use --inference-groups instead. Number of images/groups to process. Interpretation depends on gridsize: "
                            "gridsize=1 means individual images, gridsize>1 means number of groups")
    parser.add_argument("--inference_raw_selection", type=int, required=False,
                       default=argparse.SUPPRESS,
                       help="Number of images to subsample from test data (independent control). "
                            "When provided, controls data selection separately from grouping.")
    parser.add_argument("--n_subsample", type=int, required=False,
                       default=argparse.SUPPRESS,
                       help="DEPRECATED: Use --inference-raw-selection instead.")
    parser.add_argument("--subsample_seed", type=int, required=False,
                       default=argparse.SUPPRESS,
                       help="Random seed for reproducible subsampling")
    parser.add_argument("--phase_vmin", type=float, required=False, default=None,
                       help="Minimum value for phase color scale (default: auto)")
    parser.add_argument("--phase_vmax", type=float, required=False, default=None,
                       help="Maximum value for phase color scale (default: auto)")
    parser.add_argument("--debug_dump", nargs='?', const='__AUTO__', default=None,
                       help="Directory to store inference debug artifacts (patch grid, offsets, stats). "
                            "Defaults to <output_dir>/debug_dump when invoked without a value.")
    parser.add_argument(
        "--patch-weighting",
        choices=["uniform", "probe"],
        default="uniform",
        help=(
            "PyTorch stitching weight. 'probe' selects the strict-load mmap "
            "barycentric workflow; 'uniform' preserves legacy stitching."
        ),
    )
    parser.add_argument(
        "--varpro-scaling",
        action="store_true",
        help="Apply VarPro scaling during mmap barycentric reconstruction.",
    )
    parser.add_argument(
        "--groups-per-center",
        type=int,
        default=1,
        help=(
            "Runtime coordinate groups per eligible center for PyTorch mmap "
            "barycentric reconstruction (default: 1)."
        ),
    )
    # Backend selection (POLICY-001: PyTorch mandatory, CONFIG-001: update_legacy_dict required)
    parser.add_argument("--backend", type=str, choices=['tensorflow', 'pytorch'],
                       default=argparse.SUPPRESS,
                       help="Backend to use for inference: 'tensorflow' (default) or 'pytorch'. "
                            "PyTorch backend requires torch>=2.2 (POLICY-001). "
                            "Both backends handle params.cfg restoration via CONFIG-001.")

    # PyTorch-only execution flags (see docs/workflows/pytorch.md §12)
    parser.add_argument("--torch-accelerator", type=str,
                       choices=['auto', 'cpu', 'cuda', 'gpu', 'mps', 'tpu'],
                       default='cuda',
                       help="PyTorch accelerator for inference (only applies when --backend pytorch). "
                            "Options: 'cuda' (default GPU baseline per POLICY-001), 'auto' (auto-detect with CUDA preference), "
                            "'cpu' (fallback), 'gpu', 'mps', 'tpu'. "
                            "Override with '--torch-accelerator cpu' for CPU-only runs. "
                            "See docs/workflows/pytorch.md §12 for details.")
    parser.add_argument("--torch-num-workers", type=int, default=0,
                       help="Number of dataloader worker processes for PyTorch inference (default: 0). "
                            "Set to 0 for main process only (CPU-safe). "
                            "Only applies when --backend pytorch.")
    parser.add_argument("--torch-inference-batch-size", type=int, default=None,
                       help="Batch size for PyTorch inference (default: None, uses model default). "
                            "Only applies when --backend pytorch.")
    return parser.parse_args()

def interpret_sampling_parameters(
    config: InferenceConfig,
    *,
    gridsize: int,
) -> tuple:
    """
    Interpret sampling parameters for inference based on gridsize and user input.
    
    This function determines the actual values for n_subsample and n_groups based on:
    1. If n_subsample is provided: use it for subsampling, n_groups for grouping
    2. Otherwise: use n_groups for both legacy-compatible sampling controls
    
    Args:
        config: Inference configuration with sampling parameters.
        gridsize: Authoritative grouping geometry restored from the archive.
        
    Returns:
        tuple: (n_subsample, n_groups, interpretation_message)
    """
    if type(gridsize) is not int or gridsize <= 0:
        raise ValueError(
            f"archive gridsize must be a positive integer, got {gridsize!r}"
        )
    
    # Case 1: Independent control with inference_raw_selection
    if config.inference_raw_selection is not None:
        n_subsample = config.inference_raw_selection
        n_groups = config.inference_groups
        
        if gridsize == 1:
            if n_groups is None:
                n_groups = n_subsample
            message = (f"Independent sampling control: subsampling {n_subsample} images, "
                      f"using {n_groups} for inference")
        else:
            if n_groups is None:
                message = (
                    "Independent sampling control: subsampling "
                    f"{n_subsample} images and using all available groups"
                )
            else:
                total_from_groups = n_groups * gridsize * gridsize
                message = (
                    "Independent sampling control: subsampling "
                    f"{n_subsample} images, creating {n_groups} groups "
                    f"(approx {total_from_groups} patterns from groups)"
                )
        
        return n_subsample, n_groups, message
    
    # Case 2: Canonical grouping controls both
    else:
        if config.inference_groups is not None:
            if gridsize == 1:
                n_subsample = config.inference_groups
                n_groups = config.inference_groups
                message = f"Using {n_groups} individual images (gridsize=1)"
            else:
                n_subsample = None  # Use full dataset for subsampling
                n_groups = config.inference_groups
                total_patterns = n_groups * gridsize * gridsize
                message = (f"Using {n_groups} groups "
                           f"(gridsize={gridsize}, approx {total_patterns} patterns)")
        else:
            n_subsample = None
            n_groups = None
            message = "Using full dataset for inference"
        
        return n_subsample, n_groups, message

def setup_inference_configuration(args: argparse.Namespace, yaml_path: Optional[str]) -> InferenceConfig:
    """Resolve YAML and explicitly supplied CLI inference configuration."""
    yaml_data = {}
    if yaml_path:
        print(f"Loading configuration from YAML: {yaml_path}")
        yaml_data = load_yaml_config(Path(yaml_path))

    cli_destinations = {
        "model_path": "model_path",
        "test_data": "test_data_file",
        "output_dir": "output_dir",
        "debug": "debug",
        "inference_groups": "inference_groups",
        "n_images": "n_images",
        "inference_raw_selection": "inference_raw_selection",
        "subsample_seed": "subsample_seed",
        "backend": "backend",
    }
    cli_patch = {
        config_name: getattr(args, argument_name)
        for argument_name, config_name in cli_destinations.items()
        if hasattr(args, argument_name)
    }
    if getattr(args, "n_groups", None) is not None:
        warnings.warn(
            "--n_groups is deprecated; use --inference_groups",
            DeprecationWarning,
            stacklevel=2,
        )
        if "inference_groups" in cli_patch and cli_patch["inference_groups"] != args.training_groups:
            raise ValueError(
                f"--n_groups conflicts with explicit --inference_groups "
                f"({args.training_groups!r} vs {cli_patch['inference_groups']!r})"
            )
        cli_patch["inference_groups"] = args.training_groups
    if getattr(args, "n_subsample", None) is not None:
        warnings.warn(
            "--n_subsample is deprecated; use --inference_raw_selection",
            DeprecationWarning,
            stacklevel=2,
        )
        if (
            "inference_raw_selection" in cli_patch
            and cli_patch["inference_raw_selection"] != args.train_raw_selection
        ):
            raise ValueError(
                f"--n_subsample conflicts with explicit --inference_raw_selection "
                f"({args.train_raw_selection!r} vs {cli_patch['inference_raw_selection']!r})"
            )
        cli_patch["inference_raw_selection"] = args.train_raw_selection

    inference_config = resolve_inference_config(yaml_data, cli_patch)
    print(f"Final inference config - gridsize: {inference_config.model.gridsize}")
    return inference_config


def _dump_tf_inference_debug_artifacts(
    debug_path: Path,
    patch_tensor,
    global_offsets,
    canvas,
    patch_limit: int = 16,
):
    """
    Persist patch-level diagnostics for TensorFlow inference.

    Args:
        debug_path: Directory where artifacts will be written.
        patch_tensor: Complex numpy array of predicted patches.
        global_offsets: Offsets returned by reconstruct_image().
        canvas: Complex numpy array of the stitched object.
        patch_limit: Number of flattened patches to visualize.
    """
    debug_path = Path(debug_path)
    debug_path.mkdir(parents=True, exist_ok=True)

    patch_complex = np.asarray(patch_tensor)
    if patch_complex.ndim >= 4 and patch_complex.shape[-1] == 1:
        patch_complex = np.squeeze(patch_complex, axis=-1)
    patch_amp = np.abs(patch_complex)
    flat_patches = patch_amp.reshape(-1, patch_amp.shape[-2], patch_amp.shape[-1])
    limit = max(0, min(patch_limit, flat_patches.shape[0]))

    if limit > 0:
        cols = 4
        rows = math.ceil(limit / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        axes = np.atleast_1d(axes).ravel()
        for idx, ax in enumerate(axes):
            ax.axis('off')
            if idx >= limit:
                continue
            ax.imshow(flat_patches[idx], cmap='magma')
            ax.set_title(f"Patch {idx}", fontsize=8)
        fig.tight_layout()
        fig.savefig(debug_path / "pred_patches_amp_grid.png", dpi=200)
        plt.close(fig)

    offsets_arr = np.asarray(global_offsets)
    if offsets_arr.size == 0:
        offsets_vec = np.zeros((0, 2), dtype=np.float32)
    else:
        offsets_vec = offsets_arr.reshape(-1, offsets_arr.shape[-1])
        if offsets_vec.shape[-1] != 2:
            offsets_vec = offsets_arr.reshape(-1, 2)
    canvas_complex = np.squeeze(np.asarray(canvas))
    canvas_amp = np.abs(canvas_complex)

    centered_offsets = offsets_vec - np.mean(offsets_vec, axis=0, keepdims=True) if offsets_vec.size else offsets_vec
    offsets_payload = {
        "count": int(offsets_vec.shape[0]),
        "first_offsets_px": [
            {"dx": float(offsets_vec[i, 0]), "dy": float(offsets_vec[i, 1])}
            for i in range(min(limit, offsets_vec.shape[0]))
        ],
        "mean_dx": float(np.mean(offsets_vec[:, 0])) if offsets_vec.size else 0.0,
        "mean_dy": float(np.mean(offsets_vec[:, 1])) if offsets_vec.size else 0.0,
        "std_dx": float(np.std(offsets_vec[:, 0])) if offsets_vec.size else 0.0,
        "std_dy": float(np.std(offsets_vec[:, 1])) if offsets_vec.size else 0.0,
        "centered_std_dx": float(np.std(centered_offsets[:, 0])) if offsets_vec.size else 0.0,
        "centered_std_dy": float(np.std(centered_offsets[:, 1])) if offsets_vec.size else 0.0,
    }
    with open(debug_path / "offsets.json", "w", encoding="utf-8") as fp:
        json.dump(offsets_payload, fp, indent=2)

    patch_zero_mean = flat_patches - flat_patches.mean(axis=(-2, -1), keepdims=True)
    patch_variance = float(np.mean(patch_zero_mean ** 2)) if flat_patches.size else 0.0
    stats_payload = {
        "patch_amplitude": {
            "mean": float(np.mean(flat_patches)) if flat_patches.size else 0.0,
            "std": float(np.std(flat_patches)) if flat_patches.size else 0.0,
            "min": float(np.min(flat_patches)) if flat_patches.size else 0.0,
            "max": float(np.max(flat_patches)) if flat_patches.size else 0.0,
            "var_zero_mean": patch_variance,
        },
        "canvas_amplitude": {
            "mean": float(np.mean(canvas_amp)) if canvas_amp.size else 0.0,
            "std": float(np.std(canvas_amp)) if canvas_amp.size else 0.0,
            "min": float(np.min(canvas_amp)) if canvas_amp.size else 0.0,
            "max": float(np.max(canvas_amp)) if canvas_amp.size else 0.0,
        },
    }
    with open(debug_path / "stats.json", "w", encoding="utf-8") as fp:
        json.dump(stats_payload, fp, indent=2)

    canvas_payload = {
        "patch_size": int(flat_patches.shape[-1]) if flat_patches.size else 0,
        "canvas_size": int(canvas_amp.shape[-1]) if canvas_amp.size else 0,
        "num_patch_slots": int(flat_patches.shape[0]),
        "max_abs_dx": float(np.max(np.abs(offsets_vec[:, 0]))) if offsets_vec.size else 0.0,
        "max_abs_dy": float(np.max(np.abs(offsets_vec[:, 1]))) if offsets_vec.size else 0.0,
    }
    with open(debug_path / "canvas.json", "w", encoding="utf-8") as fp:
        json.dump(canvas_payload, fp, indent=2)


def extract_ground_truth(raw_data: "RawData") -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract ground truth amplitude/phase from RawData if available and valid.

    Args:
        raw_data: RawData instance with potential ground truth in objectGuess.

    Returns:
        (amplitude, phase) tuple if valid ground truth exists, else None.

    Notes:
        - Returns None if objectGuess is missing, all zeros, or uniform.
        - Uses crop_to_non_uniform_region_with_buffer for processing.
    """
    from ptycho.nbutils import crop_to_non_uniform_region_with_buffer

    if not hasattr(raw_data, 'objectGuess') or raw_data.objectGuess is None:
        return None
    if np.allclose(raw_data.objectGuess, 0, atol=1e-10):
        return None
    obj_complex = raw_data.objectGuess
    if (np.allclose(obj_complex.real, obj_complex.real.flat[0], atol=1e-10) and
            np.allclose(obj_complex.imag, obj_complex.imag.flat[0], atol=1e-10)):
        return None

    epie_phase = crop_to_non_uniform_region_with_buffer(np.angle(obj_complex), buffer=-20)
    epie_amplitude = crop_to_non_uniform_region_with_buffer(np.abs(obj_complex), buffer=-20)
    return (epie_amplitude, epie_phase)


def _run_tf_inference_and_reconstruct(
    model: tf.keras.Model,
    raw_data: "RawData",
    config: dict,
    K: int = 4,
    nsamples: Optional[int] = None,
    quiet: bool = False,
    debug_dump_dir: Optional[Path] = None,
    debug_patch_limit: int = 16,
    seed: int = 45,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core TensorFlow inference helper for programmatic use.

    Mirrors PyTorch `_run_inference_and_reconstruct()` signature for API parity.
    See docs/specs/spec-ptycho-interfaces.md for contract details.

    Args:
        model: Loaded TensorFlow model (from load_inference_bundle_with_backend).
        raw_data: RawData instance with test data.
        config: Dict with 'N', 'gridsize' keys (from model bundle).
        K: Number of nearest neighbors (default: 4).
        nsamples: Number of samples; if None, uses all available.
        quiet: Suppress progress output.
        debug_dump_dir: Optional directory for debug artifacts.
        debug_patch_limit: Patches to visualize in debug mode.
        seed: Random seed for reproducibility (default: 45).

    Returns:
        Tuple of (amplitude, phase) as numpy arrays.

    Raises:
        ValueError: If there's an error during inference.

    Notes:
        - Expects params.cfg to be populated via CONFIG-001 before call.
        - Ground truth not returned (use extract_ground_truth separately).
    """
    from ptycho.nbutils import reconstruct_image
    from ptycho import loader
    from ptycho.tf_helper import reassemble_position

    try:
        # Set random seeds for reproducibility
        tf.random.set_seed(seed)
        np.random.seed(seed)

        from ptycho import debug_parity
        debug_parity.log_array_stats("tf.diffraction_raw", raw_data.diff3d)
        debug_parity.log_array_stats("tf.probe_raw", raw_data.probeGuess)

        # Generate grouped data
        if not quiet:
            logger.info(f"DEBUG: Using gridsize={config.get('gridsize', 'NOT_SET')} for data generation")
        test_dataset = raw_data.generate_grouped_data(
            config['N'], K=K, nsamples=nsamples, gridsize=config.get('gridsize', 1)
        )

        # Debug: check shapes
        if not quiet:
            if 'diffraction' in test_dataset:
                logger.info(f"DEBUG: Generated diffraction data shape: {test_dataset['diffraction'].shape}")
            if 'Y' in test_dataset and test_dataset['Y'] is not None:
                logger.info(f"DEBUG: Generated Y data shape: {test_dataset['Y'].shape}")

        # Create PtychoDataContainer
        test_data_container = loader.load(
            lambda: test_dataset, raw_data.probeGuess, which=None, create_split=False
        )

        if not quiet:
            logger.info(
                f"DEBUG: PtychoDataContainer shapes - X (diffraction): {test_data_container.X.shape}, "
                f"Y: {test_data_container.Y.shape if test_data_container.Y is not None else 'None'}"
            )

        # Perform reconstruction
        start_time = time.time()
        obj_tensor_full, global_offsets = reconstruct_image(test_data_container, diffraction_to_obj=model)
        reconstruction_time = time.time() - start_time
        if not quiet:
            logger.info(f"Reconstruction completed in {reconstruction_time:.2f} seconds")

        debug_parity.log_offsets_stats("tf.offsets_global", global_offsets)

        # Process the reconstructed image
        obj_image = reassemble_position(obj_tensor_full, global_offsets, M=20)
        debug_parity.log_array_stats("tf.reassembly_output", obj_image)

        # Extract amplitude and phase
        reconstructed_amplitude = np.abs(obj_image)
        reconstructed_phase = np.angle(obj_image)

        if not quiet:
            logger.info(f"Reconstructed amplitude shape: {reconstructed_amplitude.shape}")
            logger.info(f"Reconstructed phase shape: {reconstructed_phase.shape}")

        # Debug artifact dump (conditional)
        if debug_dump_dir is not None:
            _dump_tf_inference_debug_artifacts(
                debug_dump_dir,
                patch_tensor=obj_tensor_full,
                global_offsets=global_offsets,
                canvas=obj_image,
                patch_limit=debug_patch_limit,
            )

        return reconstructed_amplitude, reconstructed_phase

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise ValueError(f"Error during inference: {str(e)}")


def perform_inference(model: tf.keras.Model, test_data: RawData, config: dict, K: int, nsamples: int,
                      debug_dump_dir: Path | None = None, debug_patch_limit: int = 16) -> tuple:
    """
    Perform inference using the loaded model and test data.

    .. deprecated::
        Use `_run_tf_inference_and_reconstruct()` for new code.
        This wrapper maintains backward compatibility with the 4-tuple return.

    Args:
        model (tf.keras.Model): The loaded TensorFlow model.
        test_data (RawData): The RawData object containing test data.
        config (dict): The model's configuration dictionary.
        K (int): Number of nearest neighbors for grouped data generation.
        nsamples (int): Number of samples for grouped data generation.
        debug_dump_dir: Optional directory for debug artifacts.
        debug_patch_limit: Number of patches to visualize in debug mode.

    Returns:
        tuple: (np.ndarray, np.ndarray, np.ndarray, np.ndarray) - Reconstructed amplitude,
               reconstructed phase, ePIE amplitude, and ePIE phase.

    Raises:
        ValueError: If there's an error during inference.
    """
    import warnings
    warnings.warn(
        "perform_inference is deprecated; use _run_tf_inference_and_reconstruct",
        DeprecationWarning,
        stacklevel=2
    )

    amp, phase = _run_tf_inference_and_reconstruct(
        model=model,
        raw_data=test_data,
        config=config,
        K=K,
        nsamples=nsamples,
        quiet=False,
        debug_dump_dir=debug_dump_dir,
        debug_patch_limit=debug_patch_limit,
        seed=45,
    )

    gt = extract_ground_truth(test_data)
    if gt:
        return amp, phase, gt[0], gt[1]
    return amp, phase, None, None

def save_comparison_plot(reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase, output_dir, phase_vmin=None, phase_vmax=None):
    """
    Save a comparison plot of reconstructed and ground truth images.
    
    Args:
        reconstructed_amplitude (np.ndarray): The reconstructed amplitude array
        reconstructed_phase (np.ndarray): The reconstructed phase array
        epie_amplitude (np.ndarray): The ground truth amplitude array or None
        epie_phase (np.ndarray): The ground truth phase array or None
        output_dir (str or Path): Directory to save the output images
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Squeeze any extra dimensions
        reconstructed_amplitude = np.squeeze(reconstructed_amplitude)
        reconstructed_phase = np.squeeze(reconstructed_phase)
        epie_amplitude = np.squeeze(epie_amplitude)
        epie_phase = np.squeeze(epie_phase)
        
        # Create the comparison figure with a smaller size
        fig, axs = plt.subplots(2, 2, figsize=(4, 4))
        
        # PtychoPINN phase
        im_pinn_phase = axs[0, 0].imshow(reconstructed_phase, cmap='gray', vmin=phase_vmin, vmax=phase_vmax)
        axs[0, 0].set_title('PtychoPINN Phase')
        fig.colorbar(im_pinn_phase, ax=axs[0, 0], fraction=0.046, pad=0.04)
        
        # ePIE phase
        im_epie_phase = axs[0, 1].imshow(epie_phase, cmap='gray', vmin=phase_vmin, vmax=phase_vmax)
        axs[0, 1].set_title('ePIE Phase')
        fig.colorbar(im_epie_phase, ax=axs[0, 1], fraction=0.046, pad=0.04)
        
        # PtychoPINN amplitude
        im_pinn_amp = axs[1, 0].imshow(reconstructed_amplitude, cmap='viridis')
        axs[1, 0].set_title('PtychoPINN Amplitude')
        fig.colorbar(im_pinn_amp, ax=axs[1, 0], fraction=0.046, pad=0.04)
        
        # ePIE amplitude
        im_epie_amp = axs[1, 1].imshow(epie_amplitude, cmap='viridis')
        axs[1, 1].set_title('ePIE Amplitude')
        fig.colorbar(im_epie_amp, ax=axs[1, 1], fraction=0.046, pad=0.04)
        
        # Remove axis ticks
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Save the figure
        comparison_path = os.path.join(output_dir, "comparison_plot.png")
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to: {comparison_path}")
        
    except Exception as e:
        print(f"Error saving comparison plot: {str(e)}")

def save_reconstruction_images(reconstructed_amplitude, reconstructed_phase, output_dir, phase_vmin=None, phase_vmax=None):
    """
    Save the reconstructed amplitude and phase as separate PNG files.
    
    Args:
        reconstructed_amplitude (np.ndarray): The reconstructed amplitude array
        reconstructed_phase (np.ndarray): The reconstructed phase array
        output_dir (str or Path): Directory to save the output images
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Squeeze any extra dimensions
        reconstructed_amplitude = np.squeeze(reconstructed_amplitude)
        reconstructed_phase = np.squeeze(reconstructed_phase)
        
        print(f"Amplitude array shape: {reconstructed_amplitude.shape}")
        print(f"Phase array shape: {reconstructed_phase.shape}")
        
        # Save amplitude image
        amplitude_path = os.path.join(output_dir, "reconstructed_amplitude.png")
        plt.figure(figsize=(8, 8))
        plt.imshow(reconstructed_amplitude, cmap='gray')
        plt.colorbar()
        plt.savefig(amplitude_path)
        plt.close()
        
        # Save phase image
        phase_path = os.path.join(output_dir, "reconstructed_phase.png")
        plt.figure(figsize=(8, 8))
        plt.imshow(reconstructed_phase, cmap='viridis', vmin=phase_vmin, vmax=phase_vmax)
        plt.colorbar()
        plt.savefig(phase_path)
        plt.close()
        
        print(f"Reconstructed amplitude saved to: {amplitude_path}")
        print(f"Reconstructed phase saved to: {phase_path}")
        
    except Exception as e:
        print(f"Error saving reconstruction images: {str(e)}")

def save_probe_visualization(test_data: RawData, output_path: str):
    """
    Generate and save the probe visualization.

    Args:
        test_data (RawData): The RawData object containing test data.
        output_path (str): Path to save the probe visualization.

    Raises:
        OSError: If there's an error creating the output directory or saving the image.
    """
    from ptycho.nbutils import probeshow
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Generate the probe visualization
        fig = probeshow(test_data.probeGuess, test_data)
        
        # Save the figure
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Probe visualization saved to: {output_path}")

    except OSError as e:
        raise OSError(f"Error saving probe visualization: {str(e)}")


def _resolve_unified_pytorch_runtime(execution_request):
    """Resolve the unified CLI's Torch request once and disclose notices."""
    if not execution_request.explicit_fields:
        print(
            "POLICY-001: No --torch-* execution flags provided. Backend will "
            "use GPU-first defaults (auto-detects CUDA if available, else CPU). "
            "CPU-only users should pass --torch-accelerator cpu."
        )
    from ptycho_torch.execution_request import resolve_runtime_execution_request

    runtime = resolve_runtime_execution_request(
        execution_request,
        mode="inference",
    )
    execution_config = runtime.config
    logger.debug("PyTorch execution runtime audit: %s", runtime.audit_dict())
    import warnings

    for notice in runtime.notices:
        warnings.warn(notice.message, notice.category, stacklevel=2)
    if execution_config.accelerator in ("cuda", "gpu"):
        device_str = "cuda"
    elif execution_config.accelerator == "mps":
        device_str = "mps"
    else:
        device_str = "cpu"
    return execution_config, device_str


@scoped_legacy_params
def main():
    """Main entry point for the ptychography inference script."""
    config = None
    try:
        raw_argv = tuple(sys.argv[1:])
        print("Starting ptychography inference script...")
        args = parse_arguments()
        config = setup_inference_configuration(args, args.config)
        execution_request = None
        if config.backend == 'pytorch':
            from ptycho_torch.cli.shared import build_execution_request_from_args

            execution_request = build_execution_request_from_args(
                args,
                mode='inference',
                explicit_options=raw_argv,
                lane='unified-inference',
            )
        debug_dump_dir = None
        if args.debug_dump is not None:
            debug_dump_dir = (
                Path(config.output_dir) / "debug_dump"
                if args.debug_dump == '__AUTO__'
                else Path(args.debug_dump)
            )

        if config.backend == "pytorch":
            from ptycho_torch.inference import (
                _resolve_reassembly_route,
                reconstruct_npz_barycentric,
            )

            patch_weighting = getattr(args, "patch_weighting", "uniform")
            varpro_scaling = bool(getattr(args, "varpro_scaling", False))
            reassembly_route = _resolve_reassembly_route(
                patch_weighting,
                varpro_scaling,
            )
            if reassembly_route == "barycentric":
                execution_config, device_str = (
                    _resolve_unified_pytorch_runtime(execution_request)
                )
                from ptycho_torch.config_params import (
                    InferenceConfig as PTInferenceConfig,
                )

                runtime_inference_knobs = PTInferenceConfig(
                    patch_weighting=patch_weighting,
                    varpro_scaling=varpro_scaling,
                )
                precision = getattr(execution_config, "precision", "32-true")
                if precision not in {"32-true", "16-mixed", "bf16-mixed"}:
                    precision = "32-true"
                result = reconstruct_npz_barycentric(
                    Path(config.model_path),
                    Path(config.test_data_file),
                    run_root=Path(config.output_dir),
                    groups_per_center=getattr(args, "groups_per_center", 1),
                    inference_config=runtime_inference_knobs,
                    device=device_str,
                    num_workers=int(execution_config.num_workers or 0),
                    inference_batch_size=(
                        execution_config.inference_batch_size
                    ),
                    precision=precision,
                    quiet=False,
                )
                save_reconstruction_images(
                    result.amplitude,
                    result.phase,
                    config.output_dir,
                    phase_vmin=args.phase_vmin,
                    phase_vmax=args.phase_vmax,
                )
                print("Inference process completed successfully.")
                sys.exit(0)

        # The selector bridges this validated bootstrap request before loading.
        # The backend loader may then restore authoritative archived params,
        # which take precedence per CONFIG-001.

        # Load model using backend selector
        print("Loading model...")
        model, archive_params = load_inference_bundle_with_backend(
            config.model_path,
            config,
        )
        archive_gridsize = archive_params.get("gridsize")
        n_subsample, n_groups, interpretation_message = (
            interpret_sampling_parameters(
                config,
                gridsize=archive_gridsize,
            )
        )
        print(interpretation_message)

        if (
            config.inference_raw_selection is not None
            and archive_gridsize > 1
            and n_groups is not None
        ):
            min_required = n_groups * archive_gridsize * archive_gridsize
            if n_subsample < min_required:
                print(
                    f"WARNING: n_subsample ({n_subsample}) may be too small "
                    f"to create {n_groups} groups of size "
                    f"{archive_gridsize}². Consider increasing n_subsample "
                    f"to at least {min_required}"
                )

        # For PyTorch backend, move model to execution device and set to eval mode
        if config.backend == 'pytorch':
            execution_config, device_str = _resolve_unified_pytorch_runtime(
                execution_request
            )
            # Move model to execution device and ensure eval mode (DEVICE-MISMATCH-001 fix)
            model.to(device_str)
            model.eval()
            print(f"PyTorch model moved to device: {device_str}")

        # Load test data with new independent sampling parameters
        print("Loading test data...")
        test_data = load_data(
            config.test_data_file,
            n_images=n_groups,
            n_subsample=n_subsample,
            subsample_seed=config.subsample_seed
        )

        # Determine number of samples for inference based on loaded data
        gridsize = archive_gridsize
        total_patterns = len(test_data.xcoords)
        
        if n_groups is not None:
            # User specified number of images/groups (already interpreted above)
            if gridsize == 1:
                nsamples = min(n_groups, total_patterns)
                print(f"Inference config: gridsize={gridsize}, using {nsamples} individual patterns")
            else:
                max_groups = total_patterns // (gridsize ** 2)
                nsamples = min(n_groups, max_groups)
                if nsamples == 0:
                    nsamples = 1  # Minimum of 1 group
                print(f"Inference config: gridsize={gridsize}, using {nsamples} groups (≈{nsamples * gridsize**2} total patterns)")
        else:
            # Default behavior: use full dataset
            if gridsize == 1:
                nsamples = total_patterns
                print(f"Inference config: gridsize={gridsize}, using all {nsamples} individual patterns")
            else:
                nsamples = total_patterns // (gridsize ** 2)
                if nsamples == 0:
                    nsamples = 1  # Minimum of 1 group
                print(f"Inference config: gridsize={gridsize}, using {nsamples} groups (≈{nsamples * gridsize**2} total patterns)")

        # Perform inference - branch based on backend
        print("Performing inference...")

        if config.backend == 'pytorch':
            # PyTorch inference path
            from ptycho_torch.inference import _run_inference_and_reconstruct

            # execution_config and device_str already resolved above after model loading
            # to ensure model.to(device) happens before inference

            print(f"PyTorch inference config: accelerator={execution_config.accelerator}, "
                  f"num_workers={execution_config.num_workers}, "
                  f"inference_batch_size={execution_config.inference_batch_size}")

            # Call PyTorch-native inference helper
            reconstructed_amplitude, reconstructed_phase = _run_inference_and_reconstruct(
                model,
                test_data,
                config,
                execution_config,
                device_str,
                quiet=False,
                debug_dump_dir=debug_dump_dir,
            )

            # PyTorch path doesn't return ground truth comparison data (not in scope for Phase R)
            epie_amplitude = None
            epie_phase = None

        else:
            # TensorFlow inference path (legacy)
            reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase = perform_inference(
                model,
                test_data,
                archive_params,
                K=config.neighbor_count,
                nsamples=nsamples,
                debug_dump_dir=debug_dump_dir,
            )

        # Save separate reconstruction images
        print("Saving reconstruction images...")
        save_reconstruction_images(reconstructed_amplitude, reconstructed_phase, config.output_dir,
                                  phase_vmin=args.phase_vmin, phase_vmax=args.phase_vmax)

        # Generate comparison plot if requested and ground truth is available
        if args.comparison_plot and epie_amplitude is not None and epie_phase is not None:
            print("Generating comparison plot...")
            save_comparison_plot(reconstructed_amplitude, reconstructed_phase,
                                epie_amplitude, epie_phase, config.output_dir,
                                phase_vmin=args.phase_vmin, phase_vmax=args.phase_vmax)
        elif args.comparison_plot:
            print("Skipping comparison plot generation - ground truth not available")

        print("Inference process completed successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"Script execution failed: {str(e)}")
        sys.exit(1)
    finally:
        print("Cleaning up resources...")
        # Only call TensorFlow cleanup if we used TensorFlow backend
        if config is not None and config.backend == 'tensorflow':
            tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()

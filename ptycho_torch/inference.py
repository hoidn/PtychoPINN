"""
PyTorch Inference Module for Ptychography Reconstruction

This module provides two inference modes:

1. **MLflow-based inference** (legacy mode):
   - Loads models from MLflow tracking server
   - Uses run_id to locate trained models
   - Suitable for production deployments with MLflow infrastructure

2. **Lightning checkpoint inference** (Phase E2.C2):
   - Loads PyTorch Lightning checkpoints directly
   - CLI interface mirroring TensorFlow inference workflow
   - Generates reconstruction visualizations (amplitude/phase PNGs)

Usage Examples:

  # Lightning checkpoint inference (Phase E2.C2)
  python -m ptycho_torch.inference \\
      --model_path training_outputs \\
      --test_data datasets/Run1084_recon3_postPC_shrunk_3.npz \\
      --output_dir inference_outputs \\
      --n_images 32 \\
      --device cpu

  # MLflow-based inference (legacy)
  python -m ptycho_torch.inference \\
      --run_id abc123 \\
      --infer_dir datasets/ \\
      --file_index 0

References:
  - Phase E2 plan: plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md §E2.C2
  - Test contract: tests/torch/test_integration_workflow_torch.py
  - Red phase evidence: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T213500Z/red_phase.md §2.3
"""

#Generic
import os
import time
from datetime import datetime
import argparse
import sys
from pathlib import Path

#ML libraries
import matplotlib.pyplot as plt
import numpy as np

# MLflow is only needed for legacy inference path
# Imported conditionally in load_and_predict() to avoid blocking new CLI path

def load_all_configs(config_path, file_index):
    """
    Helper functions that loads all relevant configs specifically for inference
    File index is updated based on argument from argparse
    """
    # Import MLflow-specific utilities (only needed for legacy path)
    from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig
    from ptycho_torch.utils import load_config_from_json, validate_and_process_config
    from ptycho_torch.config_params import update_existing_config

    print('Loading configs...')
    try:
        config_data = load_config_from_json(config_path)
        d_config_replace, m_config_replace, t_config_replace, i_config_replace, dgen_config_replace = validate_and_process_config(config_data)
    except Exception as e:
        print(f"Failed to open/validate config because of: {e}")

    data_config = DataConfig()
    if d_config_replace is not None:
        update_existing_config(data_config, d_config_replace)

    model_config = ModelConfig()
    if m_config_replace is not None:
        update_existing_config(model_config, m_config_replace)

    training_config = TrainingConfig()
    if t_config_replace is not None:
        update_existing_config(training_config, t_config_replace)

    inference_config = InferenceConfig()
    if i_config_replace is not None:
        update_existing_config(inference_config, i_config_replace)

    datagen_config = DatagenConfig()
    if dgen_config_replace is not None:
        update_existing_config(datagen_config, dgen_config_replace)

    return data_config, model_config, training_config, inference_config, datagen_config




#Loads model, training settings
def load_and_predict(run_id,
                     ptycho_files_dir,
                     relative_mlflow_path = 'mlruns',
                     config_override_path = None,
                     file_index = 0,
                     save_dir = "inference/output",
                     plot_name = "Test",
                     verbose = False):
    '''
    Given MLFlow run id, as well as ptycho file directory, will provide predictions
    Args:
        run_id: Unique MLflow run id generated upon training finishing
        ptycho_files_dir: File where all experimental ptychography files are saved
        relative_mlflow_path: directory where mlruns is bineg saved. Should be modifiable/configurable in train.py
    '''
    # Import MLflow dependencies (only needed for legacy path)
    try:
        import mlflow
        from ptycho_torch.utils import load_all_configs_from_mlflow
        from ptycho_torch.reassembly import reconstruct_image_barycentric
        from ptycho_torch.dataloader import PtychoDataset
        from ptycho_torch.config_params import update_existing_config
    except ImportError as e:
        raise RuntimeError(
            "MLflow-based inference requires 'mlflow' and related dependencies. "
            "Install via: pip install -e .[torch]\n"
            f"Import error: {e}"
        )

    #MLFlow tracking for model
    tracking_uri = f"file:{os.path.abspath(relative_mlflow_path)}"
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    #Loading config
    if not config_override_path:
        data_config, model_config, training_config, inference_config, datagen_config = load_all_configs_from_mlflow(run_id,
                                                                                         tracking_uri)
    else:
        data_config, model_config, training_config, inference_config, datagen_config = load_all_configs(config_override_path)

    # Manually overriding experiment number indexing
    i_config_replace = {}
    i_config_replace['experiment_number'] = file_index
    update_existing_config(inference_config, i_config_replace)

    #Loading model
    model_load_start = time.time()
    loaded_model = mlflow.pytorch.load_model(model_uri)
    loaded_model.to(training_config.device)
    loaded_model.training = True
    model_load_time = time.time() - model_load_start

    #Load data into dataset structure
    data_load_start = time.time()
    ptycho_dataset = PtychoDataset(ptycho_files_dir, model_config, data_config,
                                remake_map=True)

    data_load_time = time.time() - data_load_start

    #Reconstructing. Automatically puts dataset into dataloader, so don't worry about it
    if verbose:
        print(f"Data config: {data_config}")
        print(f"Model config: {model_config}")
        print(f"Inference config: {inference_config}")
    result, recon_dataset, assembly_stats = reconstruct_image_barycentric(loaded_model, ptycho_dataset,
                           training_config, data_config, model_config, inference_config, gpu_ids = None,
                           use_mixed_precision=True, verbose = False)


    #Save results
    result_im = result.to('cpu')
    if len(result_im.shape) == 3:
        result_im = result_im[0].squeeze()

    w = inference_config.window
    result_amp = np.abs(result_im)
    result_phase = np.angle(result_im)
    gt_amp = np.abs(recon_dataset.data_dict['objectGuess']).squeeze()
    gt_phase = np.angle(recon_dataset.data_dict['objectGuess']).squeeze()

    plot_amp_and_phase(result_amp[w:-w,w:-w], result_phase[w:-w,w:-w],
                       gt_amp[w:-w,w:-w], gt_phase[w:-w,w:-w],
                       save_dir = save_dir, filename = plot_name)

    print(f"Model load time: {model_load_time} \n "
          f"Data load time: {data_load_time}\n"
          f"Total inference time: {assembly_stats[0]}\n"
          f"Total assembly time: {assembly_stats[1]}")

    return result


def plot_amp_and_phase(obj_amp, obj_phase, gt_amp, gt_phase, save_dir = None, filename = None):
    """
    Plot amplitude and phase comparison with ground truth.

    Creates a 2x2 grid showing reconstructed amplitude, reconstructed phase,
    ground truth amplitude, and ground truth phase.

    Args:
        obj_amp: Reconstructed amplitude array
        obj_phase: Reconstructed phase array
        gt_amp: Ground truth amplitude array
        gt_phase: Ground truth phase array
        save_dir: Optional directory to save plot
        filename: Optional filename for saved plot
    """
    fig, axs = plt.subplots(2,2, figsize=(5,5))

    #Object amp
    obj_plot = axs[0,0].imshow(obj_amp, cmap = 'gray')
    plt.colorbar(obj_plot, ax = axs[0,0])
    axs[0,0].set_title('Object Amplitude')
    axs[0,0].axis('off')

    #Object Phase
    phase_plot = axs[0,1].imshow(obj_phase, cmap = 'gray')#, vmin=-1, vmax=1)
    plt.colorbar(phase_plot, ax = axs[0,1])
    axs[0,1].set_title('Object Phase')
    axs[0,1].axis('off')

    #Ground turth amp
    gtamp_plot = axs[1,0].imshow(gt_amp, cmap = 'gray')
    plt.colorbar(gtamp_plot, ax = axs[1,0])
    axs[1,0].set_title('Ground Truth Amplitude')
    axs[1,0].axis('off')

    #ground truth phase
    gtphase_plot = axs[1,1].imshow(gt_phase, cmap = 'gray')
    plt.colorbar(gtphase_plot, ax = axs[1,1])
    axs[1,1].set_title('Ground Truth Phase')
    axs[1,1].axis('off')

    # Save the plot if save_dir is provided
    if save_dir is not None:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"amp_phase_comparison_{timestamp}.svg"

        # Ensure filename has an extension
        if not filename.endswith(('.png', '.jpg', '.pdf', '.svg')):
            filename += '.svg'

        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

        plt.show()


def save_individual_reconstructions(obj_amp, obj_phase, output_dir):
    """
    Save individual amplitude and phase reconstructions as separate PNG files.

    This function generates the specific output artifacts expected by the PyTorch
    integration test workflow (Phase E2.C2).

    Args:
        obj_amp: Reconstructed amplitude array (numpy array)
        obj_phase: Reconstructed phase array (numpy array)
        output_dir: Directory to save output images (str or Path)

    Outputs:
        - <output_dir>/reconstructed_amplitude.png
        - <output_dir>/reconstructed_phase.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create amplitude figure
    fig_amp, ax_amp = plt.subplots(figsize=(6, 6))
    im_amp = ax_amp.imshow(obj_amp, cmap='gray')
    plt.colorbar(im_amp, ax=ax_amp)
    ax_amp.set_title('Reconstructed Amplitude')
    ax_amp.axis('off')

    amp_path = output_dir / "reconstructed_amplitude.png"
    plt.savefig(amp_path, dpi=150, bbox_inches='tight')
    plt.close(fig_amp)
    print(f"Saved amplitude reconstruction to: {amp_path}")

    # Create phase figure
    fig_phase, ax_phase = plt.subplots(figsize=(6, 6))
    im_phase = ax_phase.imshow(obj_phase, cmap='gray')
    plt.colorbar(im_phase, ax=ax_phase)
    ax_phase.set_title('Reconstructed Phase')
    ax_phase.axis('off')

    phase_path = output_dir / "reconstructed_phase.png"
    plt.savefig(phase_path, dpi=150, bbox_inches='tight')
    plt.close(fig_phase)
    print(f"Saved phase reconstruction to: {phase_path}")


def _run_inference_and_reconstruct(model, raw_data, config, execution_config, device, quiet=False):
    """
    Extract inference logic into testable helper function (Phase D.C C3).

    Args:
        model: Loaded Lightning module (should be in eval mode)
        raw_data: RawData instance with test data
        config: TFInferenceConfig with n_groups, etc.
        execution_config: PyTorchExecutionConfig with device, batch size, etc.
        device: Torch device string ('cpu', 'cuda', 'mps')
        quiet: Suppress progress output (default: False)

    Returns:
        Tuple of (amplitude, phase) numpy arrays

    Notes:
        - Wraps existing simplified inference logic (lines 563-641)
        - Enforces DTYPE-001 (float32 for diffraction, complex64 for probe)
        - Handles shape permutations (H,W,N → N,H,W)
        - Averages across batch for single reconstruction
        - DEVICE-MISMATCH-001: Ensures model is on the correct device
    """
    import torch
    import numpy as np

    # DEVICE-MISMATCH-001 fix: Ensure model is on the requested device and in eval mode
    model.to(device)
    model.eval()

    # DTYPE ENFORCEMENT (Phase D1d): Cast to float32 per DATA-001
    diffraction = torch.from_numpy(raw_data.diff3d).to(device, dtype=torch.float32)
    probe = torch.from_numpy(raw_data.probeGuess).to(device, dtype=torch.complex64)

    # Handle different diffraction shapes (H, W, n) vs (n, H, W)
    # Auto-detect legacy (H, W, n) format where the last dim (n) is the largest
    if diffraction.ndim == 3 and diffraction.shape[-1] > max(diffraction.shape[0], diffraction.shape[1]):
        # Transpose from (H, W, n) to (n, H, W)
        diffraction = diffraction.permute(2, 0, 1)

    # Limit to n_groups
    diffraction = diffraction[:config.n_groups]

    # Add channel dimension if needed: (n, H, W) -> (n, 1, H, W)
    if diffraction.ndim == 3:
        diffraction = diffraction.unsqueeze(1)

    # Ensure probe is complex64
    if not torch.is_complex(probe):
        probe = probe.to(torch.complex64)

    # Add batch dimension to probe if needed
    if probe.ndim == 2:
        probe = probe.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)

    # Prepare positions (API requires it), real offsets computed for reassembly below
    batch_size = diffraction.shape[0]
    N = diffraction.shape[-1]
    positions = torch.zeros((batch_size, 1, 1, 2), device=device)

    # Prepare scaling factors (match training normalization)
    from ptycho_torch import helper as hh
    from ptycho_torch.config_params import DataConfig as PTDataConfig

    data_cfg_norm = PTDataConfig(N=int(N), grid_size=(1, 1))
    rms_scale = hh.get_rms_scaling_factor(diffraction.squeeze(1), data_cfg_norm)
    physics_scale = hh.get_physics_scaling_factor(diffraction.squeeze(1), data_cfg_norm)
    if not isinstance(rms_scale, torch.Tensor):
        rms_scale = torch.from_numpy(rms_scale)
    if not isinstance(physics_scale, torch.Tensor):
        physics_scale = torch.from_numpy(physics_scale)
    rms_scale = rms_scale.to(device=device, dtype=torch.float32)
    physics_scale = physics_scale.to(device=device, dtype=torch.float32)
    if rms_scale.ndim == 1:
        rms_scale = rms_scale.view(-1, 1, 1, 1)
    if physics_scale.ndim == 1:
        physics_scale = physics_scale.view(-1, 1, 1, 1)

    physics_weight = 1.0 if getattr(model, 'torch_loss_mode', 'poisson') == 'poisson' else 0.0
    input_scale_factor = rms_scale
    output_scale_factor = (1.0 - physics_weight) * rms_scale + physics_weight * physics_scale

    if not quiet:
        print(f"Running inference on {batch_size} images...")

    # Forward pass through model to get per-patch complex predictions
    with torch.no_grad():
        patch_complex = model.forward_predict(
            diffraction,
            positions,
            probe,
            input_scale_factor
        )

    # Compute pixel offsets relative to center-of-mass (B, 1, 1, 2)
    x = torch.from_numpy(raw_data.xcoords[:batch_size]).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(raw_data.ycoords[:batch_size]).to(device=device, dtype=torch.float32)
    dx = x - torch.mean(x)
    dy = y - torch.mean(y)
    offsets = torch.stack([dx, dy], dim=-1).view(batch_size, 1, 1, 2)

    # Position-aware reassembly using torch helper to produce stitched canvas
    from ptycho_torch.config_params import DataConfig, ModelConfig
    from ptycho_torch import helper as hh

    # Minimal configs required for padding and translation
    N = patch_complex.shape[-1]
    data_cfg = DataConfig(N=int(N), grid_size=(1, 1))
    model_cfg = ModelConfig()
    # Ensure channel consistency for reassembly (C_forward must match predicted channels)
    model_cfg.C_forward = int(patch_complex.shape[1])

    # Compute dynamic canvas size to avoid clipping: M >= N + 2*max(|dx|, |dy|)
    max_shift = torch.max(torch.stack([dx.abs(), dy.abs()], dim=0)).item()
    M = int(np.ceil(N + 2 * max_shift))
    imgs_merged, _, _ = hh.reassemble_patches_position_real(
        patch_complex, offsets, data_cfg, model_cfg, padded_size=M
    )

    # Convert to numpy amplitude/phase
    canvas = imgs_merged[0]  # (M, M)
    result_amp = torch.abs(canvas).cpu().numpy()
    result_phase = torch.angle(canvas).cpu().numpy()

    if not quiet:
        print(f"Reconstruction shape: {result_amp.shape}")
        print(f"Amplitude range: [{result_amp.min():.4f}, {result_amp.max():.4f}]")
        print(f"Phase range: [{result_phase.min():.4f}, {result_phase.max():.4f}]")

    return result_amp, result_phase


def cli_main():
    """
    CLI entrypoint for PyTorch Lightning checkpoint inference (ADR-003 Phase D.C thin wrapper).

    Thin wrapper that delegates to shared helpers (ptycho_torch.cli.shared) for validation,
    execution config construction, and device resolution. Inference orchestration extracted
    to _run_inference_and_reconstruct() helper for testability.

    Usage:
        python -m ptycho_torch.inference \\
            --model_path <training_output_dir> \\
            --test_data <npz_file> \\
            --output_dir <inference_output_dir> \\
            --n_images 32 \\
            --accelerator cpu \\
            [--quiet]

    Expected Output Artifacts:
        - <output_dir>/reconstructed_amplitude.png
        - <output_dir>/reconstructed_phase.png

    References:
        - Blueprint: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T114500Z/phase_d_cli_wrappers_inference/inference_refactor.md
        - Test contract: tests/torch/test_cli_inference_torch.py
        - Shared helpers: ptycho_torch/cli/shared.py
    """
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning checkpoint inference for ptychography reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on trained model
  python -m ptycho_torch.inference \\
      --model_path training_outputs \\
      --test_data datasets/Run1084_recon3_postPC_shrunk_3.npz \\
      --output_dir inference_outputs \\
      --n_images 32 \\
      --device cpu

  # Run with quiet output
  python -m ptycho_torch.inference \\
      --model_path training_outputs \\
      --test_data test.npz \\
      --output_dir outputs \\
      --n_images 64 \\
      --device cuda \\
      --quiet
        """
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to training output directory containing Lightning checkpoint (expects checkpoints/last.ckpt or wts.pt)'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='Path to test data NPZ file (must conform to specs/data_contracts.md)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save reconstruction outputs (amplitude/phase PNGs)'
    )
    parser.add_argument(
        '--n_images',
        type=int,
        default=32,
        help='Number of images to use for reconstruction (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to run inference on (cpu or cuda, default: cpu)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    # Execution config flags (Phase C4.C5 - ADR-003)
    parser.add_argument(
        '--accelerator',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps'],
        help=(
            'Hardware accelerator: auto (auto-detect, default), cpu (CPU-only), '
            'gpu (NVIDIA GPU), cuda (alias for gpu), tpu (Google TPU), mps (Apple Silicon).'
        )
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        dest='num_workers',
        help=(
            'Number of DataLoader worker processes (default: 0 = synchronous). '
            'Typical values: 2-8 for multi-core systems.'
        )
    )
    parser.add_argument(
        '--inference-batch-size',
        type=int,
        default=None,
        dest='inference_batch_size',
        help=(
            'Batch size for inference DataLoader (default: None = use training batch_size). '
            'Larger values increase throughput. Typical: 16-64 for GPU, 4-8 for CPU.'
        )
    )

    args = parser.parse_args()

    # --- Phase D.C C3: Validate paths using shared helper ---
    from ptycho_torch.cli.shared import validate_paths
    try:
        validate_paths(
            train_file=None,  # Inference mode: no training file
            test_file=Path(args.test_data),
            output_dir=Path(args.output_dir),
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # --- Phase D.C C3: Build execution config using shared helper ---
    from ptycho_torch.cli.shared import build_execution_config_from_args
    try:
        execution_config = build_execution_config_from_args(args, mode='inference')
    except ValueError as e:
        print(f"ERROR: Invalid execution config: {e}")
        sys.exit(1)

    # Fail-fast: Check Lightning availability
    try:
        import lightning as L
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch Lightning backend requires 'lightning' and 'torch' packages. "
            "Install via: pip install -e .[torch]\n"
            f"Import error: {e}"
        )

    # Phase C4.C6+C4.C7: Delegate to factory for CONFIG-001 compliance (ADR-003)
    # Replaces manual checkpoint loading and config construction with centralized
    # factory pattern. The factory handles:
    # 1. Path validation and checkpoint discovery
    # 2. CONFIG-001 bridging (update_legacy_dict before any IO)
    # 3. Config translation (PyTorch → TensorFlow canonical dataclasses)
    # 4. Execution config merging with override precedence

    from ptycho_torch.config_factory import create_inference_payload
    from ptycho.raw_data import RawData

    # Convert paths to Path objects
    model_path = Path(args.model_path)
    test_data_path = Path(args.test_data)
    output_dir = Path(args.output_dir)

    # Build overrides dict for factory
    overrides = {
        'n_groups': args.n_images,  # Map CLI arg to config field
    }

    # Call factory to construct all configs and populate params.cfg
    try:
        payload = create_inference_payload(
            model_path=model_path,
            test_data_file=test_data_path,
            output_dir=output_dir,
            overrides=overrides,
            execution_config=execution_config,
        )

        # Extract configs from payload (factory already populated params.cfg)
        pt_data_config = payload.pt_data_config
        tf_inference_config = payload.tf_inference_config
        execution_config = payload.execution_config

        if not args.quiet:
            print(f"Loaded configuration from model checkpoint")
            print(f"Test data: {test_data_path}")
            print(f"Output directory: {output_dir}")
            print(f"N groups: {tf_inference_config.n_groups}")
            print(f"Execution config: accelerator={execution_config.accelerator}, "
                  f"num_workers={execution_config.num_workers}")

    except Exception as e:
        raise RuntimeError(
            f"Failed to create inference payload.\n"
            f"Error: {e}\n"
            "Ensure model_path contains wts.h5.zip and test_data conforms to DATA-001."
        )

    # Load checkpoint via spec-compliant bundle loader (Phase C4.C6/C4.C7 - ADR-003)
    # Replaces manual checkpoint search with factory-validated wts.h5.zip loading
    try:
        import torch
        from ptycho_torch.workflows.components import load_inference_bundle_torch

        # load_inference_bundle_torch expects bundle_dir containing wts.h5.zip
        # It handles CONFIG-001 (restores params.cfg from archive) and returns
        # (models_dict, params_dict) matching TensorFlow baseline API
        models_dict, params_dict = load_inference_bundle_torch(
            bundle_dir=model_path,
            model_name='diffraction_to_obj'
        )

        # Extract Lightning module from models dict
        model = models_dict['diffraction_to_obj']
        model.eval()

        # Resolve device from execution config
        device_map = {
            'cpu': 'cpu',
            'gpu': 'cuda',
            'cuda': 'cuda',
            'mps': 'mps',
            'auto': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
        device = device_map.get(execution_config.accelerator, 'cpu')
        model.to(device)

        if not args.quiet:
            print(f"Loaded model bundle from: {model_path / 'wts.h5.zip'}")
            print(f"Model device: {device}")
            print(f"Restored params.cfg from bundle (N={params_dict.get('N', 'N/A')}, "
                  f"gridsize={params_dict.get('gridsize', 'N/A')})")

    except Exception as e:
        raise RuntimeError(
            f"Failed to load inference bundle from {model_path}.\n"
            f"Error: {e}\n"
            "Ensure model_path contains wts.h5.zip archive (spec-compliant format)."
        )

    # Load test data via RawData (factory already validated path)
    # NOTE: params.cfg already populated by factory, so RawData.from_file is safe to call
    try:
        raw_data = RawData.from_file(str(test_data_path))

        if not args.quiet:
            print(f"Loaded test data: {raw_data.diff3d.shape[0]} scan positions")

    except Exception as e:
        raise RuntimeError(
            f"Failed to load test data from {test_data_path}.\n"
            f"Error: {e}\n"
            "Ensure NPZ conforms to specs/data_contracts.md"
        )

    # --- Phase D.C C3: Delegate to inference helper ---
    try:
        amplitude, phase = _run_inference_and_reconstruct(
            model=model,
            raw_data=raw_data,
            config=tf_inference_config,
            execution_config=execution_config,
            device=device,
            quiet=args.quiet,
        )

        # Save individual reconstructions (required by test contract)
        save_individual_reconstructions(amplitude, phase, output_dir)

        if not args.quiet:
            print(f"\nInference completed successfully!")
            print(f"Output artifacts saved to: {output_dir}")

        return 0

    except Exception as e:
        print(f"ERROR: Inference failed with exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    # Determine which mode to run based on command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['--model_path', '--help', '-h']:
        # New CLI path (Phase E2.C2)
        sys.exit(cli_main())
    else:
        # Legacy MLflow-based inference path
        parser = argparse.ArgumentParser(description="Arguments for inference script (MLflow mode)")
        parser.add_argument('--run_id', type = str, help = "Unique run id associated with training run")
        parser.add_argument('--infer_dir', type = str, help = "Inference directory")
        parser.add_argument('--file_index', type = int, default = 0, help = "File index if more than one file in infer_dir")
        parser.add_argument('--config', type = str, default = None, help = "Config to override loaded values")

        args = parser.parse_args()

        run_id = args.run_id
        infer_dir = args.infer_dir
        file_index = args.file_index
        config_override = args.config

        try:
            load_and_predict(run_id, infer_dir, 'mlruns',
                             config_override_path=config_override,
                             file_index = file_index)
        except Exception as e:
            print(f"Inference failed because of: {str(e)}")
            sys.exit(1)

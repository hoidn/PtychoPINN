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


def cli_main():
    """
    CLI entrypoint for PyTorch Lightning checkpoint inference.

    This function implements Phase E2.C2 of INTEGRATE-PYTORCH-001, providing
    a command-line interface for loading trained PyTorch models and generating
    reconstructions from test data.

    Usage:
        python -m ptycho_torch.inference \\
            --model_path <training_output_dir> \\
            --test_data <npz_file> \\
            --output_dir <inference_output_dir> \\
            --n_images 32 \\
            --device cpu \\
            [--quiet]

    Expected Output Artifacts:
        - <output_dir>/reconstructed_amplitude.png
        - <output_dir>/reconstructed_phase.png

    References:
        - Phase E2 plan: plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md §E2.C2
        - Test contract: tests/torch/test_integration_workflow_torch.py
        - Red phase evidence: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T213500Z/red_phase.md §2.3
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

    args = parser.parse_args()

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

    # Convert paths to Path objects
    model_path = Path(args.model_path)
    test_data_path = Path(args.test_data)
    output_dir = Path(args.output_dir)

    # Validate input paths
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}\n"
            "Provide a valid training output directory containing a checkpoint."
        )

    if not test_data_path.exists():
        raise FileNotFoundError(
            f"Test data file does not exist: {test_data_path}\n"
            "Provide a valid NPZ file conforming to specs/data_contracts.md"
        )

    # Locate Lightning checkpoint
    checkpoint_candidates = [
        model_path / "checkpoints" / "last.ckpt",  # Lightning default
        model_path / "wts.pt",                      # Custom bundle format
        model_path / "model.pt",                    # Alternative naming
    ]

    checkpoint_path = None
    for candidate in checkpoint_candidates:
        if candidate.exists():
            checkpoint_path = candidate
            break

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No Lightning checkpoint found in {model_path}.\n"
            f"Searched for: {[str(p) for p in checkpoint_candidates]}\n"
            "Ensure training completed successfully and checkpoint was saved."
        )

    if not args.quiet:
        print(f"Loading Lightning checkpoint from: {checkpoint_path}")
        print(f"Test data: {test_data_path}")
        print(f"Output directory: {output_dir}")
        print(f"Device: {args.device}")
        print(f"N images: {args.n_images}")

    # Load Lightning module from checkpoint
    try:
        from ptycho_torch.model import PtychoPINN_Lightning

        model = PtychoPINN_Lightning.load_from_checkpoint(
            str(checkpoint_path),
            map_location=args.device
        )
        model.eval()
        model.to(args.device)

        if not args.quiet:
            print(f"Successfully loaded model from checkpoint")
            print(f"Model type: {type(model).__name__}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Lightning checkpoint from {checkpoint_path}.\n"
            f"Error: {e}\n"
            "Ensure checkpoint was saved during training and is compatible with current code."
        )

    # Load test data
    try:
        import torch
        test_data = np.load(test_data_path)

        if not args.quiet:
            print(f"Loaded test data with keys: {list(test_data.keys())}")

        # Validate required fields (per specs/data_contracts.md)
        required_fields = ['diffraction', 'probeGuess', 'objectGuess']
        missing_fields = [f for f in required_fields if f not in test_data]
        if missing_fields:
            raise ValueError(
                f"Test data missing required fields: {missing_fields}\n"
                f"Available fields: {list(test_data.keys())}\n"
                "Ensure NPZ conforms to specs/data_contracts.md"
            )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load test data from {test_data_path}.\n"
            f"Error: {e}"
        )

    # Run inference using the model's existing inference pipeline
    # NOTE: This is a simplified inference path for Phase E2.C2.
    # Full reconstruction using reassembly logic is handled by the existing
    # load_and_predict() function. For now, we perform a forward pass and
    # extract amplitude/phase from the autoencoder output.

    try:
        # Extract data from NPZ
        # DTYPE ENFORCEMENT (Phase D1d): Cast to float32 to prevent Conv2d dtype mismatch
        # Requirement: specs/data_contracts.md §1 mandates diffraction arrays be float32
        # Root cause: torch.from_numpy preserves dtype; legacy NPZ files may contain float64
        # Symptom: RuntimeError "Input type (double) and bias type (float)" in model forward
        diffraction = torch.from_numpy(test_data['diffraction']).to(args.device, dtype=torch.float32)
        probe = torch.from_numpy(test_data['probeGuess']).to(args.device, dtype=torch.complex64)

        # Handle different diffraction shapes (H, W, n) vs (n, H, W)
        # Per specs/data_contracts.md, diffraction should be (H, W, n)
        if diffraction.ndim == 3 and diffraction.shape[-1] < diffraction.shape[0]:
            # Transpose from (H, W, n) to (n, H, W)
            diffraction = diffraction.permute(2, 0, 1)

        # Limit to n_images
        diffraction = diffraction[:args.n_images]

        # Add channel dimension if needed: (n, H, W) -> (n, 1, H, W)
        if diffraction.ndim == 3:
            diffraction = diffraction.unsqueeze(1)

        # Ensure probe is complex64
        if not torch.is_complex(probe):
            probe = probe.to(torch.complex64)

        # Add batch dimension to probe if needed
        if probe.ndim == 2:
            probe = probe.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)

        # Prepare dummy positions (needed for forward pass signature)
        # In real reconstruction, these come from scan positions
        batch_size = diffraction.shape[0]
        positions = torch.zeros((batch_size, 1, 1, 2), device=args.device)

        # Prepare scaling factors (simplified for Phase E2.C2)
        # Full scaling logic handled by dataloader in production
        input_scale_factor = torch.ones((batch_size, 1, 1, 1), device=args.device)

        if not args.quiet:
            print(f"Running inference on {batch_size} images...")

        # Forward pass through model
        with torch.no_grad():
            # Use forward_predict to get complex reconstruction
            reconstruction = model.forward_predict(
                diffraction,
                positions,
                probe,
                input_scale_factor
            )

        # Extract amplitude and phase
        reconstruction_cpu = reconstruction.cpu().numpy()

        # Average across batch for single reconstruction
        reconstruction_avg = np.mean(reconstruction_cpu, axis=0)

        # Remove channel dimension if present
        if reconstruction_avg.ndim == 3:
            reconstruction_avg = reconstruction_avg[0]

        result_amp = np.abs(reconstruction_avg)
        result_phase = np.angle(reconstruction_avg)

        if not args.quiet:
            print(f"Reconstruction shape: {reconstruction_avg.shape}")
            print(f"Amplitude range: [{result_amp.min():.4f}, {result_amp.max():.4f}]")
            print(f"Phase range: [{result_phase.min():.4f}, {result_phase.max():.4f}]")

        # Save individual reconstructions (required by test)
        save_individual_reconstructions(result_amp, result_phase, output_dir)

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

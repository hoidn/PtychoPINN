"""
PyTorch Training Script for PtychoPINN (INTEGRATE-PYTORCH-001)

This module provides the canonical CLI training entry point for the PyTorch
backend.

CLI Interface (Phase E2.C1):
    python -m ptycho_torch.train \\
        --train_data_file <path>       # Required: Training NPZ dataset
        --test_data_file <path>        # Optional: Validation NPZ dataset
        --output_dir <path>            # Required: Checkpoint output directory
        --max_epochs <int>             # Optional: Training epochs (default: 100)
        --n_images <int>               # Optional: Number of diffraction groups (default: 512)
        --gridsize <int>               # Optional: Grid size for grouping (default: 2)
        --batch_size <int>             # Optional: Training batch size (default: 16)
        --device <cpu|cuda>            # Optional: Compute device (default: cpu)
        --disable_mlflow               # Optional: Suppress MLflow autologging

Key Features:
- CONFIG-001 compliant: Populates params.cfg before workflow dispatch
- Lightning integration: Automatic checkpointing, early stopping, distributed training
- Configuration bridge: PyTorch configs → TensorFlow dataclasses → params.cfg

References:
    - Phase E2.C1 spec: plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md
    - Test contract: tests/torch/test_integration_workflow_torch.py
    - Config bridge: ptycho_torch/config_bridge.py
"""

#Most basic modules
import sys
import argparse
from pathlib import Path

from ptycho.config.legacy_state import scoped_legacy_params

@scoped_legacy_params
def cli_main():
    """
    CLI entrypoint for PyTorch training workflow (Phase E2.C1).

    The CLI resolves configuration through the supported factory and delegates
    training to the versioned workflow path.
    """
    raw_argv = tuple(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning training for ptychographic reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ptycho_torch.train --train_data_file data/train.npz --output_dir ./outputs --max_epochs 10 --device cpu --disable_mlflow
        """
    )

    # New CLI interface flags (Phase E2.C1)
    parser.add_argument('--train_data_file', type=str,
                       help='Path to training NPZ dataset (required for new CLI interface)')
    parser.add_argument('--test_data_file', type=str,
                       help='Path to validation NPZ dataset (optional)')
    parser.add_argument('--output_dir', type=str,
                       help='Directory for checkpoint outputs (required for new CLI interface)')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs (default: 100)')
    parser.add_argument('--n_images', type=int, default=512,
                       help='Number of diffraction groups to process (default: 512)')
    parser.add_argument('--gridsize', type=int, default=2,
                       help='Grid size for spatial grouping (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size (default: 16)')
    parser.add_argument('--log-patch-stats', action='store_true',
                       help='Log per-patch statistics during training/inference (default: disabled)')
    parser.add_argument('--patch-stats-limit', type=int, default=None,
                       help='Maximum number of batches to log for patch stats (default: no limit)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                       help='[DEPRECATED] Use --accelerator instead. Compute device: cpu or cuda (default: cpu)')
    parser.add_argument('--disable_mlflow', action='store_true',
                       help='[DEPRECATED] Use --logger none instead. Disable all experiment tracking loggers.')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress bars and verbose output')
    parser.add_argument(
        '--torch-loss-mode',
        type=str,
        default='poisson',
        choices=['poisson', 'mae'],
        help=(
            "Select the Torch backend loss pipeline. "
            "'poisson' matches the physics-weighted Poisson NLL used in TensorFlow, "
            "while 'mae' disables the physics loss and trains purely on amplitude MAE."
        )
    )
    parser.add_argument(
        '--profile',
        choices=['ci'],
        default=None,
        help=(
            "Named configuration profile. 'ci' selects the coherent "
            "PtychoPINN-CI bundle (ci_intensity_v2/count_intensity contract, "
            "rectangular_scaled forward, Poisson loss, dose-closure-initialized "
            "trainable s1/s2, real/imag CNN heads). Explicit flags that "
            "contradict the profile's contract fields fail closed."
        ),
    )
    parser.add_argument(
        '--rect-s1s2-init',
        choices=['ones', 'dose_closure'],
        default=None,
        help=(
            'Startup initialization for the rectangular s1/s2 gauge. '
            'dose_closure solves the initial gauge from training counts; '
            'this is separate from whether s1/s2 remain trainable.'
        ),
    )
    parser.add_argument(
        '--scale-contract-version',
        choices=['ci_intensity_v2', 'legacy_v1'],
        default=None,
        help='Scaling profile override; must be paired with --measurement-domain.',
    )
    parser.add_argument(
        '--measurement-domain',
        choices=['count_intensity', 'normalized_amplitude'],
        default=None,
        help='Measurement-domain override; must be paired with --scale-contract-version.',
    )
    parser.add_argument(
        '--probe-mask',
        dest='probe_mask',
        action='store_true',
        default=False,
        help='Enable Torch probe masking (default: disabled).'
    )
    parser.add_argument(
        '--no-probe-mask',
        dest='probe_mask',
        action='store_false',
        help='Disable Torch probe masking.'
    )
    parser.add_argument(
        '--probe-mask-sigma',
        type=float,
        default=1.0,
        dest='probe_mask_sigma',
        help='Gaussian sigma (pixels) for probe-mask edge smoothing (default: 1.0 smooth edge).'
    )
    parser.add_argument(
        '--probe-mask-diameter',
        type=float,
        default=None,
        dest='probe_mask_diameter',
        help='Probe-mask disk diameter in pixels (default: N/2).'
    )

    # Execution config flags (Phase C4.C1 - ADR-003)
    parser.add_argument(
        '--accelerator',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps'],
        help=(
            'Hardware accelerator for training: '
            'auto (auto-detect, default), cpu (CPU-only), gpu (NVIDIA GPU), '
            'cuda (alias for gpu), tpu (Google TPU), mps (Apple Silicon). '
            'Default: auto.'
        )
    )
    parser.add_argument(
        '--deterministic',
        dest='deterministic',
        action='store_true',
        default=True,
        help=(
            'Enable deterministic training for reproducibility (default: enabled). '
            'Sets torch.use_deterministic_algorithms(True) and Lightning deterministic=True. '
            'Use --no-deterministic to disable for potential performance gains.'
        )
    )
    parser.add_argument(
        '--no-deterministic',
        dest='deterministic',
        action='store_false',
        help='Disable deterministic training. May improve performance but results are non-reproducible.'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        dest='num_workers',
        help=(
            'Number of DataLoader worker processes for parallel data loading (default: 0 = synchronous). '
            'Typical values: 2-8 for multi-core systems. Higher values increase data loading throughput '
            'but consume more memory. Set to 0 for single-threaded loading (safest for CI).'
        )
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        dest='learning_rate',
        help=(
            'Learning rate for Adam optimizer (default: 1e-3). '
            'Typical range: 1e-5 (slow, stable) to 1e-2 (fast, may diverge). '
            'Adjust based on convergence behavior during training.'
        )
    )

    # Logger backend flags (Phase EB3.B - ADR-003)
    parser.add_argument(
        '--logger',
        type=str,
        default='csv',
        choices=['none', 'csv', 'tensorboard', 'mlflow'],
        dest='logger_backend',
        help=(
            'Experiment tracking logger backend (default: csv). '
            'Options: none (no logging), csv (Lightning CSVLogger), '
            'tensorboard (TensorBoard via Lightning), mlflow (MLflow via Lightning). '
            'Loss metrics are logged to {output_dir}/lightning_logs/{version}/. '
            'Use --logger none if you only need progress suppression (no metrics).'
        )
    )

    # Checkpoint and early stopping flags (Phase EB1.B - ADR-003)
    parser.add_argument(
        '--enable-checkpointing',
        dest='enable_checkpointing',
        action='store_true',
        default=True,
        help=(
            'Enable automatic model checkpointing during training (default: enabled). '
            'Checkpoints are saved based on monitored metric performance. '
            'Use --disable-checkpointing to turn off.'
        )
    )
    parser.add_argument(
        '--disable-checkpointing',
        dest='enable_checkpointing',
        action='store_false',
        help='Disable automatic model checkpointing during training.'
    )
    parser.add_argument(
        '--checkpoint-save-top-k',
        type=int,
        default=1,
        dest='checkpoint_save_top_k',
        help=(
            'Number of best checkpoints to keep (default: 1). '
            'Set to 0 to disable saving. '
            'Best checkpoints are determined by --checkpoint-monitor metric.'
        )
    )
    parser.add_argument(
        '--checkpoint-monitor',
        type=str,
        default='val_loss',
        dest='checkpoint_monitor_metric',
        help=(
            'Metric to monitor for checkpoint selection (default: val_loss). '
            'Common options: val_loss, train_loss, val_accuracy. '
            'Must match a metric logged by the Lightning module.'
        )
    )
    parser.add_argument(
        '--checkpoint-mode',
        type=str,
        default='min',
        choices=['min', 'max'],
        dest='checkpoint_mode',
        help=(
            'Mode for checkpoint metric optimization (default: min). '
            'Use "min" for metrics where lower is better (e.g., loss), '
            '"max" for metrics where higher is better (e.g., accuracy).'
        )
    )
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=100,
        dest='early_stop_patience',
        help=(
            'Early stopping patience in epochs (default: 100). '
            'Training stops if monitored metric does not improve for this many epochs. '
            'Set to large value (e.g., 1000) to effectively disable early stopping.'
        )
    )

    # Optimization knobs (Phase EB2 - ADR-003)
    parser.add_argument(
        '--scheduler',
        type=str,
        default='Default',
        choices=[
            'Default',
            'Exponential',
            'MultiStage',
            'Adaptive',
            'WarmupCosine',
            'ReduceLROnPlateau',
        ],
        dest='scheduler',
        help=(
            'Learning rate scheduler type (default: Default). '
            'Choices: Default (no scheduler), Exponential (exponential decay), '
            'MultiStage, Adaptive, WarmupCosine, or ReduceLROnPlateau. '
            'Scheduler configuration must match Lightning module expectations.'
        )
    )
    parser.add_argument(
        '--accumulate-grad-batches',
        type=int,
        default=1,
        dest='accumulate_grad_batches',
        help=(
            'Number of gradient accumulation steps (default: 1 = no accumulation). '
            'Accumulation simulates larger effective batch sizes by accumulating gradients '
            'over multiple forward/backward passes before updating weights. '
            'Effective batch size = batch_size * accumulate_grad_batches. '
            'WARNING: Values >1 increase memory efficiency but may affect training dynamics. '
            'Typical values: 1-8 depending on GPU memory and batch size constraints.'
        )
    )

    args = parser.parse_args()

    canonical_interface = (
        args.train_data_file is not None or args.output_dir is not None
    )
    if canonical_interface:
        # New CLI interface: --train_data_file --output_dir ... (Phase C4.C2 - ADR-003)
        print("Using new CLI interface with factory-based config (ADR-003)")

        # Validate required arguments
        if not args.train_data_file:
            print("ERROR: --train_data_file required for new CLI interface")
            sys.exit(1)
        if not args.output_dir:
            print("ERROR: --output_dir required for new CLI interface")
            sys.exit(1)

        # Convert paths to Path objects
        train_data_file = Path(args.train_data_file)
        test_data_file = Path(args.test_data_file) if args.test_data_file else None
        output_dir = Path(args.output_dir)

        # Validate paths using shared helper (Phase D.B - ADR-003)
        from ptycho_torch.cli.shared import validate_paths
        try:
            validate_paths(train_data_file, test_data_file, output_dir)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        # Preserve raw-option suppliedness until the factory resolves runtime.
        from ptycho_torch.cli.shared import (
            build_execution_request_from_args,
            build_training_config_patch_from_args,
        )
        try:
            execution_request = build_execution_request_from_args(
                args,
                mode='training',
                explicit_options=raw_argv,
                lane='native-training',
            )
            training_patch = build_training_config_patch_from_args(
                args,
                explicit_options=raw_argv,
                lane='native-training',
            )
        except ValueError as e:
            print(f"ERROR: Invalid CLI configuration: {e}")
            sys.exit(1)

        # Phase C4.C2: Use config factory instead of manual config construction
        print("Creating configuration via factory (CONFIG-001 compliance)...")
        from ptycho_torch.config_factory import create_training_payload

        # Build overrides dict from CLI arguments
        overrides = {
            'training_groups': args.n_images,
            'batch_size': args.batch_size,
            'gridsize': args.gridsize,
            'max_epochs': args.max_epochs,
            'torch_loss_mode': args.torch_loss_mode,
            'probe_mask': args.probe_mask,
            'probe_mask_sigma': args.probe_mask_sigma,
            'probe_mask_diameter': args.probe_mask_diameter,
            'log_patch_stats': args.log_patch_stats,
            'patch_stats_limit': args.patch_stats_limit,
            'object_big': args.gridsize > 1,
        }
        if args.scale_contract_version is not None:
            overrides['scale_contract_version'] = args.scale_contract_version
        if args.measurement_domain is not None:
            overrides['measurement_domain'] = args.measurement_domain
        if args.rect_s1s2_init is not None:
            overrides['rect_s1s2_init'] = args.rect_s1s2_init
        if test_data_file:
            overrides['test_data_file'] = test_data_file
        overrides.update(training_patch)

        try:
            # Call factory to create all configs and populate params.cfg
            payload = create_training_payload(
                train_data_file=train_data_file,
                output_dir=output_dir,
                overrides=overrides,
                execution_config=execution_request,
                profile=args.profile,
            )

            print(f"✓ Factory created configs: N={payload.pt_data_config.N}, "
                  f"gridsize={payload.pt_data_config.gridsize}, "
                  f"epochs={payload.pt_training_config.epochs}")
            print(f"✓ Execution config: accelerator={payload.execution_config.accelerator}, "
                  f"deterministic={payload.execution_config.deterministic}, "
                  f"learning_rate={payload.pt_training_config.learning_rate}")

        except Exception as e:
            print(f"ERROR: Configuration factory failed: {e}")
            print("Cannot proceed - factory responsible for CONFIG-001 compliance")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Call workflow-based training (Phase C4.D3 - bundle persistence)
        try:
            print(f"Starting training with {args.max_epochs} epochs...")

            # Load training data (CONFIG-001 already satisfied by factory)
            from ptycho.raw_data import RawData
            train_data = RawData.from_file(str(train_data_file))
            test_data = RawData.from_file(str(test_data_file)) if test_data_file else None

            # Route through run_cdi_example_torch for bundle persistence
            from ptycho_torch.workflows.legacy import run_cdi_example_torch
            amplitude, phase, results = run_cdi_example_torch(
                train_data=train_data,
                test_data=test_data,
                config=payload.tf_training_config,
                do_stitching=False,  # CLI only needs training, not reconstruction
                resolved_payload=payload,
            )

            if payload.pt_inference_config.log_patch_stats:
                from ptycho_torch.patch_stats_instrumentation import PatchStatsLogger
                import torch

                train_container = results.get('train_container') if isinstance(results, dict) else None
                amp_tensor = None
                if train_container is not None:
                    if hasattr(train_container, 'Y_I'):
                        amp_tensor = train_container.Y_I
                    elif hasattr(train_container, 'Y'):
                        amp_tensor = torch.abs(train_container.Y)

                if amp_tensor is not None:
                    amp_tensor = torch.as_tensor(amp_tensor)
                    if (amp_tensor.ndim == 4
                            and amp_tensor.shape[1] == amp_tensor.shape[2]
                            and amp_tensor.shape[3] != amp_tensor.shape[1]):
                        amp_tensor = amp_tensor.permute(0, 3, 1, 2)
                    logger = PatchStatsLogger(
                        output_dir=output_dir / "analysis",
                        enabled=True,
                        limit=payload.pt_inference_config.patch_stats_limit,
                    )
                    logger.log_batch(amp_tensor, phase="train", batch_idx=0)
                    logger.finalize()

            print(f"✓ Training completed successfully. Outputs saved to {output_dir}")
            print(f"✓ Model bundle saved to {output_dir}/wts.h5.zip")

        except Exception as e:
            print(f"Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        print(
            "ERROR: --train_data_file and --output_dir are required for "
            "PyTorch training"
        )
        parser.print_help()
        sys.exit(1)


#Define main function
if __name__ == '__main__':
    cli_main()

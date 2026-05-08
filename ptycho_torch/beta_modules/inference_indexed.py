"""Inference script for PtychoDatasetIndexed with CCNF or UNet models.

Reconstructs ptychographic images from a Lightning checkpoint using the
index-only dataloader. Reuses the probe-weighted barycentric reconstruction
pipeline from reassembly.py and plotting functions from inference.py.

Usage:
    python -m ptycho_torch.beta_modules.inference_indexed \
        --ckpt_path /tmp/ccnf_run/ccnf_YYYYMMDD/checkpoints/last.ckpt \
        --ptycho_dir data/pinn_velo_ic_2 \
        --output_dir /tmp/ccnf_infer \
        [--config ptycho_torch/configs/testing_configs/ccnf_indexed.json] \
        [--experiment_number 0] \
        [--device auto] \
        [--batch_size 1000] \
        [--middle_trim 50] \
        [--quiet]
"""

import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ptycho_torch.config_params import (
    DataConfig, ModelConfig, TrainingConfig, InferenceConfig,
    update_existing_config,
)
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.beta_modules.dataloader_index import PtychoDatasetIndexed
from ptycho_torch.reassembly import reconstruct_image_barycentric
from ptycho_torch.inference import (
    plot_amp_and_phase,
    plot_reim_histogram,
    save_individual_reconstructions,
)


def _resolve_checkpoint_path(ckpt_path):
    """Resolve a checkpoint path that may be a directory or a .ckpt file.

    Accepts:
        - Direct path to a .ckpt file
        - Run directory containing checkpoints/best-checkpoint.ckpt or last.ckpt
        - Partial directory name (glob match with trailing wildcard)
    """
    ckpt_path = Path(ckpt_path)

    if ckpt_path.is_file():
        return ckpt_path

    # Try glob match for partial directory names
    if not ckpt_path.exists():
        parent = ckpt_path.parent
        pattern = ckpt_path.name + '*'
        matches = sorted(parent.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No match for: {ckpt_path}")
        candidates = matches
    else:
        candidates = [ckpt_path]

    ckpt_candidates = ['checkpoints/best-checkpoint.ckpt',
                       'checkpoints/last.ckpt']
    for d in candidates:
        if d.is_dir():
            for c in ckpt_candidates:
                full = d / c
                if full.exists():
                    return full

    tried = ', '.join(str(d) for d in candidates)
    raise FileNotFoundError(
        f"No .ckpt files found in: {tried}")

    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")


def load_model_from_checkpoint(ckpt_path, device='cpu'):
    """Load PtychoPINN_Lightning and extract configs from checkpoint.

    Args:
        ckpt_path: Path to Lightning .ckpt file, run directory, or partial
            directory name (e.g. 'lightning_outputs/ccnf_20260428_1455').
        device: Target device string ('cpu', 'cuda', 'auto').

    Returns:
        (model, data_config, model_config, training_config, inference_config)
    """
    ckpt_path = _resolve_checkpoint_path(ckpt_path)

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PtychoPINN_Lightning.load_from_checkpoint(
        str(ckpt_path), map_location=device
    )
    model.to(device)
    model.eval()

    data_config = model.data_config
    model_config = model.model_config
    training_config = model.training_config
    inference_config = model.inference_config

    return model, data_config, model_config, training_config, inference_config


def build_indexed_dataset(ptycho_dir, model_config, data_config,
                          training_config, mmap_dir=None):
    """Build PtychoDatasetIndexed for inference.

    Args:
        ptycho_dir: Directory containing NPZ scan files.
        model_config: ModelConfig from checkpoint.
        data_config: DataConfig from checkpoint (will be copied and modified).
        training_config: TrainingConfig from checkpoint (will be copied and modified).
        mmap_dir: Directory for memory-mapped pattern store. Defaults to
            <ptycho_dir>/../memmap_inference.

    Returns:
        PtychoDatasetIndexed instance.
    """
    data_config = copy.deepcopy(data_config)
    training_config = copy.deepcopy(training_config)

    training_config.orchestrator = "Mlflow"

    update_existing_config(data_config, {
        'probe_normalize': False,
        'x_bounds': [0.03, 0.97],
        'y_bounds': [0.03, 0.97],
    })

    if mmap_dir is None:
        mmap_dir = os.path.join(os.path.dirname(ptycho_dir.rstrip('/')),
                                'memmap_inference')

    dataset = PtychoDatasetIndexed(
        ptycho_dir=str(ptycho_dir),
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        data_dir=mmap_dir,
        remake_map=True,
    )

    # Build mmap_ptycho view so reconstruct_image_barycentric can access
    # coords_global directly (it skips get_experiment_dataset when n_files==1)
    if not hasattr(dataset, 'mmap_ptycho'):
        from tensordict import TensorDict as TD
        dataset.mmap_ptycho = TD(
            {
                "coords_global": dataset.coords_global_group,
                "coords_center": dataset.coords_center,
                "coords_relative": dataset.coords_relative,
                "experiment_id": dataset.group_experiment_id,
                "nn_indices": dataset.nn_indices,
                "rms_scaling_constant": dataset.rms_scaling,
                "physics_scaling_constant": dataset.physics_scaling,
            },
            batch_size=dataset.length,
        )

    return dataset


def run_reconstruction(model, dataset, data_config, model_config,
                       training_config, inference_config, device='cpu',
                       quiet=False):
    """Run probe-weighted barycentric reconstruction.

    Args:
        model: PtychoPINN_Lightning in eval mode.
        dataset: PtychoDatasetIndexed instance.
        data_config: DataConfig.
        model_config: ModelConfig.
        training_config: TrainingConfig (device field used for single-GPU).
        inference_config: InferenceConfig.
        device: Device string.
        quiet: Suppress progress output.

    Returns:
        (result_complex, dataset_subset, assembly_stats, modified_result)
    """
    training_config = copy.deepcopy(training_config)
    training_config.device = device

    result, subset, stats, modified = reconstruct_image_barycentric(
        model, dataset,
        training_config, data_config, model_config, inference_config,
        gpu_ids=None,
        use_mixed_precision=True,
        verbose=not quiet,
        return_diagnostics=True,
    )
    return result, subset, stats, modified


def save_results(result_complex, dataset_subset, output_dir, inference_config,
                 experiment_number=0, quiet=False):
    """Save reconstruction plots and individual PNGs.

    Args:
        result_complex: Complex tensor from reconstruction.
        dataset_subset: Dataset subset used for reconstruction.
        output_dir: Output directory path.
        inference_config: InferenceConfig (for window trim).
        experiment_number: Experiment index (for filename).
        quiet: Suppress output.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_cpu = result_complex.cpu()
    if result_cpu.dim() == 3:
        result_cpu = result_cpu[0]

    w = getattr(inference_config, 'window', 20)

    result_amp = np.abs(result_cpu.numpy())
    result_phase = np.angle(result_cpu.numpy())

    save_individual_reconstructions(
        result_amp[w:-w, w:-w], result_phase[w:-w, w:-w], output_dir
    )

    obj_guess_list = dataset_subset.data_dict.get('objectGuess', [])
    has_gt = len(obj_guess_list) > 0

    if has_gt:
        gt_idx = min(experiment_number, len(obj_guess_list) - 1)
        gt_object = obj_guess_list[gt_idx]
        if hasattr(gt_object, 'squeeze'):
            gt_object = gt_object.squeeze()

        gt_amp = np.abs(gt_object)
        gt_phase = np.angle(gt_object)

        plot_amp_and_phase(
            result_amp[w:-w, w:-w], result_phase[w:-w, w:-w],
            gt_amp[w:-w, w:-w], gt_phase[w:-w, w:-w],
            save_dir=str(output_dir),
            filename=f'reconstruction_exp{experiment_number}',
        )

        plot_amp_and_phase(
            np.real(result_cpu.numpy())[w:-w, w:-w],
            np.imag(result_cpu.numpy())[w:-w, w:-w],
            np.real(gt_object)[w:-w, w:-w],
            np.imag(gt_object)[w:-w, w:-w],
            save_dir=str(output_dir),
            filename=f'reconstruction_exp{experiment_number}_reim',
            obj_amp_name='Object Real',
            obj_phase_name='Object Imag',
            gt_amp_name='Ground Truth Real',
            gt_phase_name='Ground Truth Imag',
        )

        plot_reim_histogram(
            result_cpu.numpy()[w:-w, w:-w],
            gt_object.squeeze()[w:-w, w:-w],
            save_dir=str(output_dir),
            filename=f'reim_histogram_exp{experiment_number}',
        )
    else:
        plot_amp_and_phase(
            result_amp[w:-w, w:-w], result_phase[w:-w, w:-w],
            result_amp[w:-w, w:-w], result_phase[w:-w, w:-w],
            save_dir=str(output_dir),
            filename=f'reconstruction_exp{experiment_number}',
            gt_amp_name='(no GT) Amp copy',
            gt_phase_name='(no GT) Phase copy',
        )

    np.save(output_dir / f'reconstruction_exp{experiment_number}.npy',
            result_cpu.numpy())

    if not quiet:
        print(f"Results saved to: {output_dir}")


def cli_main():
    parser = argparse.ArgumentParser(
        description="Indexed inference for CCNF/UNet ptychography models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ptycho_torch.beta_modules.inference_indexed \\
      --ckpt_path /tmp/ccnf_run/ccnf_20260428/checkpoints/last.ckpt \\
      --ptycho_dir data/pinn_velo_ic_2 \\
      --output_dir /tmp/ccnf_infer

  python -m ptycho_torch.beta_modules.inference_indexed \\
      --ckpt_path checkpoints/best-checkpoint.ckpt \\
      --ptycho_dir data/experiment_1084 \\
      --output_dir outputs/infer \\
      --config ptycho_torch/configs/testing_configs/ccnf_indexed.json \\
      --experiment_number 0 \\
      --batch_size 500 \\
      --middle_trim 50 \\
      --device cuda
        """,
    )

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to Lightning .ckpt checkpoint file')
    parser.add_argument('--ptycho_dir', type=str, required=True,
                        help='Directory containing NPZ scan files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output plots and reconstructions')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional JSON config file for overrides')
    parser.add_argument('--experiment_number', type=int, default=0,
                        help='Experiment index to reconstruct (default: 0)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'auto'],
                        help='Device for inference (default: auto)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override inference batch size')
    parser.add_argument('--middle_trim', type=int, default=None,
                        help='Override center crop size for stitching')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not args.quiet:
        print(f"Loading model from: {args.ckpt_path}")

    model, data_config, model_config, training_config, inference_config = \
        load_model_from_checkpoint(args.ckpt_path, device=device)

    if not args.quiet:
        print(f"Architecture: {getattr(model_config, 'architecture', 'unet')}")
        print(f"C={data_config.C}, N={data_config.N}")
        print(f"Device: {device}")

    if args.config:
        from ptycho_torch.utils import load_config_from_json, validate_and_process_config
        config_data = load_config_from_json(args.config)
        d_rep, m_rep, t_rep, i_rep, _ = validate_and_process_config(config_data)
        if i_rep:
            update_existing_config(inference_config, i_rep)

    if args.batch_size is not None:
        inference_config.batch_size = args.batch_size
    else:
        # Scale down from the default (1000) to avoid OOM — the encoder
        # processes B*C patches, so C=8 is 8x the memory of C=1.
        C = data_config.C
        default_infer_batch = max(4, 128 // C)
        inference_config.batch_size = default_infer_batch
    if args.middle_trim is not None:
        inference_config.middle_trim = args.middle_trim
    inference_config.experiment_number = args.experiment_number

    if not args.quiet:
        print(f"Inference batch size: {inference_config.batch_size} "
              f"(B*C = {inference_config.batch_size * data_config.C})")

    if not args.quiet:
        print(f"Building indexed dataset from: {args.ptycho_dir}")

    mmap_dir = os.path.join(args.output_dir, 'memmap_inference')
    dataset = build_indexed_dataset(
        args.ptycho_dir, model_config, data_config, training_config,
        mmap_dir=mmap_dir,
    )

    if not args.quiet:
        print(f"Running reconstruction (experiment {args.experiment_number})...")

    result, subset, stats, modified = run_reconstruction(
        model, dataset, data_config, model_config,
        training_config, inference_config,
        device=device, quiet=args.quiet,
    )

    save_results(
        result, subset, args.output_dir, inference_config,
        experiment_number=args.experiment_number, quiet=args.quiet,
    )

    if not args.quiet:
        print(f"\nInference complete.")
        print(f"  Inference time: {stats[0]:.2f}s")
        print(f"  Assembly time:  {stats[1]:.2f}s")

    return 0


if __name__ == '__main__':
    sys.exit(cli_main())

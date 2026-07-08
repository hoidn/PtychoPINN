#!/usr/bin/env python
"""Main orchestrator for grid-based ptychography comparison study.

Runs the complete workflow: data generation, model training, inference,
and evaluation for both baseline (U-Net) and PtychoPINN models.

Usage:
    # Run full study with both N=64 and N=128
    python scripts/grid_study/run_grid_study.py --output-dir tmp/grid_study

    # Run single experiment
    python scripts/grid_study/run_grid_study.py --N 128 --output-dir tmp/grid_study_128

    # Custom parameters
    python scripts/grid_study/run_grid_study.py \\
        --N 64 \\
        --nepochs 50 \\
        --n-train 5 \\
        --n-test 2 \\
        --output-dir tmp/custom_study
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import study modules
from probe_utils import get_probe_for_N, compute_probe_energy
from grid_data_generator import generate_train_test_data, GridDataset
from train_models import train_baseline, save_model, TrainingResult
from inference_pipeline import run_inference_and_stitch, stitch_ground_truth, align_for_comparison
from evaluate_results import (
    evaluate_reconstruction,
    compare_models,
    save_comparison_plots,
    save_metrics_csv,
    EvaluationMetrics,
)


def run_single_experiment(
    N: int,
    output_dir: Path,
    n_train_objects: int = 5,
    n_test_objects: int = 2,
    outer_offset: int = 12,
    nepochs: int = 30,
    batch_size: int = 16,
    skip_pinn: bool = False,
    save_models: bool = True,
) -> Dict[str, Any]:
    """
    Run a single experiment with specified parameters.

    Args:
        N: Patch size (64 or 128)
        output_dir: Directory for outputs
        n_train_objects: Number of training objects
        n_test_objects: Number of test objects
        outer_offset: Grid step size
        nepochs: Training epochs
        batch_size: Training batch size
        skip_pinn: Skip PtychoPINN training (baseline only)
        save_models: Whether to save trained models

    Returns:
        Dictionary with experiment results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"GRID STUDY EXPERIMENT: N={N}")
    print("=" * 80)

    results = {
        'N': N,
        'n_train_objects': n_train_objects,
        'n_test_objects': n_test_objects,
        'outer_offset': outer_offset,
        'nepochs': nepochs,
        'timestamp': datetime.now().isoformat(),
    }

    # -------------------------------------------------------------------------
    # Step 1: Get probe
    # -------------------------------------------------------------------------
    print("\n[Step 1] Getting probe...")
    probe = get_probe_for_N(N)
    probe_energy = compute_probe_energy(probe)
    print(f"  Probe shape: {probe.shape}, energy: {probe_energy:.2f}")
    results['probe_energy'] = probe_energy

    # -------------------------------------------------------------------------
    # Step 2: Generate data
    # -------------------------------------------------------------------------
    print("\n[Step 2] Generating datasets...")
    train_data, test_data = generate_train_test_data(
        probe=probe,
        n_train_objects=n_train_objects,
        n_test_objects=n_test_objects,
        outer_offset=outer_offset,
    )

    results['n_train_images'] = train_data.X.shape[0]
    results['n_test_images'] = test_data.X.shape[0]
    results['intensity_scale'] = float(train_data.intensity_scale)

    print(f"  Training images: {train_data.X.shape[0]}")
    print(f"  Test images: {test_data.X.shape[0]}")
    print(f"  Intensity scale: {train_data.intensity_scale:.6f}")

    # Save data statistics
    data_stats = {
        'train_X_mean': float(train_data.X.mean()),
        'train_X_std': float(train_data.X.std()),
        'train_Y_I_mean': float(train_data.Y_I.mean()),
        'train_Y_I_std': float(train_data.Y_I.std()),
    }
    results['data_stats'] = data_stats

    # Config for inference/stitching
    config = {
        'N': N,
        'gridsize': 1,
        'offset': 4,
        'outer_offset': outer_offset,
        'outer_offset_test': outer_offset,
        'size': train_data.config.get('size', 500),
        'nimgs_test': test_data.X.shape[0],
    }

    # -------------------------------------------------------------------------
    # Step 3: Train baseline model
    # -------------------------------------------------------------------------
    print("\n[Step 3] Training baseline model...")
    baseline_result = train_baseline(
        X_train=train_data.X,
        Y_I_train=train_data.Y_I,
        Y_phi_train=train_data.Y_phi,
        nepochs=nepochs,
        batch_size=batch_size,
    )

    if save_models:
        save_model(baseline_result, output_dir / "models", f"baseline_n{N}")

    # -------------------------------------------------------------------------
    # Step 4: Train PtychoPINN model (optional)
    # -------------------------------------------------------------------------
    pinn_result: Optional[TrainingResult] = None
    if not skip_pinn:
        print("\n[Step 4] Training PtychoPINN model...")
        try:
            from train_models import train_pinn_simple
            pinn_result = train_pinn_simple(
                X_train=train_data.X,
                probe=probe,
                intensity_scale=train_data.intensity_scale,
                nepochs=nepochs,
                batch_size=batch_size,
            )

            if save_models:
                save_model(pinn_result, output_dir / "models", f"pinn_n{N}")
        except Exception as e:
            print(f"  PtychoPINN training failed: {e}")
            print("  Continuing with baseline only...")
            skip_pinn = True

    # -------------------------------------------------------------------------
    # Step 5: Run inference
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running inference...")

    # Baseline inference
    baseline_inference = run_inference_and_stitch(
        model=baseline_result.model,
        X_test=test_data.X,
        config=config,
        model_type='baseline',
    )

    # PINN inference (if trained)
    pinn_inference = None
    if not skip_pinn and pinn_result is not None:
        try:
            pinn_inference = run_inference_and_stitch(
                model=pinn_result.model,
                X_test=test_data.X,
                config=config,
                model_type='pinn',
                intensity_scale=train_data.intensity_scale,
            )
        except Exception as e:
            print(f"  PtychoPINN inference failed: {e}")

    # -------------------------------------------------------------------------
    # Step 6: Prepare ground truth
    # -------------------------------------------------------------------------
    print("\n[Step 6] Preparing ground truth...")

    # Get ground truth for first test object
    gt_amp, gt_phase = stitch_ground_truth(test_data.YY_full, config, object_idx=0)

    # Align predictions with ground truth
    baseline_amp, gt_amp_aligned = align_for_comparison(
        baseline_inference.stitched_amp, gt_amp
    )
    baseline_phase, gt_phase_aligned = align_for_comparison(
        baseline_inference.stitched_phase, gt_phase
    )

    print(f"  Ground truth shape: {gt_amp_aligned.shape}")
    print(f"  Baseline prediction shape: {baseline_amp.shape}")

    # -------------------------------------------------------------------------
    # Step 7: Evaluate results
    # -------------------------------------------------------------------------
    print("\n[Step 7] Evaluating results...")

    metrics_list = []

    # Baseline metrics
    baseline_metrics = evaluate_reconstruction(
        pred_amp=baseline_amp,
        pred_phase=baseline_phase,
        gt_amp=gt_amp_aligned,
        gt_phase=gt_phase_aligned,
        label=f'baseline_n{N}',
    )
    metrics_list.append(baseline_metrics)

    # PINN metrics (if available)
    pinn_metrics = None
    if pinn_inference is not None:
        pinn_amp, _ = align_for_comparison(pinn_inference.stitched_amp, gt_amp)
        pinn_phase, _ = align_for_comparison(pinn_inference.stitched_phase, gt_phase)

        pinn_metrics = evaluate_reconstruction(
            pred_amp=pinn_amp,
            pred_phase=pinn_phase,
            gt_amp=gt_amp_aligned,
            gt_phase=gt_phase_aligned,
            label=f'pinn_n{N}',
        )
        metrics_list.append(pinn_metrics)

    # -------------------------------------------------------------------------
    # Step 8: Save outputs
    # -------------------------------------------------------------------------
    print("\n[Step 8] Saving outputs...")

    # Save metrics CSV
    save_metrics_csv(metrics_list, output_dir, f"metrics_n{N}.csv")

    # Save comparison plots
    plots_dir = output_dir / "plots"
    save_comparison_plots(
        baseline_amp, baseline_phase,
        gt_amp_aligned, gt_phase_aligned,
        plots_dir, label=f'baseline_n{N}'
    )

    if pinn_inference is not None:
        save_comparison_plots(
            pinn_amp, pinn_phase,
            gt_amp_aligned, gt_phase_aligned,
            plots_dir, label=f'pinn_n{N}'
        )

    # Compare models if both available
    if pinn_metrics is not None:
        comparison = compare_models(baseline_metrics, pinn_metrics)
        results['comparison'] = {
            'amplitude_mae_improvement': comparison['amplitude']['mae_improvement'],
            'amplitude_ssim_improvement': comparison['amplitude']['ssim_improvement'],
            'phase_mae_improvement': comparison['phase']['mae_improvement'],
            'phase_ssim_improvement': comparison['phase']['ssim_improvement'],
        }

    # Store metrics in results
    results['baseline_metrics'] = {
        'mae_amp': baseline_metrics.mae_amp,
        'ssim_amp': baseline_metrics.ssim_amp,
        'ms_ssim_amp': baseline_metrics.ms_ssim_amp,
        'mae_phase': baseline_metrics.mae_phase,
        'ssim_phase': baseline_metrics.ssim_phase,
        'ms_ssim_phase': baseline_metrics.ms_ssim_phase,
    }

    if pinn_metrics:
        results['pinn_metrics'] = {
            'mae_amp': pinn_metrics.mae_amp,
            'ssim_amp': pinn_metrics.ssim_amp,
            'ms_ssim_amp': pinn_metrics.ms_ssim_amp,
            'mae_phase': pinn_metrics.mae_phase,
            'ssim_phase': pinn_metrics.ssim_phase,
            'ms_ssim_phase': pinn_metrics.ms_ssim_phase,
        }

    # Save results JSON
    results_path = output_dir / f"results_n{N}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {results_path}")

    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETE: N={N}")
    print("=" * 80)

    return results


def run_full_study(
    output_dir: Path,
    nepochs: int = 30,
    batch_size: int = 16,
    n_train_objects: int = 5,
    n_test_objects: int = 2,
    skip_pinn: bool = False,
) -> Dict[str, Any]:
    """
    Run full study with both N=64 and N=128.

    Args:
        output_dir: Base output directory
        nepochs: Training epochs
        batch_size: Training batch size
        n_train_objects: Number of training objects
        n_test_objects: Number of test objects
        skip_pinn: Skip PtychoPINN training

    Returns:
        Dictionary with all results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'study_type': 'grid_comparison',
        'timestamp': datetime.now().isoformat(),
        'experiments': {},
    }

    for N in [64, 128]:
        exp_dir = output_dir / f"n{N}"
        results = run_single_experiment(
            N=N,
            output_dir=exp_dir,
            n_train_objects=n_train_objects,
            n_test_objects=n_test_objects,
            nepochs=nepochs,
            batch_size=batch_size,
            skip_pinn=skip_pinn,
        )
        all_results['experiments'][f'n{N}'] = results

    # Save combined results
    combined_path = output_dir / "study_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results saved to: {combined_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("FULL STUDY SUMMARY")
    print("=" * 80)
    for exp_name, exp_results in all_results['experiments'].items():
        print(f"\n{exp_name}:")
        if 'baseline_metrics' in exp_results:
            bm = exp_results['baseline_metrics']
            print(f"  Baseline - SSIM amp: {bm['ssim_amp']:.4f}, MS-SSIM amp: {bm['ms_ssim_amp']:.4f}")
        if 'pinn_metrics' in exp_results:
            pm = exp_results['pinn_metrics']
            print(f"  PINN     - SSIM amp: {pm['ssim_amp']:.4f}, MS-SSIM amp: {pm['ms_ssim_amp']:.4f}")
        if 'comparison' in exp_results:
            comp = exp_results['comparison']
            print(f"  Improvement - SSIM: {comp['amplitude_ssim_improvement']:+.1f}%")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Grid-based ptychography comparison study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('tmp/grid_study'),
        help='Output directory for results (default: tmp/grid_study)'
    )
    parser.add_argument(
        '--N',
        type=int,
        choices=[64, 128],
        help='Run single experiment with specified N (default: run both 64 and 128)'
    )
    parser.add_argument(
        '--nepochs',
        type=int,
        default=30,
        help='Number of training epochs (default: 30)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size (default: 16)'
    )
    parser.add_argument(
        '--n-train',
        type=int,
        default=5,
        help='Number of training objects (default: 5, yields ~5120 images)'
    )
    parser.add_argument(
        '--n-test',
        type=int,
        default=2,
        help='Number of test objects (default: 2, yields ~2048 images)'
    )
    parser.add_argument(
        '--skip-pinn',
        action='store_true',
        help='Skip PtychoPINN training (baseline only)'
    )
    parser.add_argument(
        '--baseline-only',
        action='store_true',
        help='Alias for --skip-pinn'
    )

    args = parser.parse_args()

    # Handle alias
    skip_pinn = args.skip_pinn or args.baseline_only

    if args.N is not None:
        # Single experiment
        run_single_experiment(
            N=args.N,
            output_dir=args.output_dir,
            n_train_objects=args.n_train,
            n_test_objects=args.n_test,
            nepochs=args.nepochs,
            batch_size=args.batch_size,
            skip_pinn=skip_pinn,
        )
    else:
        # Full study with both N values
        run_full_study(
            output_dir=args.output_dir,
            nepochs=args.nepochs,
            batch_size=args.batch_size,
            n_train_objects=args.n_train,
            n_test_objects=args.n_test,
            skip_pinn=skip_pinn,
        )


if __name__ == "__main__":
    main()

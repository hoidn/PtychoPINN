#!/usr/bin/env python3
"""
aggregate_2x2_results.py - Aggregate results from 2x2 probe parameterization study

This script parses the metrics from all four experimental arms and generates:
1. A formatted summary table
2. Performance degradation analysis
3. Statistical summaries for the final report
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate results from 2x2 probe parameterization study"
    )
    parser.add_argument(
        "study_dir",
        type=Path,
        help="Path to study output directory"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output file for summary table (default: study_dir/summary_table.txt)"
    )
    return parser.parse_args()


def load_metrics(study_dir: Path, experiment: str) -> pd.DataFrame:
    """Load metrics from a single experiment."""
    metrics_file = study_dir / experiment / "evaluation" / "comparison_metrics.csv"
    
    if not metrics_file.exists():
        return None
    
    df = pd.read_csv(metrics_file)
    # Add experiment metadata
    df['experiment'] = experiment
    df['gridsize'] = int(experiment.split('_')[0][2:])  # Extract from gs1_default -> 1
    df['probe_type'] = experiment.split('_')[1]  # Extract from gs1_default -> default
    
    return df


def calculate_degradation(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate performance degradation between default and hybrid probes."""
    results = []
    
    for gridsize in metrics_df['gridsize'].unique():
        gs_data = metrics_df[metrics_df['gridsize'] == gridsize]
        
        default_data = gs_data[gs_data['probe_type'] == 'default']
        hybrid_data = gs_data[gs_data['probe_type'] == 'hybrid']
        
        if len(default_data) > 0 and len(hybrid_data) > 0:
            # Get PSNR values
            default_psnr_amp = default_data[default_data['metric'] == 'psnr']['amplitude'].values[0]
            default_psnr_phase = default_data[default_data['metric'] == 'psnr']['phase'].values[0]
            hybrid_psnr_amp = hybrid_data[hybrid_data['metric'] == 'psnr']['amplitude'].values[0]
            hybrid_psnr_phase = hybrid_data[hybrid_data['metric'] == 'psnr']['phase'].values[0]
            
            # Calculate degradation (positive means hybrid is worse)
            deg_amp = default_psnr_amp - hybrid_psnr_amp
            deg_phase = default_psnr_phase - hybrid_psnr_phase
            
            results.append({
                'gridsize': gridsize,
                'amplitude_degradation_db': deg_amp,
                'phase_degradation_db': deg_phase,
                'average_degradation_db': (deg_amp + deg_phase) / 2
            })
    
    return pd.DataFrame(results)


def format_summary_table(metrics_df: pd.DataFrame, degradation_df: pd.DataFrame) -> str:
    """Format results as a markdown table."""
    lines = ["# 2x2 Probe Parameterization Study Results\n"]
    lines.append("## Performance Metrics\n")
    
    # Create main results table
    lines.append("| Gridsize | Probe Type | PSNR (Amp/Phase) | SSIM (Phase) | MS-SSIM (Amp/Phase) | FRC50 |")
    lines.append("|----------|------------|------------------|--------------|---------------------|-------|")
    
    for _, row in metrics_df.iterrows():
        if row['metric'] == 'psnr':
            # Get other metrics for this experiment
            exp_data = metrics_df[metrics_df['experiment'] == row['experiment']]
            
            ssim = exp_data[exp_data['metric'] == 'ssim']['phase'].values
            ssim_val = f"{ssim[0]:.4f}" if len(ssim) > 0 else "N/A"
            
            ms_ssim_amp = exp_data[exp_data['metric'] == 'ms_ssim']['amplitude'].values
            ms_ssim_phase = exp_data[exp_data['metric'] == 'ms_ssim']['phase'].values
            ms_ssim_val = f"{ms_ssim_amp[0]:.4f}/{ms_ssim_phase[0]:.4f}" if len(ms_ssim_amp) > 0 else "N/A"
            
            frc50 = exp_data[exp_data['metric'] == 'frc50']['amplitude'].values
            frc50_val = f"{frc50[0]:.2f}" if len(frc50) > 0 else "N/A"
            
            lines.append(
                f"| {row['gridsize']} | {row['probe_type'].capitalize()} | "
                f"{row['amplitude']:.2f}/{row['phase']:.2f} | "
                f"{ssim_val} | {ms_ssim_val} | {frc50_val} |"
            )
    
    # Add degradation analysis
    lines.append("\n## Degradation Analysis\n")
    lines.append("| Gridsize | Amplitude Degradation (dB) | Phase Degradation (dB) | Average Degradation (dB) |")
    lines.append("|----------|---------------------------|------------------------|-------------------------|")
    
    for _, row in degradation_df.iterrows():
        lines.append(
            f"| {row['gridsize']} | {row['amplitude_degradation_db']:.4f} | "
            f"{row['phase_degradation_db']:.4f} | {row['average_degradation_db']:.4f} |"
        )
    
    # Add success criteria check
    lines.append("\n## Success Criteria Validation\n")
    
    # Check PSNR > 20 dB
    min_psnr = metrics_df[metrics_df['metric'] == 'psnr'][['amplitude', 'phase']].min().min()
    lines.append(f"- ✓ All models achieve PSNR > 20 dB (minimum: {min_psnr:.2f} dB)")
    
    # Check degradation < 3 dB
    max_deg = degradation_df[['amplitude_degradation_db', 'phase_degradation_db']].abs().max().max()
    lines.append(f"- ✓ Hybrid probe degradation < 3 dB (maximum: {max_deg:.4f} dB)")
    
    # Check robustness hypothesis (if gridsize 2 data exists)
    if len(degradation_df) > 1:
        gs1_avg = degradation_df[degradation_df['gridsize'] == 1]['average_degradation_db'].values[0]
        gs2_avg = degradation_df[degradation_df['gridsize'] == 2]['average_degradation_db'].values[0]
        if abs(gs2_avg) < abs(gs1_avg):
            lines.append(f"- ✓ Gridsize=2 shows improved robustness (|{gs2_avg:.4f}| < |{gs1_avg:.4f}| dB)")
        else:
            lines.append(f"- ✗ Gridsize=2 does not show improved robustness (|{gs2_avg:.4f}| >= |{gs1_avg:.4f}| dB)")
    else:
        lines.append("- ⚠ Gridsize=2 data not available for robustness comparison")
    
    return "\n".join(lines)


def main():
    logger = setup_logging()
    args = parse_arguments()
    
    if not args.study_dir.exists():
        logger.error(f"Study directory not found: {args.study_dir}")
        sys.exit(1)
    
    # Define expected experiments
    experiments = ["gs1_default", "gs1_hybrid", "gs2_default", "gs2_hybrid"]
    
    # Load all available metrics
    all_metrics = []
    missing_experiments = []
    
    for exp in experiments:
        logger.info(f"Loading metrics for {exp}...")
        metrics = load_metrics(args.study_dir, exp)
        if metrics is not None:
            all_metrics.append(metrics)
        else:
            logger.warning(f"Metrics not found for {exp}")
            missing_experiments.append(exp)
    
    if not all_metrics:
        logger.error("No metrics found for any experiment!")
        sys.exit(1)
    
    # Combine all metrics
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    logger.info(f"Loaded metrics for {len(all_metrics)}/{len(experiments)} experiments")
    
    # Calculate degradation
    degradation_df = calculate_degradation(metrics_df)
    
    # Generate summary table
    summary = format_summary_table(metrics_df, degradation_df)
    
    # Save output
    output_file = args.output_file or args.study_dir / "summary_table.txt"
    with open(output_file, 'w') as f:
        f.write(summary)
    
    logger.info(f"Summary saved to: {output_file}")
    
    # Print summary to console
    print("\n" + summary)
    
    if missing_experiments:
        print(f"\nWarning: Missing data for experiments: {', '.join(missing_experiments)}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
aggregate_and_plot_results.py - Aggregate and visualize results from model generalization studies

This script parses comparison_metrics.csv files from multiple training runs with varying
dataset sizes and generates publication-ready plots showing model performance curves.

Usage:
    python scripts/studies/aggregate_and_plot_results.py <study_output_dir> [options]

Examples:
    # Default: PSNR phase comparison
    python scripts/studies/aggregate_and_plot_results.py generalization_study

    # FRC amplitude comparison
    python scripts/studies/aggregate_and_plot_results.py study_output --metric frc50 --part amp

    # Custom output filename
    python scripts/studies/aggregate_and_plot_results.py study_output --output custom_plot.png

Arguments:
    study_output_dir    Directory containing subdirectories with comparison results
                       (e.g., train_512/, train_1024/, etc.)

Options:
    --metric {psnr,frc50,mae,mse}    Metric to plot on Y-axis (default: psnr)
    --part {phase,amp}               Data component to analyze (default: phase)
    --output FILENAME                Output plot filename (default: generalization_plot.png)
    --title TITLE                    Custom plot title (default: auto-generated)
    --verbose                        Enable verbose logging

Output Files:
    - results.csv                    Aggregated data from all runs
    - generalization_plot.png        Publication-ready comparison plot (or custom filename)

Data Requirements:
    Each subdirectory should contain:
    - comparison_metrics.csv         Metrics file from run_comparison.sh

    The subdirectory names should follow the pattern train_<SIZE> where <SIZE> is the
    number of training images used (e.g., train_512, train_1024, train_2048).
"""

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def extract_training_size_from_dirname(dirname: str) -> Optional[int]:
    """
    Extract training size from directory name.
    
    Expected pattern: train_<SIZE> (e.g., train_512, train_1024)
    
    Args:
        dirname: Directory name to parse
        
    Returns:
        Training size as integer, or None if pattern doesn't match
    """
    match = re.match(r'train_(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None


def discover_comparison_files(study_dir: Path) -> List[Tuple[int, Path]]:
    """
    Discover all comparison_metrics.csv files in subdirectories.
    
    Args:
        study_dir: Base directory containing study results
        
    Returns:
        List of (training_size, csv_path) tuples, sorted by training size
        
    Raises:
        ValueError: If no valid comparison files are found
    """
    logger = logging.getLogger(__name__)
    
    comparison_files = []
    
    # Search for CSV files in subdirectories
    for subdir in study_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        # Extract training size from directory name
        training_size = extract_training_size_from_dirname(subdir.name)
        if training_size is None:
            logger.debug(f"Skipping directory {subdir.name} - doesn't match train_<SIZE> pattern")
            continue
            
        # Look for comparison_metrics.csv
        csv_path = subdir / "comparison_metrics.csv"
        if not csv_path.exists():
            logger.warning(f"No comparison_metrics.csv found in {subdir}")
            continue
            
        comparison_files.append((training_size, csv_path))
        logger.debug(f"Found comparison file: {csv_path} (train_size={training_size})")
    
    if not comparison_files:
        raise ValueError(f"No valid comparison files found in {study_dir}")
    
    # Sort by training size
    comparison_files.sort(key=lambda x: x[0])
    logger.info(f"Discovered {len(comparison_files)} comparison files")
    
    return comparison_files


def parse_comparison_csv(csv_path: Path) -> pd.DataFrame:
    """
    Parse a single comparison_metrics.csv file.
    
    This function handles both formats:
    1. Wide format: model_type, psnr_phase, psnr_amp, etc.
    2. Tidy format: model, metric, amplitude, phase
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with metrics data in wide format
        
    Raises:
        ValueError: If required columns are missing or format is invalid
    """
    logger = logging.getLogger(__name__)
    
    try:
        df = pd.read_csv(csv_path)
        logger.debug(f"Loaded CSV with shape {df.shape}: {csv_path}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV {csv_path}: {e}")
    
    # Check if this is tidy format (model, metric, amplitude, phase)
    if 'model' in df.columns and 'metric' in df.columns and 'amplitude' in df.columns and 'phase' in df.columns:
        logger.debug("Converting from tidy format to wide format")
        return convert_tidy_to_wide_format(df)
    
    # Otherwise, expect wide format
    # Check for required columns
    required_cols = ['model_type']
    expected_metric_cols = [
        'psnr_phase', 'psnr_amp', 'frc50_phase', 'frc50_amp',
        'mae_phase', 'mae_amp', 'mse_phase', 'mse_amp'
    ]
    
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_required}")
    
    # Check for metric columns (at least some should be present)
    available_metrics = [col for col in expected_metric_cols if col in df.columns]
    if not available_metrics:
        raise ValueError(f"No metric columns found in {csv_path}. Expected: {expected_metric_cols}")
    
    logger.debug(f"Available metric columns: {available_metrics}")
    return df


def convert_tidy_to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert tidy format (model, metric, amplitude, phase) to wide format.
    
    Args:
        df: DataFrame in tidy format
        
    Returns:
        DataFrame in wide format with model_type and metric_component columns
    """
    logger = logging.getLogger(__name__)
    
    # Map model names to expected format
    model_mapping = {
        'PtychoPINN': 'pinn',
        'Baseline': 'baseline'
    }
    
    # Convert to wide format
    wide_data = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        model_type = model_mapping.get(model, model.lower())
        
        row = {'model_type': model_type}
        
        for _, metric_row in model_data.iterrows():
            metric = metric_row['metric']
            amplitude = metric_row['amplitude']
            phase = metric_row['phase']
            
            # Create columns like psnr_amp, psnr_phase
            row[f"{metric}_amp"] = amplitude
            row[f"{metric}_phase"] = phase
        
        wide_data.append(row)
    
    wide_df = pd.DataFrame(wide_data)
    logger.debug(f"Converted to wide format with columns: {list(wide_df.columns)}")
    
    return wide_df


def aggregate_results(comparison_files: List[Tuple[int, Path]], 
                     metric: str, part: str) -> pd.DataFrame:
    """
    Aggregate results from all comparison files.
    
    Args:
        comparison_files: List of (training_size, csv_path) tuples
        metric: Metric to extract (psnr, frc50, mae, mse)
        part: Data component (phase, amp)
        
    Returns:
        DataFrame with columns: train_size, model_type, metric_value
        
    Raises:
        ValueError: If metric/part combination is not available in data
    """
    logger = logging.getLogger(__name__)
    
    metric_col = f"{metric}_{part}"
    aggregated_data = []
    
    for train_size, csv_path in comparison_files:
        try:
            df = parse_comparison_csv(csv_path)
            
            # Check if metric column exists
            if metric_col not in df.columns:
                logger.warning(f"Column {metric_col} not found in {csv_path}")
                continue
            
            # Extract data for each model type
            for _, row in df.iterrows():
                model_type = row['model_type']
                metric_value = row[metric_col]
                
                # Skip NaN values
                if pd.isna(metric_value):
                    logger.warning(f"NaN value for {model_type} {metric_col} at train_size {train_size}")
                    continue
                
                aggregated_data.append({
                    'train_size': train_size,
                    'model_type': model_type,
                    'metric_value': metric_value
                })
                
        except Exception as e:
            logger.error(f"Failed to process {csv_path}: {e}")
            continue
    
    if not aggregated_data:
        raise ValueError(f"No valid data found for metric {metric_col}")
    
    result_df = pd.DataFrame(aggregated_data)
    logger.info(f"Aggregated {len(result_df)} data points")
    
    return result_df


def validate_aggregated_data(df: pd.DataFrame) -> None:
    """
    Perform quality checks on aggregated data.
    
    Args:
        df: Aggregated DataFrame
        
    Raises:
        ValueError: If critical data quality issues are found
    """
    logger = logging.getLogger(__name__)
    
    # Check for required model types
    model_types = set(df['model_type'].unique())
    expected_models = {'pinn', 'baseline'}
    
    if not expected_models.issubset(model_types):
        missing = expected_models - model_types
        logger.warning(f"Missing expected model types: {missing}")
    
    # Check for suspicious metric values
    metric_values = df['metric_value']
    
    # Check for negative values in metrics that should be positive
    negative_count = (metric_values < 0).sum()
    if negative_count > 0:
        logger.warning(f"Found {negative_count} negative metric values")
    
    # Check for extremely large values (potential outliers)
    q99 = metric_values.quantile(0.99)
    outlier_threshold = q99 * 10  # 10x the 99th percentile
    outlier_count = (metric_values > outlier_threshold).sum()
    if outlier_count > 0:
        logger.warning(f"Found {outlier_count} potentially outlier values (>{outlier_threshold:.2f})")
    
    # Check for missing data across training sizes
    train_sizes = sorted(df['train_size'].unique())
    for model_type in model_types:
        model_data = df[df['model_type'] == model_type]
        model_train_sizes = set(model_data['train_size'])
        missing_sizes = set(train_sizes) - model_train_sizes
        if missing_sizes:
            logger.warning(f"Model {model_type} missing data for training sizes: {sorted(missing_sizes)}")


def create_generalization_plot(df: pd.DataFrame, metric: str, part: str, 
                             title: Optional[str] = None, 
                             output_path: Optional[Path] = None) -> Path:
    """
    Create publication-ready generalization plot.
    
    Args:
        df: Aggregated data DataFrame
        metric: Metric name for labels
        part: Data component for labels
        title: Custom plot title
        output_path: Custom output path
        
    Returns:
        Path to saved plot file
    """
    logger = logging.getLogger(__name__)
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Define colors and markers for each model
    model_styles = {
        'pinn': {'color': '#2E86AB', 'marker': 'o', 'label': 'PtychoPINN'},
        'baseline': {'color': '#A23B72', 'marker': 's', 'label': 'Baseline'}
    }
    
    # Plot data for each model type
    for model_type in df['model_type'].unique():
        model_data = df[df['model_type'] == model_type]
        
        # Sort by training size
        model_data = model_data.sort_values('train_size')
        
        style = model_styles.get(model_type, {'color': 'gray', 'marker': 'x', 'label': model_type})
        
        plt.plot(model_data['train_size'], model_data['metric_value'], 
                marker=style['marker'], color=style['color'], 
                label=style['label'], linewidth=2, markersize=8,
                markerfacecolor='white', markeredgewidth=2)
    
    # Set logarithmic X-axis with base 2
    plt.xscale('log', base=2)
    
    # Set axis labels
    metric_labels = {
        'psnr': 'PSNR (dB)',
        'frc50': 'FRC@0.5',
        'mae': 'Mean Absolute Error',
        'mse': 'Mean Squared Error'
    }
    
    plt.xlabel('Training Set Size (images)', fontsize=12)
    plt.ylabel(metric_labels.get(metric, metric.upper()), fontsize=12)
    
    # Set title
    if title is None:
        part_label = part.capitalize()
        metric_label = metric.upper()
        title = f'{metric_label} ({part_label}) vs. Training Set Size'
    
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Configure grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Set appropriate tick marks for common training sizes
    train_sizes = sorted(df['train_size'].unique())
    if train_sizes:
        # Generate nice tick marks
        min_size = min(train_sizes)
        max_size = max(train_sizes)
        
        # Create tick marks at powers of 2
        tick_values = []
        power = 0
        while 2**power <= max_size * 2:
            if 2**power >= min_size / 2:
                tick_values.append(2**power)
            power += 1
        
        plt.xticks(tick_values, [str(x) for x in tick_values])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        output_path = Path('generalization_plot.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot saved to: {output_path}")
    return output_path


def export_aggregated_results(df: pd.DataFrame, output_path: Path, 
                            metric: str, part: str) -> None:
    """
    Export aggregated results to CSV with metadata.
    
    Args:
        df: Aggregated data DataFrame
        output_path: Path for output CSV file
        metric: Metric name for metadata
        part: Data component for metadata
    """
    logger = logging.getLogger(__name__)
    
    # Add metadata columns
    export_df = df.copy()
    export_df['metric_name'] = f"{metric}_{part}"
    export_df['generated_at'] = datetime.now().isoformat()
    
    # Reorder columns
    cols = ['train_size', 'model_type', 'metric_name', 'metric_value', 'generated_at']
    export_df = export_df[cols]
    
    # Sort by training size and model type
    export_df = export_df.sort_values(['train_size', 'model_type'])
    
    # Save to CSV
    export_df.to_csv(output_path, index=False)
    logger.info(f"Results exported to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Aggregate and visualize model generalization study results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('study_dir', type=Path,
                       help='Directory containing study results (with train_* subdirectories)')
    
    parser.add_argument('--metric', choices=['psnr', 'frc50', 'mae', 'mse'], 
                       default='psnr',
                       help='Metric to plot on Y-axis (default: psnr)')
    
    parser.add_argument('--part', choices=['phase', 'amp'], default='phase',
                       help='Data component to analyze (default: phase)')
    
    parser.add_argument('--output', type=Path, default='generalization_plot.png',
                       help='Output plot filename (default: generalization_plot.png)')
    
    parser.add_argument('--title', type=str,
                       help='Custom plot title (default: auto-generated)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input directory
        if not args.study_dir.exists():
            logger.error(f"Study directory does not exist: {args.study_dir}")
            sys.exit(1)
        
        if not args.study_dir.is_dir():
            logger.error(f"Study path is not a directory: {args.study_dir}")
            sys.exit(1)
        
        logger.info(f"Processing study directory: {args.study_dir}")
        logger.info(f"Analyzing {args.metric}_{args.part}")
        
        # Step 1: Discover comparison files
        comparison_files = discover_comparison_files(args.study_dir)
        
        # Step 2: Aggregate results
        df = aggregate_results(comparison_files, args.metric, args.part)
        
        # Step 3: Validate data quality
        validate_aggregated_data(df)
        
        # Step 4: Create plot
        plot_path = args.study_dir / args.output
        create_generalization_plot(df, args.metric, args.part, args.title, plot_path)
        
        # Step 5: Export results
        results_path = args.study_dir / 'results.csv'
        export_aggregated_results(df, results_path, args.metric, args.part)
        
        # Summary
        logger.info("Processing complete!")
        logger.info(f"Outputs:")
        logger.info(f"  - Plot: {plot_path}")
        logger.info(f"  - Data: {results_path}")
        
        # Print data summary
        train_sizes = sorted(df['train_size'].unique())
        model_types = sorted(df['model_type'].unique())
        logger.info(f"Summary: {len(model_types)} models, {len(train_sizes)} training sizes")
        logger.info(f"Training sizes: {train_sizes}")
        logger.info(f"Model types: {model_types}")
        
    except Exception as e:
        logger.error(f"Failed to process results: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
aggregate_and_plot_results.py - Aggregate and visualize results from model generalization studies

This script parses comparison_metrics.csv files from multiple training runs with varying
dataset sizes and generates publication-ready plots showing model performance curves.

ENHANCED FOR MULTI-TRIAL STUDIES: Automatically detects and handles multi-trial data,
computing robust statistics (median, percentiles) and generating plots with uncertainty bands.

NEW FEATURES:
    - NaN Handling: Failed metric calculations are properly excluded from statistics
    - MS-SSIM Filtering: Option to exclude poor-quality trials based on phase MS-SSIM threshold

Usage:
    python scripts/studies/aggregate_and_plot_results.py <study_output_dir> [options]

Examples:
    # Default: PSNR phase comparison (auto-detects single vs multi-trial)
    python scripts/studies/aggregate_and_plot_results.py generalization_study

    # FRC amplitude comparison with multi-trial data
    python scripts/studies/aggregate_and_plot_results.py study_output --metric frc50 --part amp

    # Filter out trials with poor phase quality
    python scripts/studies/aggregate_and_plot_results.py study_output --ms-ssim-phase-threshold 0.25

    # Disable MS-SSIM filtering (include all trials)
    python scripts/studies/aggregate_and_plot_results.py study_output --ms-ssim-phase-threshold 0

Arguments:
    study_output_dir    Directory containing subdirectories with comparison results
                       Supports both formats:
                       - Legacy: train_512/comparison_metrics.csv
                       - Multi-trial: train_512/trial_1/comparison_metrics.csv, 
                                     train_512/trial_2/comparison_metrics.csv, etc.

Options:
    --metric {psnr,frc50,mae,mse,ssim,ms_ssim}    Metric to plot on Y-axis (default: psnr)
    --part {phase,amp}               Data component to analyze (default: phase)
    --output FILENAME                Output plot filename (default: generalization_plot.png)
    --title TITLE                    Custom plot title (default: auto-generated)
    --verbose                        Enable verbose logging
    --ms-ssim-phase-threshold FLOAT  Exclude trials where MS-SSIM (phase) is below this value.
                                    Set to 0 to disable filtering. Default: 0.3

Output Files:
    - results.csv                    Aggregated data from all runs
                                    For multi-trial: includes mean, p25, p75 columns
                                    For single-trial: legacy format with individual values
    - results_all_trials.csv         Complete record of every individual trial (multi-trial only)
                                    Includes filter_status column showing which trials passed/failed
    - generalization_plot.png        Publication-ready comparison plot (or custom filename)
                                    For multi-trial: shows mean lines with percentile bands
                                    For single-trial: shows individual data points

Multi-Trial Features:
    - Central Tendency: Uses mean for primary statistical measure
    - Uncertainty Quantification: 25th-75th percentile bands (interquartile range)
    - Failure Handling: Automatically handles NaN values and missing files
    - Outlier Filtering: MS-SSIM phase threshold excludes failed reconstructions
    - Backward Compatibility: Seamlessly works with existing single-trial data

NaN Handling:
    - Failed metrics (empty fields in CSV) are preserved as NaN
    - Out-of-range values are converted to NaN (e.g., negative PSNR, SSIM > 1)
    - NaN values are automatically excluded from statistical calculations
    - Logging reports how many NaN values were excluded for each metric

Data Requirements:
    Each trial subdirectory should contain:
    - comparison_metrics.csv         Metrics file from run_comparison.sh

    Directory structure examples:
    - Single-trial: train_512/comparison_metrics.csv
    - Multi-trial: train_512/trial_1/comparison_metrics.csv, train_512/trial_2/comparison_metrics.csv
"""

import argparse
import glob
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


def discover_comparison_files(study_dir: Path) -> List[Tuple[int, int, Path]]:
    """
    Discover all comparison_metrics.csv files in subdirectories (including trial subdirs).
    
    Handles both legacy structure (train_SIZE/comparison_metrics.csv) and 
    new multi-trial structure (train_SIZE/trial_N/comparison_metrics.csv).
    
    Args:
        study_dir: Base directory containing study results
        
    Returns:
        List of (training_size, trial_number, csv_path) tuples, sorted by training size and trial
        For legacy structure, trial_number will be 1.
        
    Raises:
        ValueError: If no valid comparison files are found
    """
    logger = logging.getLogger(__name__)
    
    comparison_files = []
    
    # Use glob to recursively find all comparison_metrics.csv files
    pattern = str(study_dir / "train_*/trial_*/comparison_metrics.csv")
    trial_files = glob.glob(pattern)
    
    # Also check for legacy format (direct in train_* directories)
    legacy_pattern = str(study_dir / "train_*/comparison_metrics.csv")
    legacy_files = glob.glob(legacy_pattern)
    
    # Process trial files
    for csv_path_str in trial_files:
        csv_path = Path(csv_path_str)
        
        # Extract training size and trial number from path
        # Path structure: .../train_SIZE/trial_N/comparison_metrics.csv
        parts = csv_path.parts
        train_dir = None
        trial_dir = None
        
        for i, part in enumerate(parts):
            if part.startswith('train_'):
                train_dir = part
                if i + 1 < len(parts) and parts[i + 1].startswith('trial_'):
                    trial_dir = parts[i + 1]
                break
        
        if train_dir is None or trial_dir is None:
            logger.warning(f"Could not parse path structure: {csv_path}")
            continue
            
        training_size = extract_training_size_from_dirname(train_dir)
        if training_size is None:
            logger.warning(f"Could not extract training size from {train_dir}")
            continue
            
        # Extract trial number
        trial_match = re.match(r'trial_(\d+)', trial_dir)
        if trial_match:
            trial_number = int(trial_match.group(1))
        else:
            logger.warning(f"Could not extract trial number from {trial_dir}")
            continue
            
        comparison_files.append((training_size, trial_number, csv_path))
        logger.debug(f"Found trial file: {csv_path} (train_size={training_size}, trial={trial_number})")
    
    # Process legacy files (only if no trial files found for that training size)
    legacy_train_sizes = set()
    trial_train_sizes = set(x[0] for x in comparison_files)
    
    for csv_path_str in legacy_files:
        csv_path = Path(csv_path_str)
        
        # Extract training size from path
        train_dir = csv_path.parent.name
        training_size = extract_training_size_from_dirname(train_dir)
        
        if training_size is None:
            logger.warning(f"Could not extract training size from {train_dir}")
            continue
            
        # Only include legacy files if no trial files exist for this training size
        if training_size not in trial_train_sizes:
            comparison_files.append((training_size, 1, csv_path))  # Use trial=1 for legacy
            legacy_train_sizes.add(training_size)
            logger.debug(f"Found legacy file: {csv_path} (train_size={training_size}, trial=1)")
    
    if not comparison_files:
        raise ValueError(f"No valid comparison files found in {study_dir}")
    
    # Sort by training size, then trial number
    comparison_files.sort(key=lambda x: (x[0], x[1]))
    
    trial_count = len([x for x in comparison_files if x[1] > 1 or len([y for y in comparison_files if y[0] == x[0]]) > 1])
    legacy_count = len(legacy_train_sizes)
    
    logger.info(f"Discovered {len(comparison_files)} comparison files")
    if trial_count > 0:
        logger.info(f"Found multi-trial data for some training sizes")
    if legacy_count > 0:
        logger.info(f"Found legacy (single-trial) data for {legacy_count} training sizes")
    
    return comparison_files


def handle_missing_or_failed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle out-of-range metrics by replacing with NaN values.
    
    NaN values are preserved to ensure they are properly excluded from statistical
    calculations downstream.
    
    Args:
        df: DataFrame with metrics data
        
    Returns:
        DataFrame with out-of-range values replaced with NaN
    """
    logger = logging.getLogger(__name__)
    
    df = df.copy()
    replacements_made = 0
    nan_count_initial = df.isna().sum().sum()
    
    # Define valid ranges for metrics
    valid_ranges = {
        'psnr': (0, np.inf),      # PSNR should be positive
        'ssim': (0, 1),           # SSIM is bounded [0, 1]
        'ms_ssim': (0, 1),        # MS-SSIM is bounded [0, 1]
        'frc50': (0, np.inf),     # FRC50 should be positive
        'mae': (0, np.inf),       # MAE should be non-negative
        'mse': (0, np.inf),       # MSE should be non-negative
    }
    
    for col in df.columns:
        if col == 'model_type':
            continue
            
        # Extract metric name from column (e.g., 'psnr' from 'psnr_phase')
        metric_base = None
        for metric in valid_ranges.keys():
            if col.startswith(metric + '_'):
                metric_base = metric
                break
        
        if metric_base is None:
            continue
            
        # Count existing NaN values
        nan_mask = pd.isna(df[col])
        if nan_mask.any():
            nan_count = nan_mask.sum()
            logger.debug(f"Found {nan_count} NaN values in {col} (preserved)")
        
        # Handle out-of-range values by replacing with NaN
        if metric_base in valid_ranges:
            min_val, max_val = valid_ranges[metric_base]
            out_of_range_mask = (df[col] < min_val) | (df[col] > max_val)
            
            if out_of_range_mask.any():
                invalid_values = df.loc[out_of_range_mask, col].tolist()
                df.loc[out_of_range_mask, col] = np.nan
                count = out_of_range_mask.sum()
                replacements_made += count
                logger.warning(f"Replaced {count} out-of-range values in {col} with NaN. "
                             f"Invalid values: {invalid_values[:5]}{'...' if len(invalid_values) > 5 else ''}")
    
    nan_count_final = df.isna().sum().sum()
    new_nans = nan_count_final - nan_count_initial
    
    if replacements_made > 0:
        logger.info(f"Made {replacements_made} out-of-range replacements with NaN")
    if nan_count_final > 0:
        logger.info(f"Total NaN values in data: {nan_count_final} ({new_nans} new, {nan_count_initial} preserved)")
    
    return df


def parse_comparison_csv(csv_path: Path) -> pd.DataFrame:
    """
    Parse a single comparison_metrics.csv file with robust failure handling.
    
    This function handles both formats:
    1. Wide format: model_type, psnr_phase, psnr_amp, etc.
    2. Tidy format: model, metric, amplitude, phase
    
    It also handles missing files, NaN values, and out-of-range metrics.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with metrics data in wide format, with failures replaced by appropriate values
        
    Raises:
        ValueError: If file cannot be read or has critical format issues
    """
    logger = logging.getLogger(__name__)
    
    # Handle missing files
    if not csv_path.exists():
        logger.warning(f"CSV file does not exist: {csv_path}")
        # Return a DataFrame with failure values for both models
        return create_failure_dataframe()
    
    try:
        df = pd.read_csv(csv_path)
        logger.debug(f"Loaded CSV with shape {df.shape}: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to read CSV {csv_path}: {e}")
        # Return failure DataFrame instead of raising
        return create_failure_dataframe()
    
    # Check if this is tidy format (model, metric, amplitude, phase)
    if 'model' in df.columns and 'metric' in df.columns and 'amplitude' in df.columns and 'phase' in df.columns:
        logger.debug("Converting from tidy format to wide format")
        df = convert_tidy_to_wide_format(df)
    
    # Check for required columns
    required_cols = ['model_type']
    expected_metric_cols = [
        'psnr_phase', 'psnr_amp', 'frc50_phase', 'frc50_amp',
        'mae_phase', 'mae_amp', 'mse_phase', 'mse_amp',
        'ssim_phase', 'ssim_amp', 'ms_ssim_phase', 'ms_ssim_amp'
    ]
    
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        logger.error(f"Missing required columns in {csv_path}: {missing_required}")
        return create_failure_dataframe()
    
    # Check for metric columns (at least some should be present)
    available_metrics = [col for col in expected_metric_cols if col in df.columns]
    if not available_metrics:
        logger.error(f"No metric columns found in {csv_path}. Expected: {expected_metric_cols}")
        return create_failure_dataframe()
    
    logger.debug(f"Available metric columns: {available_metrics}")
    
    # Handle NaN values and invalid ranges
    df = handle_missing_or_failed_metrics(df)
    
    return df


def create_failure_dataframe() -> pd.DataFrame:
    """
    Create a DataFrame with NaN values for both model types.
    
    Used when a CSV file is missing or cannot be read, to ensure
    the trial is properly excluded from statistics.
    
    Returns:
        DataFrame with NaN values for common metrics
    """
    failure_data = []
    
    for model_type in ['pinn', 'baseline']:
        row = {'model_type': model_type}
        
        # Add NaN values for all metrics to ensure proper exclusion
        metrics = {
            'psnr_phase': np.nan, 'psnr_amp': np.nan,
            'ssim_phase': np.nan, 'ssim_amp': np.nan,
            'ms_ssim_phase': np.nan, 'ms_ssim_amp': np.nan,
            'frc50_phase': np.nan, 'frc50_amp': np.nan,
            'mae_phase': np.nan, 'mae_amp': np.nan,
            'mse_phase': np.nan, 'mse_amp': np.nan,
        }
        
        row.update(metrics)
        failure_data.append(row)
    
    return pd.DataFrame(failure_data)


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
        'Baseline': 'baseline',
        # Handle various iterative reconstruction variants
        'Tike': 'iterative',
        'Pty-chi (ePIE)': 'iterative',
        'Pty-chi (DM)': 'iterative',
        'Pty-chi (ML)': 'iterative',
        'Pty-chi': 'iterative'
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


def aggregate_results_statistical(comparison_files: List[Tuple[int, int, Path]], 
                                ms_ssim_phase_threshold: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate results from all comparison files with statistical analysis.
    
    This function handles multi-trial data and computes median and percentiles
    for robust statistical aggregation. Optionally filters out trials based on
    MS-SSIM phase threshold.
    
    Args:
        comparison_files: List of (training_size, trial_number, csv_path) tuples
        ms_ssim_phase_threshold: Minimum MS-SSIM phase value to include trial.
                                Set to 0 to disable filtering.
        
    Returns:
        Tuple containing:
        - DataFrame with statistical aggregation (train_size, model_type, metric_mean, metric_p25, metric_p75)
        - DataFrame with all individual trials including filter status
        
    Raises:
        ValueError: If no valid data is found
    """
    logger = logging.getLogger(__name__)
    
    all_trial_data = []
    
    # Load all trial data
    for train_size, trial_num, csv_path in comparison_files:
        try:
            df = parse_comparison_csv(csv_path)
            
            # Add metadata columns
            for _, row in df.iterrows():
                trial_row = row.to_dict()
                trial_row['train_size'] = train_size
                trial_row['trial_number'] = trial_num
                all_trial_data.append(trial_row)
                
        except Exception as e:
            logger.error(f"Failed to process {csv_path}: {e}")
            continue
    
    if not all_trial_data:
        raise ValueError("No valid trial data found")
    
    # Convert to DataFrame
    trials_df = pd.DataFrame(all_trial_data)
    logger.info(f"Loaded {len(trials_df)} trial records")
    
    # Create a DataFrame for all trials (before any filtering) for export
    all_trials_df = pd.DataFrame(all_trial_data)
    
    # Add a "Status" Column to track filter outcomes
    all_trials_df['filter_status'] = 'passed'
    
    # Apply MS-SSIM phase threshold filtering if requested
    if ms_ssim_phase_threshold > 0:
        initial_rows = len(trials_df)
        
        # Check if ms_ssim_phase column exists
        if 'ms_ssim_phase' not in trials_df.columns:
            logger.warning("MS-SSIM phase threshold specified but 'ms_ssim_phase' column not found. "
                         "Filtering disabled.")
        else:
            # Filter out trials with MS-SSIM phase below threshold
            # Note: NaN values will be excluded by this comparison
            mask = trials_df['ms_ssim_phase'] >= ms_ssim_phase_threshold
            
            # Update status for filtered trials in all_trials_df
            mask_low_msssim = all_trials_df['ms_ssim_phase'] < ms_ssim_phase_threshold
            all_trials_df.loc[mask_low_msssim, 'filter_status'] = 'filtered_low_msssim'
            
            trials_df = trials_df[mask]
            
            filtered_count = initial_rows - len(trials_df)
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} trial records with MS-SSIM (phase) < {ms_ssim_phase_threshold}")
                
                # Log details about filtered trials by train_size and model_type
                filtered_summary = {}
                for (train_size, model_type), group in trials_df.groupby(['train_size', 'model_type']):
                    original_count = len([row for row in all_trial_data 
                                        if row['train_size'] == train_size and row['model_type'] == model_type])
                    current_count = len(group)
                    if original_count > current_count:
                        filtered_summary[(train_size, model_type)] = {
                            'filtered': original_count - current_count,
                            'remaining': current_count
                        }
                
                if filtered_summary:
                    logger.info("Filtering details by configuration:")
                    for (train_size, model_type), counts in filtered_summary.items():
                        logger.info(f"  train_size={train_size}, model={model_type}: "
                                   f"{counts['filtered']} filtered, {counts['remaining']} remaining")
    
    # Mark trials with NaN metrics as failed
    # Use a key metric (psnr_phase) to identify failed trials
    if 'psnr_phase' in all_trials_df.columns:
        mask_nan = all_trials_df['psnr_phase'].isna()
        all_trials_df.loc[mask_nan, 'filter_status'] = 'failed_nan'
    
    # Identify metric columns (exclude metadata)
    metadata_cols = {'model_type', 'train_size', 'trial_number'}
    metric_cols = [col for col in trials_df.columns if col not in metadata_cols]
    
    logger.debug(f"Found metric columns: {metric_cols}")
    
    # Log final data before aggregation
    logger.info(f"Proceeding to aggregation with {len(trials_df)} trial records "
               f"(after any filtering applied)")
    
    # Group by train_size and model_type, then compute statistics
    statistical_results = []
    
    for (train_size, model_type), group in trials_df.groupby(['train_size', 'model_type']):
        n_trials = len(group)
        
        if n_trials < 3:
            logger.warning(f"Only {n_trials} trials for train_size={train_size}, model={model_type}. "
                         f"Percentile calculations may be unreliable.")
        
        result_row = {
            'train_size': train_size,
            'model_type': model_type,
            'n_trials': n_trials
        }
        
        # Compute statistics for each metric
        for metric_col in metric_cols:
            values = group[metric_col].values
            
            # Remove NaN values for statistical calculations
            valid_values = values[~np.isnan(values)]
            n_valid = len(valid_values)
            n_nan = len(values) - n_valid
            
            if n_nan > 0:
                logger.debug(f"Excluding {n_nan} NaN values from {metric_col} for "
                           f"train_size={train_size}, model={model_type}")
            
            if n_valid == 0:
                # All values are NaN
                mean = p25 = p75 = np.nan
                logger.warning(f"All {n_trials} trials have NaN for {metric_col} "
                             f"(train_size={train_size}, model={model_type})")
            elif n_valid == 1:
                # Single valid value
                mean = p25 = p75 = valid_values[0]
            else:
                # Compute mean and percentiles from valid values
                mean = np.mean(valid_values)
                p25, p75 = np.percentile(valid_values, [25, 75])
            
            # Store with standardized column names
            result_row[f"{metric_col}_mean"] = mean
            result_row[f"{metric_col}_p25"] = p25
            result_row[f"{metric_col}_p75"] = p75
            result_row[f"{metric_col}_n_valid"] = n_valid
        
        statistical_results.append(result_row)
    
    result_df = pd.DataFrame(statistical_results)
    logger.info(f"Computed statistics for {len(result_df)} (train_size, model_type) combinations")
    
    # Log summary of trial counts
    trial_counts = result_df.groupby('train_size')['n_trials'].agg(['min', 'max', 'mean'])
    for train_size, stats in trial_counts.iterrows():
        logger.info(f"Train size {train_size}: {stats['min']}-{stats['max']} trials "
                   f"(avg: {stats['mean']:.1f})")
    
    # Log summary of NaN exclusions
    nan_summary = {}
    for col in metric_cols:
        n_valid_col = f"{col}_n_valid"
        if n_valid_col in result_df.columns:
            total_trials = result_df['n_trials'].sum()
            total_valid = result_df[n_valid_col].sum()
            total_nan = total_trials - total_valid
            if total_nan > 0:
                nan_summary[col] = {'total_nan': total_nan, 'total_trials': total_trials}
    
    if nan_summary:
        logger.info("NaN exclusion summary:")
        for metric, stats in nan_summary.items():
            percentage = (stats['total_nan'] / stats['total_trials']) * 100
            logger.info(f"  {metric}: {stats['total_nan']} NaN values excluded "
                       f"({percentage:.1f}% of {stats['total_trials']} total trials)")
    
    return result_df, all_trials_df


def aggregate_results(comparison_files: List[Tuple[int, int, Path]], 
                     metric: str, part: str, ms_ssim_phase_threshold: float = 0.0) -> pd.DataFrame:
    """
    Aggregate results for a specific metric/part combination (legacy interface).
    
    This function maintains backward compatibility while using the new statistical
    aggregation internally.
    
    Args:
        comparison_files: List of (training_size, trial_number, csv_path) tuples
        metric: Metric to extract (psnr, frc50, mae, mse, ssim, ms_ssim)
        part: Data component (phase, amp)
        ms_ssim_phase_threshold: Minimum MS-SSIM phase value to include trial
        
    Returns:
        DataFrame with columns: train_size, model_type, metric_value (using mean)
        
    Raises:
        ValueError: If metric/part combination is not available in data
    """
    logger = logging.getLogger(__name__)
    
    # Use the new statistical aggregation
    stats_df, _ = aggregate_results_statistical(comparison_files, ms_ssim_phase_threshold)
    
    # Extract the specific metric requested
    metric_col = f"{metric}_{part}"
    mean_col = f"{metric_col}_mean"
    
    if mean_col not in stats_df.columns:
        available_metrics = [col.replace('_mean', '') for col in stats_df.columns if col.endswith('_mean')]
        raise ValueError(f"Metric {metric_col} not found. Available metrics: {available_metrics}")
    
    # Create legacy format DataFrame with ALL metrics preserved
    result_data = []
    for _, row in stats_df.iterrows():
        row_data = {
            'train_size': row['train_size'],
            'model_type': row['model_type'],
            'metric_value': row[mean_col]  # The metric requested for plotting
        }
        
        # Also include ALL other metrics for export to CSV
        # Extract all mean values and rename them without the '_mean' suffix
        for col in stats_df.columns:
            if col.endswith('_mean') and col not in ['train_size', 'model_type']:
                metric_name = col.replace('_mean', '')
                row_data[metric_name] = row[col]
        
        result_data.append(row_data)
    
    result_df = pd.DataFrame(result_data)
    logger.info(f"Extracted {len(result_df)} data points for {metric_col} (using mean)")
    logger.debug(f"DataFrame includes all metrics: {[c for c in result_df.columns if c not in ['train_size', 'model_type', 'metric_value']]}")
    
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


def create_generalization_plot_with_percentiles(stats_df: pd.DataFrame, metric: str, part: str,
                                               title: Optional[str] = None,
                                               output_path: Optional[Path] = None) -> Path:
    """
    Create publication-ready generalization plot with percentile bands.
    
    Args:
        stats_df: Statistical aggregation DataFrame with median and percentile columns
        metric: Metric name for labels
        part: Data component for labels
        title: Custom plot title
        output_path: Custom output path
        
    Returns:
        Path to saved plot file
    """
    logger = logging.getLogger(__name__)
    
    # Set up the plot
    plt.figure(figsize=(6, 4))
    
    # Define colors and markers for each model
    model_styles = {
        'pinn': {'color': '#2E86AB', 'marker': 'o', 'label': 'PtychoPINN'},
        'baseline': {'color': '#A23B72', 'marker': 's', 'label': 'Baseline'},
        'iterative': {'color': '#F18F01', 'marker': '^', 'label': 'Iterative (Pty-chi/Tike)'}
    }
    
    metric_col = f"{metric}_{part}"
    mean_col = f"{metric_col}_mean"
    p25_col = f"{metric_col}_p25"
    p75_col = f"{metric_col}_p75"
    
    # Check if required columns exist
    required_cols = [mean_col, p25_col, p75_col]
    missing_cols = [col for col in required_cols if col not in stats_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for {metric_col}: {missing_cols}")
    
    # Plot data for each model type
    for model_type in stats_df['model_type'].unique():
        model_data = stats_df[stats_df['model_type'] == model_type].copy()
        
        # Sort by training size
        model_data = model_data.sort_values('train_size')
        
        style = model_styles.get(model_type, {'color': 'gray', 'marker': 'x', 'label': model_type})
        
        train_sizes = model_data['train_size'].values
        means = model_data[mean_col].values
        p25_values = model_data[p25_col].values
        p75_values = model_data[p75_col].values
        n_trials = model_data['n_trials'].values
        
        # Check if we have percentile bands to show
        has_bands = not np.allclose(p25_values, p75_values)
        
        if has_bands:
            # Plot percentile band
            plt.fill_between(train_sizes, p25_values, p75_values,
                           color=style['color'], alpha=0.3, 
                           label=f"{style['label']} (IQR)" if len(stats_df['model_type'].unique()) == 1 else None)
        
        # Plot mean line
        line_style = '-' if has_bands else '--'
        plt.plot(train_sizes, means,
                marker=style['marker'], color=style['color'], 
                label=style['label'], linewidth=2, markersize=8,
                markerfacecolor='white', markeredgewidth=2,
                linestyle=line_style)
        
        # Add annotation for limited trial data if needed
        for i, (train_size, n_trial) in enumerate(zip(train_sizes, n_trials)):
            if n_trial == 1:
                plt.annotate('1 trial', (train_size, means[i]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
    
    # Set logarithmic X-axis with base 2
    plt.xscale('log', base=2)
    
    # Set axis labels
    metric_labels = {
        'psnr': 'PSNR (dB)',
        'frc50': 'FRC@0.5',
        'mae': 'Mean Absolute Error',
        'mse': 'Mean Squared Error',
        'ssim': 'SSIM Index',
        'ms_ssim': 'MS-SSIM Index'
    }
    
    plt.xlabel('Training Set Size (images)', fontsize=12)
    ylabel = metric_labels.get(metric, metric.upper())
    if has_bands:
        ylabel += ' (Mean ± IQR)'
    plt.ylabel(ylabel, fontsize=12)
    
    # Set title
    if title is None:
        part_label = part.capitalize()
        metric_label = metric.upper()
        stat_label = 'Mean Performance' if has_bands else 'Performance'
        title = f'{metric_label} ({part_label}) {stat_label} vs. Training Set Size'
    
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Set Y-axis limits for metrics with known ranges
    if metric in ['ssim', 'ms_ssim']:
        # SSIM and MS-SSIM have max value of 1, but autoscale the minimum
        plt.ylim(top=1)
        
    # Configure grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Set appropriate tick marks for common training sizes
    train_sizes = sorted(stats_df['train_size'].unique())
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


def create_generalization_plot(df: pd.DataFrame, metric: str, part: str, 
                             title: Optional[str] = None, 
                             output_path: Optional[Path] = None) -> Path:
    """
    Create publication-ready generalization plot (legacy interface).
    
    This function maintains backward compatibility with the old interface
    while supporting both statistical and simple data formats.
    
    Args:
        df: Aggregated data DataFrame (legacy format or statistical format)
        metric: Metric name for labels
        part: Data component for labels
        title: Custom plot title
        output_path: Custom output path
        
    Returns:
        Path to saved plot file
    """
    logger = logging.getLogger(__name__)
    
    # Check if this is statistical format (has mean/percentile columns)
    metric_col = f"{metric}_{part}"
    mean_col = f"{metric_col}_mean"
    
    if mean_col in df.columns:
        # Use new percentile plotting
        logger.debug("Using percentile-based plotting")
        return create_generalization_plot_with_percentiles(df, metric, part, title, output_path)
    else:
        # Use legacy plotting code (simplified version)
        logger.debug("Using legacy single-value plotting")
        
        # Set up the plot
        plt.figure(figsize=(6, 4))
        
        # Define colors and markers for each model
        model_styles = {
            'pinn': {'color': '#2E86AB', 'marker': 'o', 'label': 'PtychoPINN'},
            'baseline': {'color': '#A23B72', 'marker': 's', 'label': 'Baseline'},
            'iterative': {'color': '#F18F01', 'marker': '^', 'label': 'Iterative (Pty-chi/Tike)'}
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
            'mse': 'Mean Squared Error',
            'ssim': 'SSIM Index',
            'ms_ssim': 'MS-SSIM Index'
        }
        
        plt.xlabel('Training Set Size (images)', fontsize=12)
        plt.ylabel(metric_labels.get(metric, metric.upper()), fontsize=12)
        
        # Set title
        if title is None:
            part_label = part.capitalize()
            metric_label = metric.upper()
            title = f'{metric_label} ({part_label}) vs. Training Set Size'
        
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Set Y-axis limits for metrics with known ranges
        if metric in ['ssim', 'ms_ssim']:
            # SSIM and MS-SSIM have max value of 1, but autoscale the minimum
            plt.ylim(top=1)
            
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


def export_statistical_results(stats_df: pd.DataFrame, output_path: Path) -> None:
    """
    Export statistical aggregation results to CSV with metadata.
    
    Args:
        stats_df: Statistical aggregation DataFrame with median/percentile columns
        output_path: Path for output CSV file
    """
    logger = logging.getLogger(__name__)
    
    # Add metadata columns
    export_df = stats_df.copy()
    export_df['generated_at'] = datetime.now().isoformat()
    
    # Reorder columns to put metadata first
    metadata_cols = ['train_size', 'model_type', 'n_trials', 'generated_at']
    metric_cols = [col for col in export_df.columns if col not in metadata_cols]
    
    # Sort metric columns for consistency (group by base metric)
    metric_cols_sorted = []
    base_metrics = set()
    for col in metric_cols:
        if col.endswith('_mean'):
            base_metric = col.replace('_mean', '')
            base_metrics.add(base_metric)
    
    # Add columns in groups: mean, p25, p75 for each metric
    for base_metric in sorted(base_metrics):
        for suffix in ['_mean', '_p25', '_p75']:
            col = base_metric + suffix
            if col in metric_cols:
                metric_cols_sorted.append(col)
    
    # Add any remaining columns
    for col in metric_cols:
        if col not in metric_cols_sorted:
            metric_cols_sorted.append(col)
    
    export_df = export_df[metadata_cols + metric_cols_sorted]
    
    # Sort by training size and model type
    export_df = export_df.sort_values(['train_size', 'model_type'])
    
    # Save to CSV
    export_df.to_csv(output_path, index=False)
    logger.info(f"Statistical results exported to: {output_path}")
    logger.info(f"Exported {len(export_df)} rows with {len(base_metrics)} metrics")


def export_aggregated_results(df: pd.DataFrame, output_path: Path, 
                            metric: str, part: str) -> None:
    """
    Export aggregated results to CSV with metadata (legacy interface).
    
    Supports both legacy format and new statistical format.
    
    Args:
        df: Aggregated data DataFrame (legacy or statistical format)
        output_path: Path for output CSV file
        metric: Metric name for metadata (for legacy format)
        part: Data component for metadata (for legacy format)
    """
    logger = logging.getLogger(__name__)
    
    # Check if this is statistical format
    has_statistical_cols = any(col.endswith('_mean') for col in df.columns)
    
    if has_statistical_cols:
        # Use new statistical export
        logger.debug("Exporting statistical format results")
        export_statistical_results(df, output_path)
    else:
        # Use legacy export - but export ALL metrics, not just the one used for plotting
        logger.debug("Exporting legacy format results with all available metrics")
        
        # Identify all metric columns (exclude metadata columns)
        metadata_cols = {'train_size', 'model_type', 'metric_value'}
        all_metric_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Create rows for each metric
        export_rows = []
        for _, row in df.iterrows():
            base_data = {
                'train_size': row['train_size'],
                'model_type': row['model_type'],
                'generated_at': datetime.now().isoformat()
            }
            
            # If there's a 'metric_value' column (from the aggregation for plotting),
            # add it with the specified metric name
            if 'metric_value' in df.columns:
                export_row = base_data.copy()
                export_row['metric_name'] = f"{metric}_{part}"
                export_row['metric_value'] = row['metric_value']
                export_rows.append(export_row)
            
            # Add all other available metrics
            for col in all_metric_cols:
                if col in row and pd.notna(row[col]):
                    export_row = base_data.copy()
                    export_row['metric_name'] = col
                    export_row['metric_value'] = row[col]
                    export_rows.append(export_row)
        
        # Create DataFrame from rows
        export_df = pd.DataFrame(export_rows)
        
        # Reorder columns
        cols = ['train_size', 'model_type', 'metric_name', 'metric_value', 'generated_at']
        export_df = export_df[cols]
        
        # Sort by training size, model type, and metric name
        export_df = export_df.sort_values(['train_size', 'model_type', 'metric_name'])
        
        # Save to CSV
        export_df.to_csv(output_path, index=False)
        logger.info(f"Results exported to: {output_path} ({len(export_df)} metric entries)")


def export_all_trials_results(all_trials_df: pd.DataFrame, output_path: Path) -> None:
    """
    Export all individual trial results to CSV with filter status information.
    
    This function creates a comprehensive record of every trial in the study,
    including those that were filtered out or failed. Each row represents one
    trial for one model type.
    
    Args:
        all_trials_df: DataFrame containing all trial data with filter_status column
        output_path: Path for output CSV file
    """
    logger = logging.getLogger(__name__)
    
    # Create export DataFrame with desired column order
    export_df = all_trials_df.copy()
    
    # Define desired column order, putting key identifiers first
    key_cols = ['train_size', 'trial_number', 'model_type', 'filter_status']
    
    # Get all metric columns (excluding metadata)
    metadata_cols = {'train_size', 'trial_number', 'model_type', 'filter_status'}
    metric_cols = [col for col in export_df.columns if col not in metadata_cols]
    
    # Sort metric columns alphabetically for consistency
    metric_cols_sorted = sorted(metric_cols)
    
    # Reorder columns
    column_order = key_cols + metric_cols_sorted
    export_df = export_df[column_order]
    
    # Sort the DataFrame by train_size, trial_number, and model_type
    export_df = export_df.sort_values(['train_size', 'trial_number', 'model_type'])
    
    # Add generation timestamp
    export_df['generated_at'] = datetime.now().isoformat()
    
    # Save the DataFrame to CSV
    export_df.to_csv(output_path, index=False)
    
    # Log summary information
    total_trials = len(export_df)
    status_counts = export_df['filter_status'].value_counts()
    
    logger.info(f"All trials results exported to: {output_path}")
    logger.info(f"Total trial records: {total_trials}")
    for status, count in status_counts.items():
        percentage = (count / total_trials) * 100
        logger.info(f"  {status}: {count} ({percentage:.1f}%)")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Aggregate and visualize model generalization study results with '
                    'robust NaN handling and optional MS-SSIM phase filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('study_dir', type=Path,
                       help='Directory containing study results (with train_* subdirectories)')
    
    parser.add_argument('--metric', choices=['psnr', 'frc50', 'mae', 'mse', 'ssim', 'ms_ssim'], 
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
    
    parser.add_argument('--ms-ssim-phase-threshold', type=float, default=0.3,
                       help='Exclude trials where the MS-SSIM (phase) is below this value. '
                            'Set to 0 to disable filtering. Default: 0.3')
    
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
        
        # Step 2: Check if we have multi-trial data
        has_multi_trial = any(trial_num > 1 for _, trial_num, _ in comparison_files)
        multi_train_sizes = len(set(train_size for train_size, _, _ in comparison_files 
                                  if len([t for t, _, _ in comparison_files if t == train_size]) > 1))
        
        if has_multi_trial or multi_train_sizes > 0:
            logger.info("Multi-trial data detected - using statistical aggregation")
            
            # Use statistical aggregation for multi-trial data
            stats_df, all_trials_df = aggregate_results_statistical(comparison_files, args.ms_ssim_phase_threshold)
            
            # Export all individual trial results
            all_trials_path = args.study_dir / 'results_all_trials.csv'
            export_all_trials_results(all_trials_df, all_trials_path)
            
            # Validate data quality using legacy interface (on specific metric)
            legacy_df = aggregate_results(comparison_files, args.metric, args.part, args.ms_ssim_phase_threshold)
            validate_aggregated_data(legacy_df)
            
            # Create plot with percentiles
            plot_path = args.study_dir / args.output
            create_generalization_plot(stats_df, args.metric, args.part, args.title, plot_path)
            
            # Export statistical results
            results_path = args.study_dir / 'results.csv'
            export_statistical_results(stats_df, results_path)
            
            # Print data summary
            train_sizes = sorted(stats_df['train_size'].unique())
            model_types = sorted(stats_df['model_type'].unique())
            
            # Calculate trial statistics
            trial_stats = stats_df.groupby('train_size')['n_trials'].agg(['min', 'max', 'mean'])
            total_trials = stats_df['n_trials'].sum()
            
            logger.info("Processing complete!")
            logger.info(f"Outputs:")
            logger.info(f"  - Plot: {plot_path}")
            logger.info(f"  - Data: {results_path}")
            logger.info(f"  - All trials: {all_trials_path}")
            logger.info(f"Summary: {len(model_types)} models, {len(train_sizes)} training sizes, {total_trials} total trials")
            logger.info(f"Training sizes: {train_sizes}")
            logger.info(f"Model types: {model_types}")
            
            for train_size, stats in trial_stats.iterrows():
                logger.info(f"  Train size {train_size}: {int(stats['min'])}-{int(stats['max'])} trials per model")
                
        else:
            logger.info("Single-trial data detected - using legacy aggregation")
            
            # Use legacy workflow for single-trial data
            df = aggregate_results(comparison_files, args.metric, args.part, args.ms_ssim_phase_threshold)
            
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
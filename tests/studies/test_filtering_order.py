#!/usr/bin/env python3
"""
Enhanced test script to verify filtering order in aggregate_and_plot_results.py

This script creates mock data with specific MS-SSIM values to test if filtering
happens before or after statistical aggregation.

Test scenario:
- 4 trials with ms_ssim_phase values: 0.8, 0.7, 0.6, 0.1
- With threshold 0.3, the outlier (0.1) should be filtered out
- Correct median after filtering: 0.7 (from [0.8, 0.7, 0.6])
- Incorrect median if filtering after aggregation: 0.65 (from [0.8, 0.7, 0.6, 0.1])
"""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def create_test_csv(path: Path, ms_ssim_phase: float, psnr_phase: float = 80.0):
    """Create a mock comparison_metrics.csv file with specific values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['model_type', 'psnr_phase', 'ms_ssim_phase'])
        
        # Write data for both models (use same values for simplicity)
        writer.writerow(['pinn', psnr_phase, ms_ssim_phase])
        writer.writerow(['baseline', psnr_phase - 5, ms_ssim_phase - 0.05])


def run_aggregation_test(study_dir: Path, threshold: float = 0.3):
    """Run aggregation and extract median values from results."""
    script_path = Path(__file__).parent / 'aggregate_and_plot_results.py'
    
    cmd = [
        sys.executable,
        str(script_path),
        str(study_dir),
        '--ms-ssim-phase-threshold', str(threshold),
        '--metric', 'ms_ssim',
        '--part', 'phase',
        '--verbose'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        return None, None
    
    # Read the results CSV to get the median values
    results_path = study_dir / 'results.csv'
    if not results_path.exists():
        print("Results CSV not found!")
        return None, None
    
    with open(results_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Extract median values for both models
    pinn_median = None
    baseline_median = None
    
    for row in rows:
        if row['model_type'] == 'pinn':
            pinn_median = float(row['ms_ssim_phase_median'])
        elif row['model_type'] == 'baseline':
            baseline_median = float(row['ms_ssim_phase_median'])
    
    return pinn_median, baseline_median


def main():
    """Run the filtering order test."""
    print("Testing MS-SSIM filtering order in aggregate_and_plot_results.py")
    print("=" * 70)
    
    # Test values: 0.8, 0.7, 0.6, 0.1 (outlier)
    # Expected median after filtering (threshold 0.3): 0.7
    # Incorrect median if no filtering: 0.65
    
    test_values = [0.8, 0.7, 0.6, 0.1]
    expected_median_correct = 0.7  # median of [0.8, 0.7, 0.6] after filtering
    expected_median_wrong = 0.65   # median of [0.8, 0.7, 0.6, 0.1] without filtering
    
    print(f"Test values: {test_values}")
    print(f"Expected median if filtering works correctly: {expected_median_correct}")
    print(f"Expected median if filtering is broken: {expected_median_wrong}")
    print()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        study_dir = Path(tmpdir) / 'filtering_test'
        train_dir = study_dir / 'train_1024'
        
        # Create 4 trials with specific MS-SSIM values
        for i, ms_ssim_val in enumerate(test_values, 1):
            psnr_val = 80.0 + i  # Make PSNR slightly different for each trial
            create_test_csv(
                train_dir / f'trial_{i}' / 'comparison_metrics.csv',
                ms_ssim_phase=ms_ssim_val,
                psnr_phase=psnr_val
            )
            print(f"Trial {i}: MS-SSIM phase = {ms_ssim_val}, PSNR phase = {psnr_val}")
        
        print()
        print("Running aggregation with threshold 0.3...")
        print("-" * 40)
        
        pinn_median, baseline_median = run_aggregation_test(study_dir, threshold=0.3)
        
        if pinn_median is None:
            print("❌ Test failed - could not extract results")
            return 1
        
        print(f"Results:")
        print(f"  PtychoPINN median: {pinn_median}")
        print(f"  Baseline median: {baseline_median}")
        print()
        
        # Check if filtering worked correctly
        tolerance = 0.01
        if abs(pinn_median - expected_median_correct) < tolerance:
            print("✅ FILTERING ORDER IS CORRECT")
            print("   The outlier (0.1) was filtered out before calculating median")
            print(f"   Median {pinn_median} ≈ {expected_median_correct} (expected for correct filtering)")
            success = True
        elif abs(pinn_median - expected_median_wrong) < tolerance:
            print("❌ BUG CONFIRMED: FILTERING ORDER IS WRONG")
            print("   The outlier (0.1) was NOT filtered out before calculating median")
            print(f"   Median {pinn_median} ≈ {expected_median_wrong} (indicates filtering after aggregation)")
            success = False
        else:
            print(f"⚠️  UNEXPECTED RESULT: Median {pinn_median} doesn't match either expectation")
            print(f"   Expected: {expected_median_correct} (correct) or {expected_median_wrong} (bug)")
            success = False
        
        print()
        print("=" * 70)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Test script to verify that aggregate_and_plot_results.py now uses mean instead of median

This script creates mock data with specific values to test the statistical aggregation:
- 3 trials with values: 10, 20, 30
- Expected mean: 20.0
- Expected 25th percentile: 15.0
- Expected 75th percentile: 25.0
- Expected median (old): 20.0 (same as mean in this case)
"""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def create_test_csv(path: Path, metric_value: float):
    """Create a mock comparison_metrics.csv file with specific metric value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['model_type', 'psnr_phase'])
        
        # Write data for both models
        writer.writerow(['pinn', metric_value])
        writer.writerow(['baseline', metric_value - 1])  # Slightly different for baseline


def run_aggregation_test(study_dir: Path):
    """Run aggregation and extract statistics from results."""
    script_path = Path(__file__).parent / 'aggregate_and_plot_results.py'
    
    cmd = [
        sys.executable,
        str(script_path),
        str(study_dir),
        '--metric', 'psnr',
        '--part', 'phase',
        '--verbose'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        return None
    
    # Read the results CSV to get the statistics
    results_path = study_dir / 'results.csv'
    if not results_path.exists():
        print("Results CSV not found!")
        return None
    
    with open(results_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Extract statistics for PtychoPINN model
    for row in rows:
        if row['model_type'] == 'pinn':
            stats = {}
            for col in row.keys():
                if col.startswith('psnr_phase_'):
                    stat_type = col.replace('psnr_phase_', '')
                    stats[stat_type] = float(row[col]) if row[col] else None
            return stats
    
    return None


def main():
    """Run the mean vs median verification test."""
    print("Testing Mean vs Median in aggregate_and_plot_results.py")
    print("=" * 60)
    
    # Test values: 10, 20, 30
    test_values = [10.0, 20.0, 30.0]
    expected_mean = 20.0
    expected_p25 = 15.0  # 25th percentile
    expected_p75 = 25.0  # 75th percentile
    
    print(f"Test values: {test_values}")
    print(f"Expected mean: {expected_mean}")
    print(f"Expected 25th percentile: {expected_p25}")
    print(f"Expected 75th percentile: {expected_p75}")
    print()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        study_dir = Path(tmpdir) / 'mean_test'
        train_dir = study_dir / 'train_1024'
        
        # Create 3 trials with specific values
        for i, value in enumerate(test_values, 1):
            create_test_csv(
                train_dir / f'trial_{i}' / 'comparison_metrics.csv',
                metric_value=value
            )
            print(f"Trial {i}: PSNR phase = {value}")
        
        print()
        print("Running aggregation...")
        print("-" * 30)
        
        stats = run_aggregation_test(study_dir)
        
        if stats is None:
            print("❌ Test failed - could not extract results")
            return 1
        
        print(f"Results from CSV:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()
        
        # Check if we have mean instead of median
        if 'mean' in stats and 'median' not in stats:
            print("✅ SUCCESS: Using MEAN (new behavior)")
            
            # Verify the calculated values
            tolerance = 0.01
            success = True
            
            if abs(stats['mean'] - expected_mean) < tolerance:
                print(f"✅ Mean calculation correct: {stats['mean']} ≈ {expected_mean}")
            else:
                print(f"❌ Mean calculation wrong: {stats['mean']} ≠ {expected_mean}")
                success = False
            
            if abs(stats['p25'] - expected_p25) < tolerance:
                print(f"✅ 25th percentile correct: {stats['p25']} ≈ {expected_p25}")
            else:
                print(f"❌ 25th percentile wrong: {stats['p25']} ≠ {expected_p25}")
                success = False
            
            if abs(stats['p75'] - expected_p75) < tolerance:
                print(f"✅ 75th percentile correct: {stats['p75']} ≈ {expected_p75}")
            else:
                print(f"❌ 75th percentile wrong: {stats['p75']} ≠ {expected_p75}")
                success = False
            
            return 0 if success else 1
            
        elif 'median' in stats and 'mean' not in stats:
            print("❌ FAILURE: Still using MEDIAN (old behavior)")
            print(f"   Found median: {stats['median']}")
            return 1
        else:
            print("⚠️  UNCLEAR: Unexpected column structure")
            print(f"   Available stats: {list(stats.keys())}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
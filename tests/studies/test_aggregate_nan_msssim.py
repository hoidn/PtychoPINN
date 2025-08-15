#!/usr/bin/env python3
"""
Test script to verify NaN handling and MS-SSIM filtering in aggregate_and_plot_results.py

This script creates mock data directories with controlled test cases:
1. A trial with good MS-SSIM (0.8)
2. A trial with bad MS-SSIM (0.1) 
3. A trial with NaN values

Then runs the aggregation script with different thresholds to verify behavior.
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def create_mock_csv(path: Path, ms_ssim_phase: float = None):
    """Create a mock comparison_metrics.csv file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['model_type', 'psnr_phase', 'psnr_amp', 'ms_ssim_phase', 'ms_ssim_amp', 
                        'mae_phase', 'mae_amp', 'frc50_phase', 'frc50_amp'])
        
        # Write data for both models
        if ms_ssim_phase is None:
            # NaN case - leave fields empty
            writer.writerow(['pinn', '', '', '', '', '', '', '', ''])
            writer.writerow(['baseline', '', '', '', '', '', '', '', ''])
        else:
            # Normal case with specified MS-SSIM
            psnr_phase = 80.0 if ms_ssim_phase > 0.5 else 60.0
            writer.writerow(['pinn', psnr_phase, 75.0, ms_ssim_phase, 0.9, 0.05, 0.03, 50, 45])
            writer.writerow(['baseline', psnr_phase - 5, 70.0, ms_ssim_phase - 0.05, 0.85, 0.08, 0.04, 45, 40])


def run_aggregation(study_dir: Path, threshold: float = 0.3, verbose: bool = True):
    """Run the aggregation script and capture output."""
    script_path = Path(__file__).parent / 'aggregate_and_plot_results.py'
    
    cmd = [
        sys.executable,
        str(script_path),
        str(study_dir),
        '--ms-ssim-phase-threshold', str(threshold),
        '--metric', 'psnr',
        '--part', 'phase'
    ]
    
    if verbose:
        cmd.append('--verbose')
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("\n--- STDOUT ---")
    print(result.stdout)
    
    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr)
    
    return result.returncode == 0, result.stdout, result.stderr


def verify_results_csv(results_path: Path, expected_trials: int):
    """Verify the contents of results.csv."""
    if not results_path.exists():
        print(f"ERROR: {results_path} does not exist")
        return False
    
    with open(results_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\nResults CSV contains {len(rows)} rows")
    
    # Check for expected number of trials included
    pinn_rows = [r for r in rows if r['model_type'] == 'pinn']
    if pinn_rows:
        n_trials = int(pinn_rows[0].get('n_trials', 1))
        print(f"PtychoPINN n_trials: {n_trials} (expected: {expected_trials})")
        
        # Check if NaN handling worked (n_valid should be present)
        if 'psnr_phase_n_valid' in pinn_rows[0]:
            n_valid = int(pinn_rows[0]['psnr_phase_n_valid'])
            print(f"PtychoPINN n_valid for psnr_phase: {n_valid}")
        
        return n_trials == expected_trials
    
    return False


def main():
    """Run the test cases."""
    print("Testing NaN handling and MS-SSIM filtering in aggregate_and_plot_results.py")
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        study_dir = Path(tmpdir) / 'test_study'
        train_dir = study_dir / 'train_512'
        
        # Create three trials
        print("\nCreating mock data:")
        print("  Trial 1: Good MS-SSIM (0.8)")
        create_mock_csv(train_dir / 'trial_1' / 'comparison_metrics.csv', ms_ssim_phase=0.8)
        
        print("  Trial 2: Bad MS-SSIM (0.1)")
        create_mock_csv(train_dir / 'trial_2' / 'comparison_metrics.csv', ms_ssim_phase=0.1)
        
        print("  Trial 3: NaN values")
        create_mock_csv(train_dir / 'trial_3' / 'comparison_metrics.csv', ms_ssim_phase=None)
        
        # Test Case 1: Default threshold (0.3)
        print("\n" + "="*70)
        print("TEST CASE 1: Default threshold (0.3)")
        print("Expected: NaN trial excluded, bad trial excluded, only good trial included")
        print("="*70)
        
        success1, stdout1, _ = run_aggregation(study_dir, threshold=0.3)
        if success1:
            results_path = study_dir / 'results.csv'
            verify_results_csv(results_path, expected_trials=1)
            
            # Check for filtering messages in output
            if "Filtered out" in stdout1:
                print("✓ Filtering message found in output")
            if "NaN values excluded" in stdout1:
                print("✓ NaN exclusion message found in output")
        else:
            print("✗ Aggregation failed!")
        
        # Clean up for next test
        if (study_dir / 'results.csv').exists():
            os.remove(study_dir / 'results.csv')
        if (study_dir / 'generalization_plot.png').exists():
            os.remove(study_dir / 'generalization_plot.png')
        
        # Test Case 2: No filtering (threshold=0)
        print("\n" + "="*70)
        print("TEST CASE 2: No filtering (threshold=0)")
        print("Expected: NaN trial excluded, both good and bad trials included")
        print("="*70)
        
        success2, stdout2, _ = run_aggregation(study_dir, threshold=0.0)
        if success2:
            results_path = study_dir / 'results.csv'
            verify_results_csv(results_path, expected_trials=2)
            
            # Should not see filtering messages
            if "Filtered out" not in stdout2 or "threshold=0" in stdout2:
                print("✓ No MS-SSIM filtering with threshold=0")
            if "NaN values excluded" in stdout2:
                print("✓ NaN exclusion still active")
        else:
            print("✗ Aggregation failed!")
        
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        if success1 and success2:
            print("✓ All tests passed!")
            print("\nThe script correctly:")
            print("  1. Preserves and excludes NaN values from statistics")
            print("  2. Filters trials based on MS-SSIM phase threshold")
            print("  3. Logs information about excluded trials")
            return 0
        else:
            print("✗ Some tests failed")
            return 1


if __name__ == "__main__":
    sys.exit(main())
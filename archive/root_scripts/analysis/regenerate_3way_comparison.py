#!/usr/bin/env python
"""Regenerate 3-way comparison metrics including pty-chi with fixed MS-SSIM."""

import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, '/home/ollie/Documents/PtychoPINN')

from scripts.compare_models import main as compare_main
import argparse

# First run the standard 2-model comparison
print("Running PtychoPINN vs Baseline comparison...")
sys.argv = [
    'compare_models.py',
    '--pinn_dir', '3way_synthetic_ptychi_1e4_128/train_128/trial_1/pinn_run',
    '--baseline_dir', '3way_synthetic_ptychi_1e4_128/train_128/trial_1/baseline_run',
    '--test_data', 'prepare_1e4_photons_5k/dataset/test.npz',
    '--output_dir', '3way_synthetic_ptychi_1e4_128/train_128/trial_1/comparison_results_fixed',
    '--n-test-images', '128',
    '--save-debug-images'
]

try:
    compare_main()
except SystemExit:
    pass

print("\nComparison complete. Now adding pty-chi evaluation...")

# Now manually evaluate pty-chi and add to metrics
from ptycho.evaluation import eval_reconstruction
from ptycho.workflows.components import load_data

# Load test data
test_data_raw = load_data('prepare_1e4_photons_5k/dataset/test.npz', n_images=128)
test_data = test_data_raw.generate_grouped_data(nsamples=128, gridsize=1)

# Load pty-chi reconstruction
with np.load('3way_synthetic_ptychi_1e4_128/train_128/trial_1/ptychi_run/ptychi_reconstruction.npz', allow_pickle=True) as data:
    ptychi_recon = data['reconstructed_object']
    
# Evaluate pty-chi
print("Evaluating pty-chi reconstruction...")
try:
    ptychi_metrics = eval_reconstruction(
        pred_object=ptychi_recon,
        target_object=test_data.Y_I[..., 0] + 1j * test_data.Y_phi[..., 0],
        positions_pred=test_data.coords_true[..., 0],
        positions_true=test_data.coords_true[..., 0],
        probe=test_data.probe,
        stitch_crop_size=20,
        ms_ssim_sigma=0.0
    )
    
    # Add pty-chi metrics to CSV
    csv_path = '3way_synthetic_ptychi_1e4_128/train_128/trial_1/comparison_results_fixed/comparison_metrics.csv'
    
    # Read existing CSV
    df = pd.read_csv(csv_path)
    
    # Add pty-chi rows
    new_rows = []
    for metric in ['mae', 'mse', 'psnr', 'ssim', 'ms_ssim', 'frc50']:
        if metric in ptychi_metrics:
            val = ptychi_metrics[metric]
            new_rows.append({
                'model': 'Pty-chi (ePIE)',
                'metric': metric,
                'amplitude': val.get('amplitude', np.nan) if isinstance(val, dict) else np.nan,
                'phase': val.get('phase', np.nan) if isinstance(val, dict) else np.nan,
                'value': np.nan
            })
    
    # Add computation time
    new_rows.append({
        'model': 'Pty-chi (ePIE)',
        'metric': 'computation_time_s',
        'amplitude': np.nan,
        'phase': np.nan,
        'value': 13.019052  # From original run
    })
    
    # Append new rows
    df_new = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    
    # Save updated CSV
    df_new.to_csv(csv_path, index=False)
    print(f"Updated metrics saved to {csv_path}")
    
    # Display MS-SSIM values
    print("\nMS-SSIM values:")
    ms_ssim_rows = df_new[df_new['metric'] == 'ms_ssim']
    print(ms_ssim_rows[['model', 'amplitude', 'phase']])
    
except Exception as e:
    print(f"Error evaluating pty-chi: {e}")
    import traceback
    traceback.print_exc()
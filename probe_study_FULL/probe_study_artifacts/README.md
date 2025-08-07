# Probe Study Artifacts

This directory contains the key artifacts from the Probe Parameterization Study completed on 2025-08-01.

## Contents

### Probes
- `default_probe.npy` - Idealized probe with experimental amplitude and flat phase
- `hybrid_probe.npy` - Hybrid probe with experimental amplitude and aberrated phase
- `probe_pair_visualization.png` - Visual comparison of the two probes

### Reports
- `2x2_study_report_final.md` - Comprehensive final report with analysis

### Study Data
The full experimental data is available in the parent directories:
- `gs1_default/` - Results from gridsize=1 with default probe
- `gs1_hybrid/` - Results from gridsize=1 with hybrid probe

Each experiment directory contains:
- `simulated_data.npz` - Training data (5000 diffraction patterns)
- `test_data.npz` - Test subset (1000 diffraction patterns)
- `model/` - Trained PtychoPINN model
- `evaluation/` - Evaluation results and metrics

## Key Finding

Models trained with phase-aberrated probes achieved **13 dB better PSNR** than those trained with idealized flat-phase probes, demonstrating that probe aberrations in training data enhance rather than hinder learning.

## Scripts Used

The study was conducted using:
- `scripts/studies/prepare_probe_study.py` - Probe preparation
- `scripts/studies/run_probe_study_corrected.sh` - Study orchestration
- `scripts/studies/aggregate_2x2_results.py` - Results aggregation
- `scripts/studies/generate_2x2_visualization.py` - Visualization generation
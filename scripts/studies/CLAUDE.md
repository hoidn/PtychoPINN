# Model Studies Agent Guide

## Quick Context  
- **Primary Tool**: `run_complete_generalization_study.sh`
- **Purpose**: Compare PtychoPINN vs baseline across training sizes
- **Output**: Statistical analysis with mean/percentiles
- **Use Case**: Publication-quality model evaluation
- **NEW Tool**: `run_2x2_probe_study.sh` for probe parameterization studies

## Essential Commands

### Complete Generalization Study
```bash
# Full statistical study with multiple training sizes and trials
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "512 1024 2048 4096" \
    --num-trials 3 \
    --output-dir study_results

# Quick test study
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "512 1024" \
    --num-trials 2 \
    --output-dir quick_study

# NEW: 3-way comparison including Tike iterative reconstruction
./scripts/studies/run_complete_generalization_study.sh \
    --add-tike-arm \
    --tike-iterations 1000 \
    --train-sizes "512 1024 2048" \
    --num-trials 3 \
    --output-dir three_way_study

# Quick 3-way test (100 iterations for faster execution)
./scripts/studies/run_complete_generalization_study.sh \
    --add-tike-arm \
    --tike-iterations 100 \
    --train-sizes "512 1024" \
    --num-trials 1 \
    --output-dir quick_3way_test
```

### Probe Parameterization Study (Phase 3 Complete)
```bash
# Full 2x2 probe study (idealized vs hybrid probe, gridsize 1 vs 2)
./scripts/studies/run_2x2_probe_study.sh \
    --output-dir probe_study_results \
    --dataset datasets/fly/fly64_transposed.npz

# Quick test to verify pipeline (256 train images, 5 epochs)
./scripts/studies/run_2x2_probe_study.sh \
    --output-dir test_probe_study \
    --quick-test

# Resume interrupted study using checkpoint markers
./scripts/studies/run_2x2_probe_study.sh \
    --output-dir probe_study_results \
    --skip-completed

# Parallel execution with multiple jobs
./scripts/studies/run_2x2_probe_study.sh \
    --output-dir probe_study_results \
    --parallel-jobs 2

# Custom dataset with dry-run mode
./scripts/studies/run_2x2_probe_study.sh \
    --output-dir custom_study \
    --dataset datasets/my_data.npz \
    --dry-run
```

### Single Model Comparison
```bash
# Compare two existing trained models (2-way)
python scripts/compare_models.py \
    --pinn_dir <pinn_model_dir> \
    --baseline_dir <baseline_model_dir> \
    --test_data <test.npz> \
    --output_dir comparison_results

# NEW: 3-way comparison with existing Tike reconstruction
python scripts/compare_models.py \
    --pinn_dir <pinn_model_dir> \
    --baseline_dir <baseline_model_dir> \
    --test_data <test.npz> \
    --tike_recon_path <tike_reconstruction.npz> \
    --output_dir three_way_comparison \
    --n-test-images 1024
```

### Result Aggregation
```bash
# Generate publication plots from study results
python scripts/studies/aggregate_and_plot_results.py \
    --study-dir study_results \
    --output-dir publication_plots

# Specific metric analysis
python scripts/studies/aggregate_and_plot_results.py \
    --study-dir study_results \
    --metric amplitude_ssim \
    --part amplitude \
    --output-dir ssim_analysis
```

## Study Architecture

### Directory Structure

#### Generalization Study
```
study_results/
├── train_512/
│   ├── trial_1/
│   │   ├── pinn_run/wts.h5.zip
│   │   └── baseline_run/baseline_model.h5
│   ├── trial_2/
│   └── trial_3/
├── train_1024/
├── train_2048/
└── train_4096/
```

#### Probe Parameterization Study (2x2 Factorial Design)
```
probe_study_results/
├── idealized_probe.npy        # Generated synthetic probe (get_default_probe)
├── hybrid_probe.npy           # Generated hybrid probe (idealized amplitude + experimental phase)
├── study_summary.csv          # Combined results across all 4 conditions
├── study_log.txt              # Complete execution log
├── gs1_idealized/             # Condition 1: Gridsize 1, idealized probe (reference)
│   ├── simulated_data.npz     # Training data for this condition
│   ├── test_data.npz          # Test subset for evaluation
│   ├── model/wts.h5.zip       # Trained PtychoPINN model
│   ├── evaluation/            # Comparison metrics and visualizations
│   ├── training.log           # Training execution log
│   ├── simulation.log         # Simulation execution log  
│   ├── evaluation.log         # Evaluation execution log
│   ├── metrics_summary.csv    # Key metrics with experiment metadata
│   ├── .simulation_done       # Checkpoint marker
│   ├── .training_done         # Checkpoint marker
│   └── .evaluation_done       # Checkpoint marker
├── gs1_hybrid/                # Condition 2: Gridsize 1, hybrid probe (phase aberration test)
├── gs2_idealized/             # Condition 3: Gridsize 2, idealized probe (patch assembly test)
└── gs2_hybrid/                # Condition 4: Gridsize 2, hybrid probe (combined stress test)
```

### Probe Parameterization Study Design

The 2x2 probe study tests PtychoPINN robustness across two factors:
- **Factor 1: Gridsize** (1 vs 2) - tests patch assembly strategy impact
- **Factor 2: Probe Type** (idealized vs hybrid) - tests phase aberration sensitivity

**Study Interpretation:**
- **gs1_idealized**: Reference baseline (synthetic probe, no patch assembly)
- **gs1_hybrid**: Phase aberration impact (realistic probe phase, no patch assembly)  
- **gs2_idealized**: Patch assembly impact (synthetic probe, patch assembly)
- **gs2_hybrid**: Combined stress test (realistic probe phase + patch assembly)

**Key Metrics:** PSNR (amplitude/phase), FRC50, SSIM, MS-SSIM for reconstruction quality assessment.

### Statistical Analysis
- **Multiple trials** per training size for robustness
- **Mean ± IQR** reporting instead of median
- **Cross-validation** across different training set sizes

## Common Patterns

### Production Study
```bash
# 1. Run complete generalization study
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "512 1024 2048 4096 8192" \
    --num-trials 5 \
    --output-dir production_study

# 2. Generate comprehensive plots
python scripts/studies/aggregate_and_plot_results.py \
    --study-dir production_study \
    --output-dir publication_figures

# 3. Extract specific results
python scripts/studies/aggregate_and_plot_results.py \
    --study-dir production_study \
    --metric amplitude_ms_ssim \
    --output-dir ms_ssim_results
```

### Quick Comparison
```bash
# Compare existing models from different experiments
python scripts/compare_models.py \
    --pinn_dir experiment_A/pinn_model \
    --baseline_dir experiment_B/baseline_model \
    --test_data datasets/common_test.npz \
    --output_dir A_vs_B_comparison \
    --save-debug-images
```

## Advanced Features

### Registration Control
```bash
# Disable automatic image registration
python scripts/compare_models.py \
    --pinn_dir <pinn> --baseline_dir <baseline> \
    --test_data <test.npz> --output_dir <output> \
    --skip-registration

# Enable debug visualization
python scripts/compare_models.py \
    --pinn_dir <pinn> --baseline_dir <baseline> \
    --test_data <test.npz> --output_dir <output> \
    --save-debug-images
```

### Custom Metrics Configuration
```bash
# Advanced evaluation options
python scripts/compare_models.py \
    --pinn_dir <pinn> --baseline_dir <baseline> \
    --test_data <test.npz> --output_dir <output> \
    --ms-ssim-sigma 2.0 \
    --phase-align-method plane \
    --frc-sigma 1.0
```

### NPZ Export Control
```bash
# Disable NPZ exports to save disk space
python scripts/compare_models.py \
    --pinn_dir <pinn> --baseline_dir <baseline> \
    --test_data <test.npz> --output_dir <output> \
    --no-save-npz --no-save-npz-aligned
```

## Output Analysis

### Comparison Metrics
- **MAE/MSE/PSNR**: Traditional image quality metrics
- **SSIM/MS-SSIM**: Perceptual similarity metrics (NEW)
- **FRC50**: Fourier Ring Correlation resolution metric
- **Registration offsets**: Alignment quality assessment

### Files Generated
- **`comparison_plot.png`**: Side-by-side visual comparison
- **`comparison_metrics.csv`**: Quantitative metrics
- **`reconstructions.npz`**: Raw reconstruction data
- **`reconstructions_aligned.npz`**: Aligned reconstruction data
- **`debug_images_*/`**: Preprocessing visualization (if enabled)

## Troubleshooting

### Study Script Issues
**Problem**: `run_complete_generalization_study.sh` fails  
**Solutions**:
- Check dataset paths are correct and accessible
- Verify sufficient disk space for multiple trials
- Ensure all dependencies are installed
- Check CUDA/GPU availability for training

### Comparison Failures
**Problem**: Model comparison produces errors  
**Solutions**:
- Verify both model directories contain required files
- Check test data format matches training data format
- Ensure models were trained with compatible configurations

### Registration Problems
**Problem**: Large registration offsets (>20 pixels)  
**Diagnosis**:
- Check model convergence during training
- Verify reconstruction quality
- Compare with known-good test datasets
- Use `--save-debug-images` to visualize alignment

### Memory Issues
**Problem**: Out of memory during studies  
**Solutions**:
- Reduce number of trials for initial testing
- Use smaller training sizes
- Process studies sequentially rather than parallel

## Standard Test Datasets

### Recommended Data
- **Primary**: `tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz`
- **Training**: `datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz`

### Model Structure Reference
```
large_generalization_study_tike_test/
├── train_512/pinn_run/wts.h5.zip
├── train_1024/pinn_run/wts.h5.zip
├── train_2048/pinn_run/wts.h5.zip
└── train_4096/pinn_run/wts.h5.zip
```

## Performance Optimization

### Parallel Processing
- Studies can run multiple trials in parallel
- Use `--num-trials` to control statistical robustness vs. computation time
- Consider cluster resources for large-scale studies

### Resource Management
- Monitor disk space (studies generate large amounts of data)
- Use NPZ export flags to control output size
- Clean up intermediate training outputs if disk space limited

## Cross-References

- **Quick reference**: <doc-ref type="workflow-guide">scripts/studies/QUICK_REFERENCE.md</doc-ref>
- **Detailed guide**: <doc-ref type="workflow-guide">scripts/studies/README.md</doc-ref>
- **Model comparison**: <doc-ref type="guide">docs/MODEL_COMPARISON_GUIDE.md</doc-ref>
- **Training workflow**: <doc-ref type="workflow-guide">scripts/training/CLAUDE.md</doc-ref>
- **Tool selection**: <doc-ref type="guide">docs/TOOL_SELECTION_GUIDE.md</doc-ref>
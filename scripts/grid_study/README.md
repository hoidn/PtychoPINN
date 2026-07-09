# Grid-Based Ptychography Study

Standalone workflow for grid-based data simulation, model training, and evaluation.
Compares PtychoPINN (physics-informed) vs baseline (supervised U-Net) reconstruction.

## Overview

This module implements a modular pipeline based on the deprecated `ptycho_lines.ipynb`
notebook workflow, using the legacy `mk_simdata()` grid-based simulation.

## Experiments

Two parallel experiments with different patch sizes:

| Parameter | N=64 | N=128 |
|-----------|------|-------|
| Probe | Native 64x64 | Upscaled + normalized |
| Training images | 5120 | 5120 |
| Test images | 2048 | 2048 |
| gridsize | 1 | 1 |

## Usage

```bash
# Run full study (both N=64 and N=128)
python scripts/grid_study/run_grid_study.py --output-dir tmp/grid_study

# Run single experiment
python scripts/grid_study/run_grid_study.py --N 128 --output-dir tmp/grid_study_128

# Custom parameters
python scripts/grid_study/run_grid_study.py \
    --N 128 \
    --nepochs 50 \
    --output-dir tmp/custom_study
```

## Module Structure

- `probe_utils.py` - Probe extraction and upscaling with energy normalization
- `grid_data_generator.py` - Wrapper around mk_simdata for grid-based simulation
- `train_models.py` - Training for baseline and PtychoPINN models
- `inference_pipeline.py` - Inference and stitching
- `evaluate_results.py` - SSIM/MS-SSIM metrics and visualization
- `run_grid_study.py` - Main orchestrator CLI

## Known Limitations

1. **gridsize=1**: This comparison tests the physics forward model loss but not
   grouping-based spatial consistency constraints. Results may not generalize to
   gridsize>1 workflows where PINN's advantage is more pronounced.

2. **Stitching quality**: With outer_offset=12 and the border clipping formula,
   only ~6 central pixels per patch contribute to the stitched output. The
   stitched images are low-resolution visualizations, not high-fidelity reconstructions.

3. **Probe source**: Uses experimental Run1084 probe with synthetic "lines" objects.
   This is physically inconsistent but acceptable for algorithm comparison.

## References

- Original notebook: `notebooks/archive/ptycho_lines.ipynb`
- Data generation guide: `docs/DATA_GENERATION_GUIDE.md`
- Grid simulation: `ptycho/diffsim.py::mk_simdata()`

# Single Model Evaluation Script Implementation Plan

**Goal**: Create a comprehensive single-model evaluation script that fills the gap between training (which only saves PNGs) and comparison (which requires two models).

## Stage 1: Core Infrastructure & Basic Evaluation
**Goal**: Implement basic model loading, inference, and metric calculation
**Success Criteria**: Can load a model, run inference, and compute basic metrics (MAE, PSNR, SSIM)
**Tests**: Verify script runs on a trained model and produces metrics.csv

### Checklist:
- [ ] Create `scripts/evaluation/` directory structure
- [ ] Create `scripts/evaluation/evaluate_model.py` with basic structure
- [ ] Implement argument parsing with essential parameters:
  - [ ] `--model-dir` (required): Path to trained model directory
  - [ ] `--test-data` (required): Path to test NPZ file
  - [ ] `--output-dir` (required): Output directory for results
  - [ ] `--model-type` (optional): 'pinn' or 'baseline' (auto-detect if not specified)
- [ ] Implement model loading logic:
  - [ ] Auto-detect model type from directory contents
  - [ ] Use `load_inference_bundle()` for PINN models
  - [ ] Use `tf.keras.models.load_model()` for baseline models
  - [ ] Restore configuration (especially gridsize) from saved params
- [ ] Implement data loading:
  - [ ] Load test data with `load_data()`
  - [ ] Create data container with restored configuration
  - [ ] Handle missing ground truth gracefully
- [ ] Implement basic inference:
  - [ ] Run model prediction with correct input format
  - [ ] Handle different model output formats (PINN vs baseline)
  - [ ] Reassemble patches with `reassemble_position()`
- [ ] Implement basic evaluation:
  - [ ] Call `eval_reconstruction()` with default parameters
  - [ ] Save metrics to CSV file
  - [ ] Print metrics summary to console

**Status**: Not Started

---

## Stage 2: Advanced Alignment & Registration
**Goal**: Add sophisticated alignment and registration capabilities
**Success Criteria**: Properly aligns reconstructions with ground truth before evaluation
**Tests**: Compare metrics with/without registration, verify offset detection

### Checklist:
- [ ] Add alignment parameters:
  - [ ] `--stitch-crop-size` (default: 20): Crop size M for patch stitching
  - [ ] `--skip-registration`: Skip automatic registration
  - [ ] `--phase-align-method` (default: 'plane'): Phase alignment method
- [ ] Implement coordinate-based alignment:
  - [ ] Extract scan coordinates from data container
  - [ ] Call `align_for_evaluation()` to crop to scanned region
  - [ ] Handle different reconstruction sizes properly
- [ ] Implement fine-scale registration:
  - [ ] Call `find_translation_offset()` to detect misalignment
  - [ ] Apply `apply_shift_and_crop()` for sub-pixel alignment
  - [ ] Log detected offsets for transparency
  - [ ] Handle registration failures gracefully
- [ ] Add registration metrics to output:
  - [ ] Include offset_dy and offset_dx in metrics CSV
  - [ ] Add offset information to visualizations

**Status**: Not Started

---

## Stage 3: Comprehensive Metrics & Visualization
**Goal**: Add full metric suite and rich visualizations
**Success Criteria**: Generates all metrics from compare_models.py plus visualizations
**Tests**: Verify all metrics are computed correctly, check visualization quality

### Checklist:
- [ ] Add advanced metric parameters:
  - [ ] `--frc-sigma` (default: 0.0): Gaussian smoothing for FRC
  - [ ] `--ms-ssim-sigma` (default: 1.0): Gaussian smoothing for MS-SSIM
  - [ ] `--save-debug-images`: Save preprocessing visualizations
- [ ] Implement comprehensive metrics:
  - [ ] MAE (Mean Absolute Error) for amplitude and phase
  - [ ] MSE (Mean Squared Error) for amplitude and phase
  - [ ] PSNR (Peak Signal-to-Noise Ratio) for amplitude and phase
  - [ ] SSIM (Structural Similarity) for amplitude and phase
  - [ ] MS-SSIM (Multi-Scale SSIM) for amplitude and phase
  - [ ] FRC50 (Fourier Ring Correlation at 0.5) for amplitude and phase
  - [ ] Full FRC curves saved to separate CSV
- [ ] Create visualization outputs:
  - [ ] Reconstruction amplitude PNG
  - [ ] Reconstruction phase PNG
  - [ ] Ground truth amplitude PNG (if available)
  - [ ] Ground truth phase PNG (if available)
  - [ ] Comparison plot (2x2 grid: recon vs GT, amp vs phase)
  - [ ] Add color scale percentile controls (`--p-min`, `--p-max`)
  - [ ] Optional manual vmin/vmax controls for phase
- [ ] Add timing information:
  - [ ] Measure and report inference time
  - [ ] Include computation_time_s in metrics CSV
  - [ ] Compare with training time if available in model metadata

**Status**: Not Started

---

## Stage 4: Data Export & Compatibility
**Goal**: Add NPZ export and ensure compatibility with existing workflows
**Success Criteria**: Can export data for downstream analysis, integrates with existing tools
**Tests**: Verify NPZ files contain correct data, test integration with analysis scripts

### Checklist:
- [ ] Add data export parameters:
  - [ ] `--save-npz` (default: True): Save raw reconstruction NPZ
  - [ ] `--no-save-npz`: Disable NPZ export
  - [ ] `--save-npz-aligned` (default: True): Save aligned NPZ
  - [ ] `--no-save-npz-aligned`: Disable aligned NPZ export
- [ ] Implement NPZ export:
  - [ ] Create unified NPZ with amplitude, phase, and complex arrays
  - [ ] Include both raw and aligned versions (if registration performed)
  - [ ] Add metadata about alignment offsets
  - [ ] Generate metadata.txt describing NPZ contents
- [ ] Add sampling controls for large datasets:
  - [ ] `--n-test-groups`: Number of test groups to process
  - [ ] `--n-test-subsample`: Subsample images from test data
  - [ ] `--test-subsample-seed`: Random seed for reproducibility
- [ ] Ensure compatibility:
  - [ ] Support both PINN and baseline models automatically
  - [ ] Handle different gridsize configurations properly
  - [ ] Compatible with downstream analysis scripts
  - [ ] Integrate with existing logging infrastructure

**Status**: Not Started

---

## Stage 5: Polish, Testing & Documentation
**Goal**: Create production-ready script with tests and documentation
**Success Criteria**: Script is robust, well-tested, and documented
**Tests**: Unit tests pass, integration tests with real models work, documentation is clear

### Checklist:
- [ ] Add robust error handling:
  - [ ] Validate all input paths exist
  - [ ] Check model directory structure
  - [ ] Verify data format compatibility
  - [ ] Handle missing ground truth gracefully
  - [ ] Provide clear error messages
- [ ] Implement logging:
  - [ ] Use centralized logging with `setup_logging()`
  - [ ] Add `--quiet`, `--verbose` flags
  - [ ] Log all important steps and decisions
  - [ ] Save complete debug.log
- [ ] Create test suite:
  - [ ] Unit test for model type detection
  - [ ] Unit test for metric calculations
  - [ ] Integration test with sample model
  - [ ] Test with different gridsize configurations
  - [ ] Test with/without ground truth
- [ ] Write documentation:
  - [ ] Create `scripts/evaluation/README.md`
  - [ ] Add docstrings to all functions
  - [ ] Include usage examples
  - [ ] Document output file formats
  - [ ] Add to CLAUDE.md workflows section
- [ ] Create entry point:
  - [ ] Add `ptycho_evaluate` command to `pyproject.toml`
  - [ ] Ensure script is executable
  - [ ] Test command-line interface

**Status**: Not Started

---

## Implementation Notes

### Key Dependencies to Import:
```python
from ptycho.workflows.components import load_data, create_ptycho_data_container, logger, load_inference_bundle
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params as p
from ptycho.tf_helper import reassemble_position
from ptycho.evaluation import eval_reconstruction
from ptycho.image.cropping import align_for_evaluation
from ptycho.image.registration import find_translation_offset, apply_shift_and_crop
from ptycho.cli_args import add_logging_arguments, get_logging_config
from ptycho.log_config import setup_logging
```

### Model Type Detection Logic:
```python
def detect_model_type(model_dir: Path) -> str:
    """Auto-detect whether directory contains PINN or baseline model."""
    # Check for PINN model artifacts
    if (model_dir / "wts.h5.zip").exists():
        return "pinn"
    # Check for baseline model
    for root, dirs, files in os.walk(model_dir):
        if "baseline_model.h5" in files:
            return "baseline"
    raise ValueError(f"Could not detect model type in {model_dir}")
```

### Gridsize Handling:
```python
# Critical: Load model FIRST to restore configuration
model = load_model(args.model_dir)
restored_gridsize = p.cfg.get('gridsize', 1)
# Use restored gridsize when loading data
test_data = load_data(str(args.test_data), n_images=args.n_test_groups)
```

### Output Structure:
```
output_dir/
├── logs/
│   └── debug.log                    # Complete execution log
├── evaluation_metrics.csv           # All quantitative metrics
├── frc_curves.csv                   # Detailed FRC curves
├── reconstruction_amplitude.png     # Amplitude visualization
├── reconstruction_phase.png         # Phase visualization
├── ground_truth_amplitude.png       # GT amplitude (if available)
├── ground_truth_phase.png          # GT phase (if available)
├── comparison_plot.png             # 2x2 comparison grid
├── reconstruction.npz              # Raw reconstruction data
├── reconstruction_aligned.npz      # Aligned reconstruction (if registered)
└── reconstruction_metadata.txt     # NPZ content description
```

### Testing Strategy:
1. Test with gridsize=1 PINN model
2. Test with gridsize=2 PINN model  
3. Test with baseline model
4. Test without ground truth
5. Test with different sampling parameters
6. Test registration on/off

### Integration Points:
- Should work seamlessly after `ptycho_train`
- Output format compatible with `aggregate_and_plot_results.py`
- NPZ files usable by visualization scripts
- Metrics CSV compatible with comparison workflows

---

## Success Metrics

The implementation is successful when:
1. Users can evaluate a single model with one command
2. All metrics match those from compare_models.py (when comparing same model)
3. Script handles edge cases gracefully
4. Documentation is clear and comprehensive
5. Tests provide good coverage
6. Performance is reasonable (inference + evaluation < 5 minutes for typical dataset)

---

## Risk Mitigation

### Potential Issues & Solutions:
1. **Gridsize mismatch**: Always restore configuration from model before data loading
2. **Memory issues with large datasets**: Implement batch processing if needed
3. **Registration failures**: Provide fallback to unregistered evaluation
4. **Missing ground truth**: Clearly indicate when metrics cannot be computed
5. **Model format changes**: Maintain backward compatibility, version detection

---

## Timeline Estimate

- Stage 1: 2-3 hours (core functionality)
- Stage 2: 2 hours (alignment & registration)
- Stage 3: 2-3 hours (metrics & visualization)
- Stage 4: 1-2 hours (data export)
- Stage 5: 2-3 hours (polish & documentation)

**Total: 9-13 hours of focused development**

---

## Next Steps

1. Create the directory structure
2. Start with Stage 1 core implementation
3. Test each stage before moving to the next
4. Get user feedback after Stage 3
5. Refine based on testing and feedback
# Debug Report: GridSize=2 Comparison Script Failure

**Date:** 2025-07-19  
**Issue:** Comparison script completes training but fails to generate comparison metrics/plots  
**Status:** Active Investigation  

## 🔍 Problem Summary

### Symptoms
- Both PINN and Baseline models train successfully (50 epochs each)
- Comparison script (`compare_models.py`) fails to generate expected output files
- Process terminates silently without creating `comparison_metrics.csv` or visualization plots
- Checkpoint warnings appear during baseline model loading but may not be fatal

### Expected Outputs (Missing)
```
comparison_gs2_10k_train/
├── comparison_metrics.csv      ❌ Missing
├── comparison_plot.png         ❌ Missing  
├── reconstructions.npz         ❌ Missing
├── pinn_frc_curves.csv         ❌ Missing
└── baseline_frc_curves.csv     ❌ Missing
```

### Actual Outputs (Present)
```
comparison_gs2_10k_train/
├── pinn_run/
│   └── wts.h5.zip              ✅ Present (30MB)
└── baseline_run/
    └── 07-18-2025-23.57.12_baseline_gs2/
        └── baseline_model.h5   ✅ Present
```

## 🚀 Reproduction Steps

### Workflow Executed
```bash
# Step 1: Modify configuration for gridsize=2
cp configs/comparison_config.yaml configs/comparison_config.yaml.bak
# Edit configs/comparison_config.yaml: set gridsize: 2

# Step 2: Run comparison workflow
./scripts/run_comparison.sh \
    datasets/fly64/fly001_64_train_converted.npz \
    datasets/fly64/fly001_64_train_converted.npz \
    comparison_gs2_10k_train \
    --n-train-images 2500 \
    --n-test-images 1024

# Step 3: Manual comparison (after training completed)
python scripts/compare_models.py \
    --pinn_dir comparison_gs2_10k_train/pinn_run \
    --baseline_dir comparison_gs2_10k_train/baseline_run \
    --test_data datasets/fly64/fly001_64_train_converted.npz \
    --output_dir comparison_gs2_10k_train \
    --save-npz
```

### Parameters Used
- **Training Images:** 2500 groups (gridsize=2) = ~10,000 total patterns
- **Test Images:** 1024 groups (gridsize=2) = ~4,096 total patterns  
- **Dataset:** `datasets/fly64/fly001_64_train_converted.npz`
- **GridSize:** 2 (grouping-aware subsampling)
- **Epochs:** 50

## 📋 Observed Behavior

### Successful Phases
1. ✅ **Configuration Setup**: GridSize=2 config applied correctly
2. ✅ **PINN Training**: Completed successfully, model saved to `wts.h5.zip`
3. ✅ **Baseline Training**: Completed successfully, model saved to `baseline_model.h5`
4. ✅ **Model Detection**: Comparison script finds both models correctly
5. ✅ **Data Loading**: Test data loads with grouping-aware subsampling
6. ✅ **PINN Model Loading**: Loads without issues
7. ✅ **PINN Inference**: Progress bar shows 322/322 steps completed

### Failure Point
8. ⚠️ **Baseline Model Loading**: Shows checkpoint warnings but claims success
9. ❌ **Comparison Generation**: No output files created, process terminates

### Key Log Messages
```
2025-07-19 00:09:09,673 - INFO - Found baseline model at: comparison_gs2_10k_train/baseline_run/07-18-2025-23.57.12_baseline_gs2/baseline_model.h5

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer._variables.87
WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer._variables.88
WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).keras_api.metrics.0.total
WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).keras_api.metrics.0.count
[Multiple similar warnings for metrics.1-3]
```

## 🤔 Current Theories

### ✅ CONFIRMED ROOT CAUSE: Data Channel Mismatch
**Hypothesis:** Baseline model expects 4 input channels but comparison script attempts 1-channel prediction in some code paths

**Evidence:**
- Baseline model architecture: Input shape `(None, 64, 64, 4)` - expects 4 channels
- Data container correctly produces: `X shape: (10304, 64, 64, 4)` - 4 channels ✅
- Error: `expected axis -1 of input shape to have value 4, but received input with shape (None, 64, 64, 1)`
- Independent baseline prediction test fails with same error

**Resolution Status:** ✅ **SOLVED** - Baseline model works with 4-channel input, script has configuration bug

**Fix Required:** Ensure comparison script uses consistent gridsize configuration for both data loading and model inference

### Alternative Theories

#### Theory 2: Silent Exception in Comparison Pipeline
**Hypothesis:** Error occurs downstream in comparison generation but isn't logged

**Evidence:**
- No explicit error messages about comparison failure
- Process terminates cleanly without output files

#### Theory 3: GridSize=2 Configuration Mismatch  
**Hypothesis:** GridSize=2 config affects model compatibility or data expectations

**Evidence:**
- This is first attempt at gridsize=2 comparison workflow
- Training logs showed gridsize=1 despite config having gridsize=2

#### Theory 4: Resource/Permission Issue
**Hypothesis:** Script fails to write output files due to disk space, permissions, or memory

**Evidence:**
- No explicit I/O errors in logs
- Output directory exists and is writable

## 🔬 Investigation Areas

### High Priority
1. **Exception Handling in compare_models.py**
   - Check for try/catch blocks that might suppress errors
   - Add verbose logging to identify silent failure points
   - Lines 565-800+ where main comparison logic runs

2. **Baseline Model Loading Validation**
   - Verify if checkpoint warnings are actually fatal
   - Test baseline model loading independently
   - Check if model architecture matches expectations

3. **Comparison Pipeline Steps**
   - Verify each step in comparison generation pipeline
   - Check metric calculation, registration, visualization steps
   - Look for resource constraints or memory issues

### Medium Priority  
4. **GridSize=2 Compatibility**
   - Verify if comparison script handles gridsize=2 data correctly
   - Check data shape expectations in comparison pipeline
   - Test if training actually used gridsize=2 as intended

5. **Configuration Validation**
   - Verify configuration file changes took effect
   - Check if comparison script reads configuration correctly
   - Validate parameter interpretation consistency

### Low Priority
6. **Environment Dependencies**
   - Check matplotlib/visualization library functionality
   - Verify file system permissions and disk space
   - Test with smaller datasets to isolate resource issues

## 🧪 Proposed Experiments

### Experiment 1: Minimal Baseline Model Test
```bash
# Test baseline model loading independently
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('comparison_gs2_10k_train/baseline_run/07-18-2025-23.57.12_baseline_gs2/baseline_model.h5')
print('Model loaded successfully')
print(f'Model input shape: {model.input_shape}')
print(f'Model output shape: {model.output_shape}')
"
```

### Experiment 2: Comparison Script with Debug Logging
```bash
# Add debug prints to compare_models.py around line 565-800
# Run with explicit error catching and logging
python scripts/compare_models.py \
    --pinn_dir comparison_gs2_10k_train/pinn_run \
    --baseline_dir comparison_gs2_10k_train/baseline_run \
    --test_data datasets/fly64/fly001_64_train_converted.npz \
    --output_dir debug_comparison \
    --save-npz 2>&1 | tee debug_comparison.log
```

### Experiment 3: GridSize=1 Comparison (Control)
```bash
# Test if comparison works with gridsize=1 configuration
# Restore original config and run same workflow
mv configs/comparison_config.yaml.bak configs/comparison_config.yaml
./scripts/run_comparison.sh \
    datasets/fly64/fly001_64_train_converted.npz \
    datasets/fly64/fly001_64_train_converted.npz \
    comparison_gs1_control \
    --n-train-images 2500 \
    --n-test-images 1024
```

## 📊 Resolution & Next Steps

### ✅ Root Cause Identified
**Issue:** Configuration inconsistency in comparison script
- PINN model loads with gridsize=1 configuration → expects 1-channel input
- Data container created with gridsize=2 configuration → produces 4-channel input  
- Baseline model trained with gridsize=2 → expects 4-channel input
- **Result:** Input shape mismatch causes baseline model prediction to fail

### 🔧 Solution Required
**Fix:** Ensure consistent gridsize configuration throughout comparison script
1. **Check model training configs**: Verify both models were trained with same gridsize setting
2. **Fix data loading**: Ensure comparison script uses correct gridsize when creating data container
3. **Validate input shapes**: Add assertions to verify input/model compatibility before prediction

### Immediate Actions
1. ✅ **Model Loading Test**: Both models load successfully independently
2. ✅ **Input Format Test**: Baseline model works with correct 4-channel input
3. 🔄 **Configuration Fix**: Update comparison script to use consistent gridsize settings
4. 🔄 **Validation Test**: Re-run comparison with fixed configuration

## 📚 Related Documentation

- **Commands Reference**: `docs/COMMANDS_REFERENCE.md` - Contains the gridsize=2 workflow
- **Model Comparison Guide**: `docs/MODEL_COMPARISON_GUIDE.md` - Expected comparison outputs
- **Developer Guide**: `docs/DEVELOPER_GUIDE.md` - Architecture and debugging tips
- **GridSize Issues**: `docs/archive/GRIDSIZE_INFERENCE_GOTCHAS.md` - Known gridsize pitfalls

## 🎯 Success Criteria

**Root Cause Identified When:**
- [x] Specific line/function where failure occurs identified ✅ (Line 586: baseline_model.predict())
- [x] Minimal reproduction case created ✅ (Input shape mismatch test)
- [ ] Fix implemented and tested 🔄 (Configuration consistency fix needed)
- [ ] Prevention measures added to avoid recurrence 🔄 (Input validation checks)

**Issue Resolved When:**
- [ ] `comparison_metrics.csv` file generated with metric comparisons 🔄
- [ ] Comparison visualization plot created 🔄
- [ ] NPZ files with reconstruction data saved 🔄
- [ ] No silent failures in comparison pipeline 🔄
- [ ] Workflow documented and reproducible 🔄
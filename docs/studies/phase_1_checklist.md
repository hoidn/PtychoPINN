# Phase 1 Implementation Checklist: Enhance Single-Run Capability

**Phase Goal:** Update the core `run_comparison.sh` script to support configurable training and testing set sizes.

**Target Deliverable:** A modified `run_comparison.sh` that correctly accepts and utilizes `--n-train-images` and `--n-test-images` arguments.

---

## 📝 **IMPLEMENTATION TASKS**

### **Task 1: Update run_comparison.sh Script Arguments**

- [ ] **1.1:** Update the usage message and help text in `run_comparison.sh` to document the new optional arguments
  - [ ] Modify lines 16-18 to show the new argument format
  - [ ] Add examples showing usage with the new arguments

- [ ] **1.2:** Add argument parsing logic for the new named parameters
  - [ ] Add parsing for `--n-train-images` argument (with validation)
  - [ ] Add parsing for `--n-test-images` argument (with validation)
  - [ ] Ensure backward compatibility when these arguments are not provided
  - [ ] Add validation to ensure arguments are positive integers if provided

- [ ] **1.3:** Store the parsed arguments in shell variables
  - [ ] Create `N_TRAIN_IMAGES` variable (default to empty for backward compatibility)
  - [ ] Create `N_TEST_IMAGES` variable (default to empty for backward compatibility)

### **Task 2: Forward Arguments to Training Scripts**

- [ ] **2.1:** Modify PtychoPINN training command (lines 72-77)
  - [ ] Add conditional logic to pass `--n_images` to `train.py` when `N_TRAIN_IMAGES` is specified
  - [ ] Verify the argument name matches what `train.py` expects

- [ ] **2.2:** Modify Baseline training command (lines 86-91)
  - [ ] Add conditional logic to pass `--n_images` to `run_baseline.py` when `N_TRAIN_IMAGES` is specified
  - [ ] Verify the argument name matches what `run_baseline.py` expects

- [ ] **2.3:** Update output logging to show the number of images being used
  - [ ] Add echo statements to display the effective number of training/test images
  - [ ] Ensure the log output is clear about when custom image counts are being used

### **Task 3: Verify Python Script Compatibility**

- [ ] **3.1:** Examine `scripts/training/train.py` to confirm `n_images` parameter handling
  - [ ] Check that the script accepts `--n_images` command line argument
  - [ ] Verify it correctly passes this to the configuration system
  - [ ] Test that it overrides the config file value when provided

- [ ] **3.2:** Examine `scripts/run_baseline.py` to confirm `n_images` parameter handling
  - [ ] Check that the script accepts `--n_images` command line argument  
  - [ ] Verify it correctly passes this to the configuration system
  - [ ] Test that it overrides the config file value when provided

- [ ] **3.3:** Verify the configuration precedence logic
  - [ ] Confirm command-line arguments override config file values
  - [ ] Test that the `n_images` parameter flows correctly through the data loading pipeline

### **Task 4: Handle Test Dataset Size for Comparison**

- [ ] **4.1:** Examine `scripts/compare_models.py` for test data size handling
  - [ ] Check if it needs explicit control over test dataset size
  - [ ] Determine if `N_TEST_IMAGES` should be passed to the comparison script

- [ ] **4.2:** Update comparison command if needed
  - [ ] Add `--n_test_images` argument to comparison script call if supported
  - [ ] Otherwise, document that test size is controlled by the dataset itself

### **Task 5: Testing and Validation**

- [ ] **5.1:** Create test scenarios
  - [ ] Test backward compatibility: `./run_comparison.sh train.npz test.npz output_dir`
  - [ ] Test with training size only: `./run_comparison.sh train.npz test.npz output_dir --n-train-images 512`
  - [ ] Test with both sizes: `./run_comparison.sh train.npz test.npz output_dir --n-train-images 512 --n-test-images 1000`

- [ ] **5.2:** Verify log output correctness
  - [ ] Check that training logs show the correct number of images loaded
  - [ ] Verify that baseline training logs show the correct number of images loaded
  - [ ] Confirm the main script output clearly indicates when custom sizes are used

- [ ] **5.3:** Run end-to-end test
  - [ ] Execute a complete run with known-good data and small image counts
  - [ ] Verify both PtychoPINN and baseline complete successfully
  - [ ] Check that comparison analysis generates the expected outputs

### **Task 6: Error Handling and Edge Cases**

- [ ] **6.1:** Add robust error handling
  - [ ] Validate that requested `n_train_images` doesn't exceed dataset size
  - [ ] Validate that requested `n_test_images` doesn't exceed dataset size
  - [ ] Provide clear error messages for invalid arguments

- [ ] **6.2:** Handle edge cases
  - [ ] Test behavior when `n_images` is larger than the dataset
  - [ ] Test with very small values (e.g., `n_images=10`)
  - [ ] Verify proper handling of mixed argument styles

### **Task 7: Documentation and Cleanup**

- [ ] **7.1:** Update script documentation
  - [ ] Add clear comments explaining the new functionality
  - [ ] Update the header comment with new usage examples

- [ ] **7.2:** Verify code style and consistency
  - [ ] Ensure new code follows the existing bash script style
  - [ ] Check that variable naming is consistent
  - [ ] Validate that error handling follows existing patterns

---

## ✅ **PHASE COMPLETION CRITERIA**

**The phase is complete when:**

1. [ ] All tasks above are marked as complete
2. [ ] The command `./run_comparison.sh datasets/fly/fly001_transposed.npz datasets/fly/fly001_transposed.npz test_output --n-train-images 512 --n-test-images 1000` executes successfully
3. [ ] Log output clearly shows that 512 images were used for training
4. [ ] Both PtychoPINN and baseline models complete training
5. [ ] Comparison analysis generates the expected output files
6. [ ] Backward compatibility is maintained (old usage still works)

**Success Verification Command:**
```bash
# Test the enhanced script
./scripts/run_comparison.sh \
    datasets/fly/fly001_transposed.npz \
    datasets/fly/fly001_transposed.npz \
    phase1_test_output \
    --n-train-images 512 \
    --n-test-images 1000
```

---

## 📋 **NOTES & CONSIDERATIONS**

- **Backward Compatibility:** The script must continue to work with the original 3-7 argument format
- **Argument Order:** New named arguments should be parsed after positional arguments
- **Config Override:** Command-line image counts should override config file values
- **Logging:** Clear indication in logs when custom image counts are being used
- **Error Messages:** Helpful error messages for common mistakes (e.g., exceeding dataset size)
# Phase 1 Checklist: Shuffle Dataset Tool Creation

**Project:** Spatially-Biased Randomized Sampling Study  
**Phase:** 1 of 2  
**Goal:** Create and validate the core `shuffle_dataset_tool.py` script that can randomize NPZ datasets while preserving data relationships.

---

## 🎯 **PHASE 1 DELIVERABLE**
A working `scripts/tools/shuffle_dataset_tool.py` that correctly shuffles per-scan arrays in unison while preserving global arrays.

---

## ✅ **IMPLEMENTATION TASKS**

### **Task 1.1: Create Basic Tool Structure**
- [ ] Create `scripts/tools/shuffle_dataset_tool.py` file
- [ ] Add command-line argument parsing with `argparse`
  - [ ] `--input-file` (required): Path to input NPZ file
  - [ ] `--output-file` (required): Path to output NPZ file  
  - [ ] `--seed` (optional): Random seed for reproducible shuffling
  - [ ] `--dry-run` (optional): Show what would be shuffled without writing output
- [ ] Add basic error handling for file I/O operations
- [ ] Add logging setup with informative messages

### **Task 1.2: Implement NPZ Loading and Analysis**
- [ ] Load input NPZ file using `np.load()`
- [ ] Identify per-scan arrays vs global arrays based on first dimension
  - [ ] Per-scan arrays: First dimension matches number of images (e.g., `diffraction`, `xcoords`, `ycoords`)
  - [ ] Global arrays: Fixed size regardless of number of images (e.g., `objectGuess`, `probeGuess`)
- [ ] Log discovered array types and shapes for verification
- [ ] Validate that all per-scan arrays have consistent first dimension size

### **Task 1.3: Implement Shuffling Logic**
- [ ] Generate random permutation indices using `np.random.permutation()`
- [ ] Apply permutation to all per-scan arrays simultaneously
- [ ] Leave global arrays unchanged
- [ ] Ensure all per-scan arrays are shuffled with identical permutation order

### **Task 1.4: Implement Output Generation**
- [ ] Create output NPZ file with shuffled per-scan arrays and original global arrays
- [ ] Preserve all original array names and data types
- [ ] Add metadata comment or array indicating shuffling was applied
- [ ] Verify output file can be loaded successfully

### **Task 1.5: Create Validation Test Framework**
- [ ] Create test function that generates synthetic NPZ data
  - [ ] Include mock `diffraction` array (per-scan)
  - [ ] Include mock `xcoords`, `ycoords` arrays (per-scan)
  - [ ] Include mock `objectGuess`, `probeGuess` arrays (global)
  - [ ] Use known, predictable values for easy verification
- [ ] Test that per-scan arrays are reordered consistently
- [ ] Test that global arrays remain unchanged
- [ ] Test that data relationships are preserved (e.g., `diffraction[i]` still corresponds to `xcoords[i]`)
- [ ] Test edge cases (empty arrays, single-element arrays)

### **Task 1.6: Add Dry-Run and Verification Features**
- [ ] Implement `--dry-run` mode that shows analysis without writing files
- [ ] Add verification output showing:
  - [ ] Which arrays were identified as per-scan vs global
  - [ ] Sample indices before and after shuffling
  - [ ] Confirmation that relationships are preserved
- [ ] Add seed support for reproducible testing

### **Task 1.7: Error Handling and Edge Cases**
- [ ] Handle missing required arrays gracefully
- [ ] Handle NPZ files with unexpected structure
- [ ] Validate input file exists and is readable
- [ ] Validate output directory is writable
- [ ] Add meaningful error messages for common failure modes

---

## 🧪 **VALIDATION TESTS**

### **Unit Test 1: Basic Shuffling**
- [ ] Create synthetic NPZ with 10 data points
- [ ] Run shuffling tool
- [ ] Verify per-scan arrays are reordered but global arrays unchanged
- [ ] Verify all 10 indices appear exactly once in shuffled output

### **Unit Test 2: Relationship Preservation**
- [ ] Create synthetic data where `xcoords[i] = i` and `diffraction[i, 0, 0] = i`
- [ ] Run shuffling tool
- [ ] Verify that for any index `j` in output: `xcoords[j] == diffraction[j, 0, 0]`

### **Unit Test 3: Reproducibility**
- [ ] Run shuffling tool twice with same seed
- [ ] Verify outputs are identical
- [ ] Run with different seeds
- [ ] Verify outputs are different

---

## ✅ **COMPLETION CRITERIA**

**This phase is complete when:**
- [ ] All implementation tasks above are checked off
- [ ] All validation tests pass
- [ ] The tool runs successfully on synthetic test data
- [ ] Tool includes proper command-line interface with help text
- [ ] Code includes appropriate comments and docstrings

**Ready for Phase 2 when:**
- [ ] `shuffle_dataset_tool.py` exists and passes all unit tests
- [ ] Tool demonstrates correct shuffling behavior on known test data
- [ ] All edge cases and error conditions are handled gracefully

---

## 📝 **NOTES & DECISIONS**

### **Design Decisions:**
- **Array Classification Strategy:** Use first dimension size to distinguish per-scan from global arrays
- **Permutation Strategy:** Generate single permutation and apply to all per-scan arrays for consistency
- **Metadata Strategy:** Preserve all original array names and types, add minimal shuffling indicator

### **Implementation Notes:**
- Arrays with first dimension != n_images are considered global (e.g., `objectGuess`, `probeGuess`)
- Arrays with first dimension == n_images are considered per-scan (e.g., `diffraction`, `xcoords`, `ycoords`)
- Tool should be agnostic to specific array names to work with different dataset formats

---

**Last Updated:** [Date when checklist was created/modified]  
**Next Phase:** Final Phase: Validation & Documentation (`phase_final_checklist_sampling_study.md`)
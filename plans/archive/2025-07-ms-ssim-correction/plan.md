# R&D Plan: Correcting the MS-SSIM Implementation

## 🎯 OBJECTIVE & HYPOTHESIS

**Project/Initiative Name:** MS-SSIM Implementation Correction

**Problem Statement:** The current Multi-Scale SSIM (ms_ssim) function in `ptycho/evaluation.py` is mathematically incorrect, leading to unreliable and often inflated metric values. This prevents fair comparison of perceptual image quality between models.

**Proposed Solution / Hypothesis:** By replacing the flawed ms_ssim function with a correct implementation based on the standard algorithm, we will produce valid MS-SSIM metrics that accurately reflect multi-scale structural similarity, enabling robust model evaluation.

**Scope & Deliverables:**
- A corrected `ms_ssim` function in `ptycho/evaluation.py`
- A new unit test specifically for the `ms_ssim` function to prevent future regressions
- A re-run of a model comparison to generate corrected metrics and verify the fix
- Updated documentation if necessary

## 🔬 EXPERIMENTAL DESIGN & CAPABILITIES

**Core Capabilities (Must-have for this cycle):**
- **Correct MS-SSIM Logic:** The new `ms_ssim` function must properly separate luminance, contrast, and structure components and combine them using the standard weighted product formula
- **Unit Testing:** A new test file `tests/test_evaluation.py` will be created with a test case for `ms_ssim` that verifies its behavior with known inputs
- **Integration & Verification:** The `scripts/compare_models.py` workflow must be run to generate a new `comparison_metrics.csv` file, which will be inspected to confirm the corrected MS-SSIM values are reasonable (i.e., typically MS-SSIM ≤ SSIM)

## 🛠️ TECHNICAL IMPLEMENTATION DETAILS

**Key Modules to Modify:**
- `ptycho/evaluation.py`: To replace the existing `ms_ssim` function
- `tests/test_evaluation.py`: To be created for unit testing the new implementation

**Key Dependencies / APIs:**
- **External:** `skimage.metrics.structural_similarity(..., full=True)` will be used to extract the full SSIM map, which is necessary for separating contrast and structure components. `scipy.ndimage.gaussian_filter` will be used for downsampling between scales.

**Implementation Specifications:**
- **Formula:** The implementation will follow the formula `MS-SSIM = (l_M^α_M) * Π(c_j^β_j * s_j^γ_j)`. For simplicity and standard practice, we will use equal weights for contrast and structure (β_j = γ_j).
- **Downsampling:** A 2x downsampling between each scale will be performed using a Gaussian filter followed by subsampling, as is standard practice.
- **Weights:** The standard weights `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]` will be used.

## ✅ VALIDATION & VERIFICATION PLAN

**Unit Tests:**
- **Test ms_ssim self-comparison:** `ms_ssim(image, image)` must return a value extremely close to 1.0
- **Test ms_ssim against SSIM:** For a simple pair of images, verify that `ms_ssim(img1, img2)` returns a value less than or equal to `ssim(img1, img2)`
- **Test ms_ssim with known values:** Compare the output against a trusted third-party implementation (if available) or pre-computed values for a reference image pair

**Integration / Regression Tests:**
- **Run compare_models.py:** Execute the full comparison workflow using the same inputs as the run that produced the faulty metrics
- **Inspect comparison_metrics.csv:**
  - Verify that the new ms_ssim values are present
  - Confirm that ms_ssim is now less than or equal to ssim for both PtychoPINN and Baseline models
  - Check that other metrics (MAE, PSNR, etc.) have not been unintentionally altered

**Success Criteria (How we know we're done):**
- The `ms_ssim` function in `ptycho/evaluation.py` is replaced with the new, correct implementation
- All new unit tests in `tests/test_evaluation.py` pass
- The re-run of `scripts/compare_models.py` produces a `comparison_metrics.csv` where MS-SSIM values are no longer greater than SSIM values
- The project's overall test suite passes, ensuring no regressions were introduced

## 🚀 CURRENT STATUS

**Status:** Planning Phase  
**Priority:** High  
**Estimated Effort:** 2-3 hours  
**Dependencies:** None  

## 📋 IMPLEMENTATION CHECKLIST

### Phase 1: Research & Design
- [ ] Review MS-SSIM literature and standard implementation
- [ ] Identify the exact mathematical formula to implement
- [ ] Design the function signature and parameters

### Phase 2: Implementation
- [ ] Implement the corrected `ms_ssim` function
- [ ] Create unit tests in `tests/test_evaluation.py`
- [ ] Verify the implementation against known test cases

### Phase 3: Integration & Validation
- [ ] Run integration tests with `scripts/compare_models.py`
- [ ] Validate that MS-SSIM ≤ SSIM in all test cases
- [ ] Ensure no regressions in other metrics

### Phase 4: Documentation & Cleanup
- [ ] Update function documentation
- [ ] Add comments explaining the MS-SSIM formula
- [ ] Update any related documentation

## 📊 IMPACT ASSESSMENT

**Before Fix:**
- MS-SSIM values frequently > SSIM values (mathematically impossible)
- Unreliable perceptual quality metrics
- Difficulty in fair model comparison

**After Fix:**
- Mathematically correct MS-SSIM implementation
- Reliable multi-scale structural similarity metrics
- Accurate model evaluation capabilities
- Prevention of future metric calculation errors

## 🔗 RELATED ISSUES

- **Root Cause:** Mathematical error in MS-SSIM formula implementation
- **Discovered During:** Model comparison analysis showing MS-SSIM > SSIM
- **Impact:** Affects all model evaluation workflows using MS-SSIM
### **R&D Plan: Implementing an Image Registration Step for Evaluation**

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Project/Initiative Name:** Add Robust Image Registration to Evaluation Pipeline

**Problem Statement:** Reconstructions from different models (e.g., PtychoPINN vs. Baseline) may have small, systematic translational shifts relative to each other and to the ground truth, leading to inaccurate evaluation metrics.

**Proposed Solution / Hypothesis:** By implementing a cross-correlation-based image registration step within the evaluation workflow, we can computationally align the images before calculating metrics. We hypothesize this will correct for any learned offsets and produce more accurate and reliable PSNR and FRC values, resolving the issue of identical FRCs.

**Scope & Deliverables:**
1.  A new, standalone function `register_and_align` in a new module, `ptycho/image/registration.py`.
2.  Modification of `scripts/compare_models.py` to call this new function after reassembly and before evaluation.
3.  Updated documentation in `DEVELOPER_GUIDE.md` explaining the purpose of the registration step.

---

## 🔬 **EXPERIMENTAL DESIGN & CAPABILITIES**

**Core Capabilities (Must-have for this cycle):**
1.  **Capability 1: Cross-Correlation Registration Function:**
    *   Create a function `find_translation_offset(image, reference_image)` that uses phase correlation to find the `(dy, dx)` pixel shift of `image` relative to `reference_image`.
    *   The function should handle complex-valued images by operating on their magnitudes.
2.  **Capability 2: Image Shifting and Cropping Function:**
    *   Create a function `apply_shift_and_crop(image, reference_image, offset)` that takes an image, a reference, and the calculated offset.
    *   It should shift `image` by `(dy, dx)` and then crop both `image` and `reference_image` to their maximum overlapping area to ensure they have identical final shapes.
3.  **Capability 3: Integration into Evaluation Workflow:**
    *   Modify `scripts/compare_models.py` to use these new functions. The new workflow will be:
        1.  Reassemble PINN and Baseline reconstructions.
        2.  Register PINN reconstruction against the ground truth to get `offset_pinn`.
        3.  Register Baseline reconstruction against the ground truth to get `offset_baseline`.
        4.  Apply the shifts and crop all three images (PINN, Baseline, GT) to a common, valid, overlapping region.
        5.  Pass these perfectly aligned, same-sized images to `eval_reconstruction`.

---

## 🛠️ **TECHNICAL IMPLEMENTATION DETAILS**

**Key Modules to Create/Modify:**
*   `ptycho/image/registration.py` (New File): To house the core registration logic.
*   `scripts/compare_models.py` (Modify): To orchestrate the new registration and alignment steps.
*   `ptycho/evaluation.py` (No change needed): This function will now receive perfectly aligned images, so it doesn't need to be modified further.

**Key Dependencies / APIs:**
*   **External:** `scipy.signal.fftconvolve` for fast, normalized cross-correlation. `skimage.registration.phase_cross_correlation` is an even better, more direct choice if available. `numpy` for array manipulation.

**Data Requirements:**
*   **Input:** Two complex-valued 2D NumPy arrays (the image to align and the reference image).
*   **Output:** Two complex-valued 2D NumPy arrays of the *exact same shape*, ready for pixel-wise metric calculation.

---

## ✅ **VALIDATION & VERIFICATION PLAN**

**Unit Tests (for `ptycho/image/registration.py`):**
*   [ ] **Test Case 1:** Create a reference image and a second image that is a known `(dy, dx)` shift of the first. Assert that `find_translation_offset` recovers the correct `(dy, dx)`.
*   [ ] **Test Case 2:** Test `apply_shift_and_crop` with a known shift. Verify that the output images are correctly shifted, have identical shapes, and that no data from outside the overlapping region is present.
*   [ ] **Test Case 3:** Test with images that are already perfectly aligned. The function should return an offset of `(0, 0)` and not crop the images unnecessarily.

**Integration / Regression Tests:**
*   [ ] Modify `scripts/compare_models.py` and run it on the non-square dataset where the 12-pixel shift was observed.
*   **Success Criteria:**
    1.  The script completes without any errors.
    2.  The final `comparison_plot.png` shows all three images (PINN, Baseline, GT) are now visually aligned.
    3.  The `comparison_metrics.csv` file is generated, and the `frc50` values for PtychoPINN and Baseline are now **different and plausible**.

This plan provides a clear path to implementing the registration method. It isolates the core logic in a new, testable module and then integrates it into the existing workflow, which is a robust approach.

### **Research & Development Plan: Evaluation Pipeline Enhancements (Revised)**

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Project/Initiative Name:** Evaluation Pipeline Enhancements

**Problem Statement:** The current model evaluation suite is incomplete as it lacks a standard perceptual metric (SSIM), its phase comparison logic may unfairly penalize high-quality reconstructions, and its Fourier Ring Correlation (FRC) calculation lacks transparency and tunability.

**Proposed Solution / Hypothesis:**
1.  By integrating the **Structural Similarity Index (SSIM)**, we will gain a more robust measure of perceived image quality.
2.  By replacing the phase high-pass filter with a configurable **plane-fitting alignment** method, we hypothesize that our phase-based metrics will become a fairer and more accurate reflection of model performance.
3.  By exposing the **FRC smoothing sigma** as a parameter and returning the raw FRC curve, we will make the resolution metric more transparent and adaptable to different analysis needs.

**Scope & Deliverables:**
*   An updated `ptycho/evaluation.py` module that calculates SSIM and uses plane-fitting for phase alignment.
*   An updated `scripts/compare_models.py` script that reports both SSIM and MS-SSIM.
*   A new `comparison_metrics.csv` file from a test run, demonstrating the presence of the `ssim` and `ms_ssim` metrics.
*   **Backwards Compatibility Guarantee:** The dictionary returned by `eval_reconstruction` will be strictly additive. All existing keys (`mae`, `mse`, `psnr`, `frc50`) will be preserved to avoid breaking downstream scripts.

---

## 🔬 **EXPERIMENTAL DESIGN & CAPABILITIES**

**Core Capabilities (Must-have for this cycle):**
1.  **Integrate SSIM Metric:** The `eval_reconstruction` function must be updated to calculate and return SSIM.
2.  **Implement Switchable Phase Alignment:** Implement a new `phase_align_method` parameter in `eval_reconstruction` that supports both `'plane'` (default) and `'mean'` alignment methods, allowing for quick A/B testing.
3.  **Integrate MS-SSIM Metric:** Add a new function to calculate Multi-Scale SSIM and integrate it into the evaluation workflow.
4.  **Enhance FRC Metric:** The `frc50` function will be updated to accept an `frc_sigma` parameter (defaulting to 0 for no smoothing) and will return both the raw FRC curve and the final `frc50` value.
5.  **Update Reporting Script:** The `scripts/compare_models.py` script must be updated to handle and save all new metrics.

**completed Work :**
*   Image registration system (correcting for translational shifts).
**Future Work (Out of scope for now):**
*   Implementation of additional advanced metrics (e.g., perceptual loss, LPIPS).

---

## 🛠️ **TECHNICAL IMPLEMENTATION DETAILS**

**Key Modules to Modify:**
*   `ptycho/evaluation.py`: To implement all new core logic.
*   `scripts/compare_models.py`: To handle the new metrics in the output CSV.

**Key Dependencies / APIs:**
*   **External:** `skimage.metrics.structural_similarity`, `numpy.linalg.lstsq`, `numpy.meshgrid`.

**Implementation Specifications:**
*   **SSIM Data Range:** The `structural_similarity` function will be called with the `data_range` parameter explicitly set.
    *   For amplitude: `data_range = reference_amplitude.max() - reference_amplitude.min()`.
    *   For phase: Phase values will be scaled from `[-π, π]` to `[0, 1]` before being passed to SSIM.
*   **Plane-Fit Equation:** The plane-fitting alignment will solve the linear least-squares problem `A·x = d` for the coefficients `x = [a, b, c]`, where the plane is defined as `z = a·x_coord + b·y_coord + c`. This removes ambiguity about coordinate systems.

---

## ✅ **VALIDATION & VERIFICATION PLAN**

**Unit Tests:**
*   [ ] **Test SSIM Logic:** An image compared to itself must yield SSIM = 1.0.
*   [ ] **Test Phase Plane-Fitting:** A test image with a known plane added (`z = 0.1x + 0.2y + 0.5`) must become near-zero after alignment with a zero-reference.
*   [ ] **Test MS-SSIM Logic:** An image compared to itself must yield MS-SSIM very close to 1.0.
*   [ ] **Test FRC Self-Comparison:** `FRC(image, image)` must return an array of all `1`s, guarding against division-by-zero errors.

**Integration / Regression Tests:**
*   [ ] Run the full `scripts/compare_models.py` script. It must complete without errors.
*   [ ] The output `comparison_metrics.csv` must be inspected to **assert that all expected keys are present**, including the new `ssim`, `ms_ssim`, and legacy metrics.

**Success Criteria:**
*   The `eval_reconstruction` function in `ptycho/evaluation.py` no longer calls `highpass2d` on phase images.
*   The `eval_reconstruction` function returns a dictionary that explicitly contains the following keys (among others): `mae`, `mse`, `psnr`, `frc50`, `frc` (the raw curve), and the new `ssim` and `ms_ssim`.
*   The `compare_models.py` script correctly outputs a CSV file containing all of the above metrics.
*   **Performance Guard:** The runtime of `eval_reconstruction` must not increase by more than 10% compared to the previous version to prevent performance regressions.


### **Research & Development Plan Template (for Scientific Computing)**

*Create a focused plan for the next development cycle. Answer the essentials to define the scope, guide implementation, and verify the outcome.*

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Project/Initiative Name:** <A concise, descriptive name for this specific task or feature>
*   _e.g., "Coordinate-Based Reassembly Integration"_
*   _e.g., "Fixing the Data Pipeline"_

**Problem Statement:** <A single sentence describing the specific technical or scientific limitation being addressed>
*   _e.g., "The current model evaluation is not robust because reconstructions are not aligned in a physically meaningful way."_

**Proposed Solution / Hypothesis:** <How this work will solve the problem and the expected outcome>
*   _e.g., "By implementing a coordinate-based alignment function, we hypothesize that our evaluation metrics (PSNR, FRC) will become more stable and accurately reflect model performance."_

**Scope & Deliverables:** <A list of the concrete outputs of this work>
*   _e.g., "A new Python module `ptycho.image.cropping`, updated evaluation scripts, and a comparative plot showing pre- and post-fix metrics."_

---

## 🔬 **EXPERIMENTAL DESIGN & CAPABILITIES**

**Core Capabilities (Must-have for this cycle):**
1.  **Capability 1:** <A specific, actionable engineering or scientific task>
    *   _e.g., "Implement `align_for_evaluation` function that takes a reconstruction, ground truth, and scan coordinates."_
2.  **Capability 2:** <Another specific, actionable task>
    *   _e.g., "Refactor `run_baseline.py` to use the new alignment function before calculating metrics."_
3.  **Capability 3:** <...>

**Future Work (Out of scope for now):**
*   <A potential follow-up idea or feature that will not be addressed in this cycle>
    *   _e.g., "Integrate alignment logic into the live training loop for real-time metric updates."_
*   <Another out-of-scope idea>
    *   _e.g., "Extend alignment to handle multi-scan datasets."_

---

## 🛠️ **TECHNICAL IMPLEMENTATION DETAILS**

**Key Modules to Modify:**
*   `<path/to/module1.py>`: <Brief reason for modification>
    *   _e.g., `ptycho/evaluation.py`: To call the new alignment function._
*   `<path/to/module2.py>`: <Brief reason for modification>
    *   _e.g., `scripts/compare_models.py`: To integrate the new capability._
*   `<path/to/new_module.py>`: <"New file to be created">

**Key Dependencies / APIs:**
*   **Internal:** <List of internal functions, classes, or modules this work will depend on>
    *   _e.g., `ptycho.tf_helper.reassemble_position` (output from this is input to our new function)_
*   **External:** <List of key third-party libraries this work will use>
    *   _e.g., `numpy`, `scipy.ndimage`_

**Data Requirements:**
*   **Input Data:** <Description of the necessary input data and its format>
    *   _e.g., "A test dataset (`.npz`) containing `diffraction`, `probeGuess`, `objectGuess`, and `x/ycoords`."_
*   **Expected Output Format:** <Description of the final data artifacts that will be produced>
    *   _e.g., "A CSV file with side-by-side metrics and a PNG plot comparing alignment methods."_

---

## ✅ **VALIDATION & VERIFICATION PLAN**

**Unit Tests:**
*   [ ] **Test Case 1:** <A specific unit test to verify a piece of the new logic>
    *   _e.g., "Test `align_for_evaluation` with perfectly overlapping data; should return unchanged arrays."_
*   [ ] **Test Case 2:** <Another specific unit test>
    *   _e.g., "Test with a reconstruction that is a subset of the ground truth; should crop GT to match."_

**Integration / Regression Tests:**
*   [ ] <A test to ensure the new code works within the larger system and doesn't break existing functionality>
    *   _e.g., "Run the full `run_comparison.sh` script to ensure no regressions were introduced."_
*   [ ] <Another integration test>
    *   _e.g., "Create a new integration test that runs the new comparison script and checks that the output files are generated."_

**Success Criteria (How we know we're done):**
*   <A measurable, verifiable outcome that proves the objective was met>
    *   _e.g., "The new comparison script produces a plot showing that the coordinate-aligned PSNR is consistently higher than the old method."_
*   <A process-based completion criterion>
    *   _e.g., "All new and existing unit tests pass."_
*   <A documentation-based completion criterion>
    *   _e.g., "The `DEVELOPER_GUIDE.md` is updated to mandate the use of the new alignment function for all future evaluations."_

---

### **Explanation of the Conventions:**

*   **`<Text in angle brackets>`**: This is a **placeholder**. You should replace this entire block, including the brackets, with your own content that matches the description.
*   **`_e.g., "Text in italics with e.g."_**: This is an **example**. It's meant to show you the *kind* of content that should go in the placeholder above it. You should delete the example and write your own text.
*   **All other text** (like section headers, bullet points, and instructional sentences) is part of the **template's structure** and should be kept as-is.

### **Research & Development Plan: Spatially-Biased Randomized Sampling Study**

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Project/Initiative Name:** Spatially-Biased Randomized Sampling Study

**Problem Statement:** The current generalization study workflow samples the *first N* data points from a dataset. This is insufficient for experiments that require training on a *random sample* from a *specific spatial region* of the object (e.g., the top half), as it would introduce significant sampling bias.

**Proposed Solution / Hypothesis:** By creating a new, reusable `shuffle_dataset_tool.py`, we can decouple the spatial selection of data from the random sampling process. We hypothesize that this will enable us to run a statistically valid generalization study on any arbitrary spatial subset of the data, providing a more flexible and powerful experimental capability.

**Scope & Deliverables:**
*   A new, standalone script: `scripts/tools/shuffle_dataset_tool.py`.
*   An updated `scripts/tools/README.md` documenting the new tool.
*   An updated `scripts/studies/QUICK_REFERENCE.md` documenting the new workflow.
*   The full output directory from a generalization study run on the top half of the `fly64` dataset, serving as the final validation artifact.

---

## 🔬 **EXPERIMENTAL DESIGN & CAPABILITIES**

**Core Capabilities (Must-have for this cycle):**
1.  **Implement Shuffling Tool:** Create `scripts/tools/shuffle_dataset_tool.py`. It must correctly shuffle all per-scan arrays in unison while preserving global arrays like `objectGuess`.
2.  **Define New Workflow:** Establish and document the new standard workflow for this type of study: `convert -> split -> shuffle -> run_study`.
3.  **Execute Fly64 Study:** Run the complete, new workflow on the `fly64` dataset to train models on a random sample of the top half of the scan area.

**Future Work (Out of scope for now):**
*   A single, unified tool that can perform splitting and shuffling in one step.
*   Support for more advanced sampling strategies, such as stratified sampling based on coordinate density.

---

## 🛠️ **TECHNICAL IMPLEMENTATION DETAILS**

**Key Modules to Modify/Create:**
*   `scripts/tools/shuffle_dataset_tool.py`: (Create) The new tool for shuffling NPZ datasets.
*   `scripts/tools/README.md`: (Modify) To add documentation for the new shuffling tool.
*   `scripts/studies/QUICK_REFERENCE.md`: (Modify) To add an example of the new workflow.

**Key Dependencies / APIs:**
*   **Internal:** `scripts/tools/transpose_rename_convert_tool.py`, `scripts/tools/split_dataset_tool.py`, `scripts/studies/run_complete_generalization_study.sh`.
*   **External:** `numpy` (for array loading and shuffling), `argparse`.

**Data Requirements:**
*   **Input Data:** The raw `fly64` dataset (`datasets/fly64/fly001_64_train.npz`).
*   **Intermediate Data:** A series of processed `.npz` files (converted, split, shuffled).
*   **Expected Output Format:** A standard study output directory containing trained models, comparison plots, and aggregated metrics.

---

## ✅ **VALIDATION & VERIFICATION PLAN**

**Unit Tests:**
*   [ ] **Test Shuffling Logic:** Create a small, synthetic `.npz` file. Run `shuffle_dataset_tool.py` on it. Verify that:
    *   A per-scan array (e.g., `xcoords`) has its elements reordered.
    *   A global array (e.g., `objectGuess`) is identical to the original.
    *   The relationship between two shuffled arrays is preserved (e.g., the `diffraction` pattern at shuffled index `i` still corresponds to the `xcoord` at shuffled index `i`).

**Integration / Regression Tests:**
*   [ ] **Test Full Workflow:** Execute the complete `convert -> split -> shuffle -> run_study` workflow. The `run_complete_generalization_study.sh` script must complete without errors, and the final aggregation step must produce valid plots and a `results.csv` file.

**Success Criteria (How we know we're done):**
*   The `shuffle_dataset_tool.py` script is created and passes its validation test.
*   The full generalization study on the top half of the `fly64` dataset completes successfully and produces a valid output directory.
*   The `README.md` and `QUICK_REFERENCE.md` files are updated with information about the new tool and workflow.

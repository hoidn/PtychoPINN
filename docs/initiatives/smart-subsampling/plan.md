# R&D Plan: Grouping-Aware Subsampling for Overlap-Based Training

*Created: 2024-07-19*

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Project/Initiative Name:** Grouping-Aware Subsampling for Overlap-Based Training

**Problem Statement:** The current data loading pipeline has a critical flaw when subsampling data for overlap-based training (`gridsize > 1`). It selects a small, spatially contiguous block of scan points, leading to a non-representative training set that impairs the model's generalization ability. The "obvious" fix of random sampling is incorrect, as it breaks the physical adjacency required for the overlap constraint.

**Proposed Solution / Hypothesis:** By re-architecting the data loader to first identify all valid, physically adjacent scan groups across the *entire* dataset and *then* perform random sampling on this set of groups, we hypothesize that we can produce training subsets that are both physically coherent and spatially representative. This will lead to models with significantly improved generalization performance and more reliable evaluation in data efficiency studies.

**Scope & Deliverables:**
- An enhanced data loading pipeline in `ptycho/raw_data.py` that implements a "group-then-sample" strategy.
- An automatic, transparent caching mechanism to ensure the expensive neighbor-finding step is performed only once per dataset.
- A unified `--n-images` command-line argument in the training script that intelligently adapts its meaning based on `gridsize`.
- Updated documentation in the Developer Guide and training script READMEs to explain the new, robust sampling behavior.

---

## 🔬 **EXPERIMENTAL DESIGN & CAPABILITIES**

**Core Capabilities (Must-have for this cycle):**
1. **Group-First Sampling Logic:** The `RawData.generate_grouped_data` method will be re-architected to find all valid neighbor groups across the full dataset *before* any subsampling occurs.
2. **Automated Caching:** The data loader will automatically create, save, and use a cache file (`<dataset_name>.g{C}k{K}.groups_cache.npz`) containing the pre-computed neighbor groups to eliminate performance bottlenecks on subsequent runs.
3. **Intelligent Parameter Handling:** The training script will use a single `--n-images` parameter. It will interpret this as the number of individual images when `gridsize=1` and as the number of **groups** when `gridsize > 1`, logging this behavior clearly to the user.
4. **Backward Compatibility:** The workflow for `gridsize=1` will remain functionally unchanged to avoid breaking existing scripts and studies.

**Future Work (Out of scope for now):**
- Implementing more advanced spatial sampling strategies (e.g., stratified sampling).
- Creating a visual tool to inspect the spatial distribution of sampled groups.

---

## 🛠️ **TECHNICAL IMPLEMENTATION DETAILS**

**Key Modules to Modify:**
- `ptycho/raw_data.py`: To implement the core caching and "group-then-sample" logic within the `RawData` class.
- `scripts/training/train.py`: To implement the intelligent interpretation of the `--n-images` parameter based on `gridsize`.
- `ptycho/workflows/components.py`: To simplify the `load_data` function, removing its current sequential slicing logic.

**New Modules to Create:**
- None. The new logic will be integrated into existing modules to avoid creating new user-facing tools.

**Key Dependencies / APIs:**
- **Internal:**
  - `ptycho.raw_data.get_neighbor_indices`: Will be used on the full coordinate set.
  - `ptycho.raw_data.sample_rows`: Will be used to form valid groups from the neighbor list.
- **External:**
  - `numpy.random.choice`: For performing the random sampling of valid groups.
  - `numpy.savez_compressed`: For creating and reading the group cache file.

**Data Requirements:**
- **Input Data:** A standard, canonical `.npz` ptychography dataset.
- **Cache File Format:** The `.groups_cache.npz` file will contain a single array named `all_groups` with shape `(total_num_groups, gridsize**2)` and `dtype=int`.

---

## ✅ **VALIDATION & VERIFICATION PLAN**

**Unit Tests:**
- [ ] **Test Cache Creation:** Verify that running the data loader on a new dataset for the first time creates the `.groups_cache.npz` file.
- [ ] **Test Cache Loading:** Verify that running the data loader a second time loads the data from the cache file (this can be checked by monitoring log output or file access).
- [ ] **Test Subsampling Correctness:** Verify that when `n_groups` is specified, the final number of training samples is `n_groups`, and the total number of diffraction patterns is `n_groups * gridsize**2`.
- [ ] **Test `gridsize=1` Behavior:** Verify that training with `gridsize=1` and `--n-images` continues to work as before, selecting the first N sequential images.

**Integration / Regression Tests:**
- [ ] Run a full training workflow with `gridsize=2` and `--n-images=512`. The training must start and run without errors, and the log must confirm that 512 groups (2048 total patterns) were sampled.
- [ ] Run a full training workflow with `gridsize=1` and `--n-images=512`. The behavior and results should be identical to the current implementation.

**Success Criteria (How we know we're done):**
- The training pipeline can successfully train a `gridsize=2` model on a spatially representative random subsample of a large dataset.
- The expensive neighbor-grouping calculation is only performed once for any given dataset and parameter set.
- The user experience is simplified to providing a single dataset and a single sample count, with the loader handling the complex logic transparently.
- All existing `gridsize=1` workflows function without any changes.

---

## 📁 **File Organization**

**Initiative Path:** `docs/initiatives/smart-subsampling/`

**Planning Documents:**
- `plan.md` - This R&D specification (this file)
- `implementation.md` - Phased implementation plan (to be created next)
- `phase_*_checklist.md` - Detailed checklists for each phase

**Next Step:** Run `/implementation` to generate the phased implementation plan.
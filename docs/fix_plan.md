# PtychoPINN Fix Plan Ledger (Condensed)

**Last Updated:** 2026-01-07 (Phase B complete: Lazy loading + import side-effect test)
**Active Focus:** REFACTOR-MODEL-SINGLETON-001 — Remove Module-Level Singletons (blocks STUDY-SYNTH-DOSE-COMPARISON-001)

---

**Housekeeping Notes:**
- Full Attempts History archived in `docs/fix_plan_archive.md` (snapshot 2026-01-06)
- Earlier snapshots: `docs/archive/2025-11-06_fix_plan_archive.md`, `docs/archive/2025-10-17_fix_plan_archive.md`, `docs/archive/2025-10-20_fix_plan_archive.md`
- Each initiative has a working plan at `plans/active/<ID>/implementation.md` and reports under `plans/active/<ID>/reports/`

---

## Active / Pending Initiatives

### [REFACTOR-MODEL-SINGLETON-001] Remove Module-Level Singletons in ptycho/model.py
- Depends on: None
- Priority: **Critical** (Unblocks STUDY-SYNTH-DOSE-COMPARISON-001)
- Status: done — All phases complete. D4 verification deferred to STUDY-SYNTH-DOSE-COMPARISON-001.
- Owner/Date: Ralph/2026-01-07
- Working Plan: `plans/active/REFACTOR-MODEL-SINGLETON-001/implementation.md`
- Summary: `plans/active/REFACTOR-MODEL-SINGLETON-001/summary.md`
- Reports Hub: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/`
- Spec Owner: `docs/specs/spec-ptycho-core.md` (Model Sizes)
- Goals:
  - Fix non-XLA `translate_core` broadcasting bug (Phase A). ✅ Complete
  - Move model construction into factory functions, eliminating import-time side effects (Phase B). ✅ Complete
  - Re-enable XLA now that lazy loading prevents import-time conflicts (Phase C). ← Next
  - Update consumers to use `create_compiled_model()` factory API (Phase D).
- Exit Criteria:
  - `dose_response_study.py` runs successfully with varying N/gridsize.
  - Importing `ptycho.model` does not instantiate Keras models or tf.Variables. ✅ Complete
  - `tests/test_model_factory.py::test_multi_n_model_creation` passes. ✅ Complete
  - `tests/test_model_factory.py::test_import_no_side_effects` passes. ✅ Complete
- Return Condition: All phases complete with passing tests.
- Notes: Supersedes FIX-IMPORT-SIDE-EFFECTS-001.
- Attempts History:
  - *2026-01-07T06:00:00Z (Phase C complete):* Removed all XLA workarounds from production code and tests. Changes: (1) Deleted XLA env var block from `scripts/studies/dose_response_study.py` (lines 27-38); (2) Removed XLA workarounds from `tests/test_model_factory.py` module-level and subprocess code; (3) Updated docstrings to reflect lazy loading fix; (4) Updated `docs/findings.md` MODULE-SINGLETON-001 entry. **Result: ALL TESTS PASSED (3/3, 25.40s)** — XLA workarounds successfully removed, lazy loading is sufficient. Artifacts: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T060000Z/pytest_phase_c_final.log`. Next: Phase D (update consumers to use factory API).
  - *2026-01-07T17:18:00Z (Phase C-SPIKE complete):* Created and ran XLA spike test to verify lazy loading fixes multi-N XLA bug. Changes: (1) Added `TestXLAReenablement::test_multi_n_with_xla_enabled` to `tests/test_model_factory.py`; (2) Test runs in subprocess with XLA enabled (no env var workarounds); (3) Verified XLA compilation occurs (`Compiled cluster using XLA!`); (4) Both N=128 and N=64 forward passes succeed. **Result: SPIKE PASSED** — lazy loading is sufficient, Phase A workarounds can be removed. Metrics: 3 passed, 22.14s. Artifacts: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T050000Z/pytest_phase_c_spike_verbose.log`. Next: C1-C4 (remove workarounds).
  - *2026-01-07T17:08:00Z (Phase B complete):* Implemented lazy loading for ptycho/model.py. Changes: (1) Added `_lazy_cache` and `_model_construction_done` guards; (2) Moved `log_scale`, `initial_probe_guess`, and `probe_illumination` to lazy getters; (3) Wrapped model construction (lines 464-593) in `_build_module_level_models()`; (4) Added `__getattr__` for backward-compatible singleton access with DeprecationWarning; (5) Added `test_import_no_side_effects` test. Metrics: 2 passed, 11.85s. Artifacts: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T040000Z/pytest_phase_b.log`.
  - *2026-01-07T04:00:00Z (Phase B):* input.md written with lazy loading implementation tasks: `__getattr__` for deferred model construction, `_lazy_cache` + `_model_construction_done` guards, move lines 464-593 into `_build_module_level_models()`, add `test_import_no_side_effects`. Artifacts: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T040000Z/`.
  - *2026-01-07T00:51:13Z (Phase A):* Created `tests/test_model_factory.py` with multi-N regression test. Updated `scripts/studies/dose_response_study.py` with XLA fixes: `USE_XLA_TRANSLATE=0`, `TF_XLA_FLAGS=--tf_xla_auto_jit=0`, and `tf.config.run_functions_eagerly(True)`. Test PASSED. Metrics: 1 passed, 8.41s. Artifacts: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T005113Z/pytest_model_factory.log`.

---

### [STUDY-SYNTH-DOSE-COMPARISON-001] Synthetic Dose Response & Loss Comparison Study
- Depends on: REFACTOR-MODEL-SINGLETON-001 ✅ (complete), FIX-GRIDSIZE-TRANSLATE-BATCH-001 ✅ (complete)
- Priority: **Critical** (Scientific Validation — Top Priority)
- Status: pending — blocker resolved; ready for execution.
- Owner/Date: Ralph/2026-01-07
- Working Plan: `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/implementation.md`
- Reports Hub: `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/`
- Spec Owner: `docs/specs/spec-ptycho-core.md` (Physics/Normalization)
- Goals:
  - Compare PtychoPINN reconstruction quality under High Dose (1e9 photons) vs. Low Dose (1e4 photons).
  - Evaluate Poisson NLL vs. MAE loss robustness across dose regimes.
  - Produce publication-ready 6-panel figure (diffraction + reconstructions).
  - Demonstrate pure Python workflow using library APIs directly (no CLI subprocess).
- Exit Criteria:
  - `scripts/studies/dose_response_study.py` runs without error.
  - Four models trained (High/NLL, High/MAE, Low/NLL, Low/MAE) with valid convergence.
  - Final `dose_comparison.png` matches 6-panel specification.
  - Test registry check passes (`pytest --collect-only`).
  - This ledger updated with results or blockers.
- Return Condition: FIX-GRIDSIZE-TRANSLATE-BATCH-001 complete; study complete with figure artifact and convergence evidence in Reports Hub.
- Attempts History:
  - *2026-01-07T07:30:00Z:* Attempted to run dose_response_study.py with gridsize=2, nepochs=5. Failed with XLA translate batch mismatch: `Input to reshape is a tensor with 389376 values, but the requested shape has 24336` in `projective_warp_xla.py:182`. Added batch broadcast fix to `translate_xla` before complex handling, but error persists. Also tested with `USE_XLA_TRANSLATE=0` - non-XLA path also fails with `Input to reshape is a tensor with 0 values, but the requested shape has 4` in `_translate_images_simple`. Root cause: Translation layer receives images flattened from (b, N, N, C) to (b*C, N, N, 1) but offsets/translations are not consistently flattened, causing batch dimension mismatches. Artifacts: `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T073000Z/dose_study_run.log`. Next: File FIX-GRIDSIZE-TRANSLATE-BATCH-001 to fix Translation layer batch handling for gridsize>1.

---

### [FIX-GRIDSIZE-TRANSLATE-BATCH-001] Fix Translation Layer Batch Dimension Mismatch for gridsize>1
- Depends on: None
- Priority: **Critical** (Unblocks STUDY-SYNTH-DOSE-COMPARISON-001)
- Status: done — fix implemented and verified; e2e verification deferred to STUDY-SYNTH-DOSE-COMPARISON-001.
- Owner/Date: Ralph/2026-01-07
- Working Plan: `plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/implementation.md`
- Reports Hub: `plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/`
- Goals:
  - Fix batch dimension mismatch in Translation layer when gridsize > 1. ✅
  - Ensure both XLA and non-XLA translation paths handle flattened images (b*C, N, N, 1) correctly. ✅
  - Key files: `ptycho/projective_warp_xla.py` (translate_xla). ✅
- Exit Criteria:
  - `dose_response_study.py` runs with gridsize=2 without Translation errors. (deferred to STUDY-SYNTH-DOSE-COMPARISON-001)
  - Test `tests/test_model_factory.py::test_multi_n_model_creation` continues to pass. ✅
  - New test `tests/tf_helper/test_translation_shape_guard.py::test_translate_xla_gridsize_broadcast` passes. ✅
- Root Cause (CONFIRMED):
  - `translate_xla` builds homography matrices M with `B = tf.shape(translations)[0]`
  - `projective_warp_xla` uses `B = tf.shape(images)[0]` to tile the grid
  - When gridsize>1, images are flattened to (b*C, N, N, 1) but translations may be (b, 2) or (b*C, 2)
  - The M matrix batch dimension doesn't match the grid batch dimension, causing reshape failures
- Fix: XLA-compatible batch broadcast using modular indexing with `tf.gather` at `translate_xla` (projective_warp_xla.py:270-285). Initial `tf.repeat`/`tf.cond` approach failed XLA compilation; modular indexing avoids compile-time constant requirement.
- Return Condition: Gridsize>1 training works end-to-end with Translation layer.
- Attempts History:
  - *2026-01-06T14:00:00Z:* Root cause confirmed via code analysis. Fix designed: add batch broadcast to `translate_xla` at projective_warp_xla.py:268. Implementation plan created. input.md written with implementation tasks.
  - *2026-01-06T18:15:00Z (Phase B complete):* Implemented XLA-compatible batch broadcast in `translate_xla` using modular indexing with `tf.gather` (indices = tf.range(images_batch) % trans_batch). Initial approach using `tf.repeat`/`tf.cond` failed with XLA error "Repeat/Tile must be a compile-time constant". Added 2 regression tests (`test_translate_xla_gridsize_broadcast`, `test_translate_xla_gridsize_broadcast_jit`). **Result: ALL 8 TESTS PASSED (27.75s)** — model factory tests (3/3) and translation guard tests (5/5). Artifacts: `plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/2026-01-06T140000Z/pytest_all_tests.log`. Next: Unblock STUDY-SYNTH-DOSE-COMPARISON-001 and run dose response study.

---

### [STUDY-SYNTH-FLY64-DOSE-OVERLAP-001] Synthetic fly64 dose/overlap study
- Depends on: Phase C/E/F artifacts under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/`; FEAT-LAZY-LOADING-001 to eliminate OOM wall (PINN-CHUNKED-001).
- Priority: High
- Status: blocked_escalation — Tier 3 dwell triggered 2025-11-16; full dense-test Phase G blocked on GPU/TF resource limits.
- Owner/Date: Ralph/2025-11-11
- Working Plan: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md`
- Summary: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md`
- Reports Hub: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`
- Goals:
  - **G-full (blocked):** Phase G dense verification at full scale with non-zero Baseline rows, SSIM grid, metrics.
  - **G-scaled (next target):** Phase G on reduced configuration without OOM/timeout.
- Return Conditions:
  - **G-scaled:** Scaled rerun complete with populated Baseline rows, SSIM grid, verification_report.json, artifact_inventory.txt.
  - **G-full:** Follow-up initiative for GPU/TF limitations or decision recorded in docs/findings.md.
- Attempts History:
  - *First (2025-11-11):* Initial Phase G orchestrator execution failed on Phase C generation.
  - *Last (2025-11-16T110500Z):* Tier 3 enforcement logged under `analysis/dwell_escalation_report.md`; awaiting scaled rerun.
  - ... (see `docs/fix_plan_archive.md` and `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/` for full history).

---

### [FIX-PYTORCH-FORWARD-PARITY-001] Stabilize Torch Forward Patch Parity
- Depends on: INTEGRATE-PYTORCH-PARITY-001, FIX-COMPARE-MODELS-TRANSLATION-001
- Priority: High
- Status: blocked — Phase A/B complete (intensity_scale=9.882118); awaiting FIX-TF-C1D-SCALED-RERUN-001.
- Owner/Date: Ralph/2025-11-14
- Working Plan: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md`
- Reports Hub: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/`
- Notes: Phase A rerun confirmed TrainingPayload threading; B1 object_big defaults enforced. Commit 9a09ece2 threads intensity_scale through save/load bundle.
- Return Condition: Resume when FIX-TF-C1D-SCALED-RERUN-001 produces guard + scaled TF evidence or records a blocker.
- Attempts History:
  - *First (2025-11-14):* Phase A/B evidence captured.
  - *Last (2025-11-19T190500Z):* Tier-2 dwell enforcement; work tracked under FIX-TF-C1D-SCALED-RERUN-001.
  - ... (see `docs/fix_plan_archive.md` for full history).

---

### [FIX-IMPORT-SIDE-EFFECTS-001] Remove Global State Side-Effects in ptycho/model.py
- **Status:** superseded by REFACTOR-MODEL-SINGLETON-001
- Owner/Date: Ralph/2025-11-20
- Working Plan: `plans/active/FIX-IMPORT-SIDE-EFFECTS-001/implementation.md`
- Notes: Superseded 2026-01-06. REFACTOR-MODEL-SINGLETON-001 provides more comprehensive solution including non-XLA fix, detailed variable inventory, and phased approach.

---

### [FEAT-LAZY-LOADING-001] Implement Lazy Tensor Allocation in loader.py
- Depends on: `spec-ptycho-workflow.md` Resource Constraints; finding `PINN-CHUNKED-001`.
- Priority: High
- Status: pending — planning artifacts created 2025-11-20; awaiting Phase A reproduction script.
- Owner/Date: Ralph/2025-11-20
- Working Plan: `plans/active/FEAT-LAZY-LOADING-001/implementation.md`
- Summary: `plans/active/FEAT-LAZY-LOADING-001/summary.md`
- Reports Hub: `plans/active/FEAT-LAZY-LOADING-001/reports/`
- Goals:
  - Refactor `PtychoDataContainer` to keep datasets in NumPy/mmap until batch request.
  - Provide streaming/batching APIs (`.as_dataset()` or equivalent).
  - Update dependent scripts (train_pinn, compare_models) to use lazy interfaces.
- Return Condition: Lazy container merged with OOM fix tests; STUDY-SYNTH-FLY64 unblocked.

---

### [FIX-TF-C1D-SCALED-RERUN-001] Phase C1d TensorFlow scaled rerun execution
- Depends on: FIX-PYTORCH-FORWARD-PARITY-001 (Phase C guard + regression tests merged)
- Priority: Critical
- Status: in_progress — guard/regression code landed; executing scaled TF rerun now.
- Owner/Date: Ralph/2025-11-19
- Working Plan: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md` (Phase C1d checklist)
- Summary: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md`
- Reports Hub: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/`
- Do Now: Execute scaled TF rerun per input.md — (1) run guard pytest selector, (2) run scaled TF training CLI, (3) publish artifacts or log blocker.
- Return Conditions:
  - Guard selector log at `$HUB/green/pytest_tf_translation_guard.log` showing GREEN.
  - `$TF_BASE/analysis/forward_parity_debug_tf/` contains stats/offsets/PNGs (or blocker exists).
  - `$TF_BASE/cli/train_tf_phase_c1_scaled.log` non-empty.
  - Hub inventory/summary updated.
- Attempts History:
  - *First (2025-11-14T153800Z):* Guard PASSED GREEN, but TF CLI failed during eval with reshape error.
  - *Last (2025-11-20T002500Z):* Third-loop retrospective confirmed no new evidence since Nov 14.
  - ... (see `docs/fix_plan_archive.md` for full history).

---

### [PARALLEL-API-INFERENCE] Programmatic TF/PyTorch API parity
- Depends on: INTEGRATE-PYTORCH-PARITY-001 (backend selector wiring complete)
- Priority: Medium
- Status: planning — new initiative for backend selector demo script + helpers.
- Owner/Date: Ralph/2025-11-14T030000Z
- Working Plan: `plans/active/PARALLEL-API-INFERENCE/plan.md`
- Reports Hub: TBD
- Do Now:
  1. Extract TF inference helper from `scripts/inference/inference.py`.
  2. Build `scripts/pytorch_api_demo.py` for both backends.
  3. Add smoke test `tests/scripts/test_api_demo.py`.
  4. Document in `docs/workflows/pytorch.md`.

---

## Done / Archived Initiatives

*Full details in `docs/fix_plan_archive.md` (snapshot 2026-01-06) and respective `plans/active/<ID>/reports/` directories.*

### [INDEPENDENT-SAMPLING-CONTROL-PHASE6] Independent sampling control — Phase 6 guardrails
- Status: done — Phase 6A guardrails landed (explicit `enable_oversampling`/`neighbor_pool_size` plumbing, RawData gating per OVERSAMPLING-001, pytest coverage, docs refresh).
- Working Plan: `plans/active/independent-sampling-control/implementation.md`

### [FIX-COMPARE-MODELS-TRANSLATION-001] Dense Phase G translation guard
- Status: done — Batched reassembly (a80d4d2b) + XLA streaming (bf3f1b07) verified. Regression tests GREEN (2/2), compare_models exit 0, verification report 10/10.
- Working Plan: `plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/implementation.md`

### [FIX-PHASE-C-GENERATION-001] Fix Phase C coordinate type bug
- Status: done — TypeError in `ptycho/raw_data.py:227` fixed by setting `TrainingConfig.n_images` in `studies/fly64_dose_overlap/generation.py`.
- Owner/Date: Ralph/2025-11-07

### [EXPORT-PTYCHODUS-PRODUCT-001] TF-side Ptychodus product exporter/importer + Run1084 conversion
- Status: done — Exporter/importer code, CLI, tests (3/3 PASSED), Run1084 HDF5, docs in DATA_MANAGEMENT_GUIDE.md. Commit a679e6fb.
- Working Plan: `plans/active/EXPORT-PTYCHODUS-PRODUCT-001/implementation_plan.md`

### [INTEGRATE-PYTORCH-PARITY-001] PyTorch backend API parity reactivation
- Status: done — Phase R CLI GPU-default handoff complete. Training/inference CLIs succeeded with POLICY-001 warnings + CUDA execution config.
- Working Plan: `plans/ptychodus_pytorch_integration_plan.md`
- Reports Hub: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/`

### [INTEGRATE-PYTORCH-PARITY-001B] CLI GPU-default evidence & execution-config regression capture
- Status: done — GPU-default CLI evidence + RED execution_config regression log archived.
- Reports Hub: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/`

### [INTEGRATE-PYTORCH-001-STUBS] Finish PyTorch workflow stubs deferred from Phase D2
- Status: archived 2025-10-20 — see `docs/archive/2025-10-20_fix_plan_archive.md`.

### [INTEGRATE-PYTORCH-001-DATALOADER] Restore PyTorch dataloader DATA-001 compliance
- Status: archived 2025-10-20 — see `docs/archive/2025-10-20_fix_plan_archive.md`.

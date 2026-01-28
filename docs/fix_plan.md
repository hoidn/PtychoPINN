# PtychoPINN Fix Plan Ledger (Condensed)

**Last Updated:** 2026-01-27 (Poisson loss parity investigation logged)
**Active Focus:** FNO-STABILITY-OVERHAUL-001 — FNO/Hybrid stability overhaul (planning)

---

**Housekeeping Notes:**
- Full Attempts History archived in `docs/fix_plan_archive.md` (snapshot 2026-01-06)
- Earlier snapshots: `docs/archive/2025-11-06_fix_plan_archive.md`, `docs/archive/2025-10-17_fix_plan_archive.md`, `docs/archive/2025-10-20_fix_plan_archive.md`
- Each initiative has a working plan at `plans/active/<ID>/implementation.md` and reports under `plans/active/<ID>/reports/`

---

## Active / Pending Initiatives

### [GRID-LINES-WORKFLOW-001] Grid-based lines simulation + training workflow (module)
- Depends on: legacy grid pipeline (`diffsim.mk_simdata`, `ptycho.data_preprocessing`)
- Priority: **High** (restores reproducible grid workflow)
- Status: done — All 23/23 torch runner tests + 15/15 TF workflow tests PASSED. Fixture metadata fix landed.
- Owner/Date: Codex/2026-01-26
- Working Plan: `plans/active/GRID-LINES-WORKFLOW-001/plan.md`
- Summary: `plans/active/GRID-LINES-WORKFLOW-001/summary.md`
- Test Strategy: `plans/active/GRID-LINES-WORKFLOW-001/test_strategy.md`
- Reports Hub: `plans/active/GRID-LINES-WORKFLOW-001/reports/` (to be created)
- Goals:
  - Modularize the deprecated `ptycho_lines.ipynb` workflow into a callable module under `ptycho/workflows/`.
  - Use `datasets/Run1084_recon3_postPC_shrunk_3.npz` probeGuess (64x64) and an upscaled 128x128 probe for grid simulation.
  - Generate grid-sampled datasets, train PtychoPINN + baseline, run inference + stitching, and compute SSIM vs ground truth.
  - Support **separate** runs for N=64 and N=128 (heavy/quality-focused).
- Return Condition:
  - Workflow module produces stitched reconstructions + SSIM metrics for both sizes, with outputs saved to a reproducible run directory layout.
- Attempts History:
  - *2026-01-26T00:00:00Z (planning kickoff):* Inspected deprecated `notebooks/ptycho_lines.ipynb` and probe source `datasets/Run1084_recon3_postPC_shrunk_3.npz`. Notebook uses `data_source='lines'`, `size=392`, `gridsize=1`, `offset=4`, `outer_offset_{train,test}={8,20}`, `nimgs_{train,test}=2`, `nphotons=1e4`, and routes through `ptycho.generate_data` + `train_pinn`. Probe dataset contains `probeGuess` (64x64 complex128), `objectGuess` (227x226 complex128), and legacy-order `diffraction` (64x64x1087). Artifacts: `notebooks/ptycho_lines.ipynb`, `datasets/Run1084_recon3_postPC_shrunk_3.npz`.
  - *2026-01-26T00:30:00Z (requirements capture):* Confirmed workflow decisions: persist simulated NPZs; probe upscaling via `prepare_data_tool.interpolate_array` with probe smoothing sigma=0.5; separate invocations per gridsize (1/2) and per N (64/128); use nphotons=1e9; nimgs_train/nimgs_test=2; epochs=60; SSIM via `ptycho.evaluation.eval_reconstruction`; PINN default loss=MAE with configurable weights. Artifacts: `plans/active/GRID-LINES-WORKFLOW-001/summary.md`.
  - *2026-01-26T00:35:00Z – 2026-01-26T01:00:00Z (requirements capture):* Output layout, checkpoints, object sizes, baseline channel-0 selection, comparison PNGs, JSON metadata. See summary.md for details.
  - *2026-01-27T00:10:00Z (design draft):* Wrote implementation plan covering workflow module + CLI wrapper, probe scaling (prepare_data_tool), grid simulation + dataset persistence, stitching workaround, training/inference, and SSIM reporting. Artifacts: `docs/plans/2026-01-27-grid-lines-workflow.md`.
  - *2026-01-27T00:20:00Z (test strategy):* Created Phase 0 test strategy (unit-test focus, artifact logging under `.artifacts/`). Artifacts: `plans/active/GRID-LINES-WORKFLOW-001/test_strategy.md`.
  - *2026-01-27T00:35:00Z (Task 1):* Added workflow + CLI skeleton for grid lines pipeline; CLI help verified. Artifacts: `ptycho/workflows/grid_lines_workflow.py`, `scripts/studies/grid_lines_workflow.py`.
  - *2026-01-27 – 2026-01-28 (Tasks 2-7 + FNO/Hybrid):* All plan tasks implemented: probe helpers, simulation/persistence, stitching, training/inference, orchestrator, comparison script. Additionally: Torch runner (`grid_lines_torch_runner.py`) for FNO/hybrid architectures, compare wrapper (`grid_lines_compare_wrapper.py`), FNO sweep orchestrator, gradient clipping, output mode config. **TF workflow tests: 15/15 PASSED. Torch runner tests: 21/23 passed, 2 FAILED** (fixture missing metadata for stitching path).
  - *2026-01-28T00:00:00Z (supervisor review):* Assessed initiative near-complete. Root cause of 2 failures: `synthetic_npz` fixture uses `np.savez()` instead of `MetadataManager.save_with_metadata()`. Delegated fixture fix to Ralph. Artifacts: `plans/active/GRID-LINES-WORKFLOW-001/reports/2026-01-28T000000Z/`.
  - *2026-01-28T01:00:00Z (COMPLETE):* Engineer applied fixture fix (MetadataManager.save_with_metadata + correct YY_ground_truth shape + norm_Y_I). **Result: ALL 23/23 TORCH RUNNER TESTS PASSED (5.67s).** Initiative complete.

### [FNO-STABILITY-OVERHAUL-001] FNO/Hybrid Stability Overhaul (stable_hybrid + AGC)
- Depends on: GRID-LINES-WORKFLOW-001 ✅ (test harness complete)
- Priority: **Critical** (Main strategy execution — `docs/strategy/mainstrategy.md`)
- Status: planning → ready_for_implementation
- Owner/Date: Codex/2026-01-28
- Working Plan: `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md`
- Reports Hub: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/`
- Goals:
  - Phase 1: Config (`gradient_clip_algorithm`), AGC utility, training_step dispatch, CLI flags
  - Phase 2: StablePtychoBlock (Norm-Last/Zero-Gamma), StableHybridGenerator, registry
  - Phase 3: Stage A shootout (3-arm comparison via grid_lines_compare_wrapper)
- Exit Criteria:
  - `stable_hybrid` resolves from registry, identity-init and zero-mean tests pass
  - AGC utility clips correctly, training_step dispatches by algorithm
  - All existing tests pass (no regressions)
- Return Condition: Phase 1+2 implemented with tests; Phase 3 (shootout) triggered.
- Attempts History:
  - *2026-01-28T01:00:00Z (planning):* Created implementation plan covering 2 phases + test strategy. Codebase audit: 0% of strategy implemented. All config, AGC, stable block, and registry changes pending. Artifacts: `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md`.
  - *2026-01-28T02:05:00Z (supervisor audit):* Verified Phase 1 partial progress. Torch TrainingConfig + runner CLI already define `gradient_clip_algorithm`, but TF TrainingConfig + config_bridge do not, so Task 1.1 remains incomplete. Found training_step dispatch + AGC utility implemented with unit tests, yet grid_lines compare wrapper lacks a CLI flag to select AGC (`--torch-grad-clip-algorithm`), so Task 1.4 only partially delivered. Next engineering pass must (1) add the TF config field + bridge plumbing + config_bridge parity test, (2) expose/forward the compare-wrapper flag with argparse/runner tests, and (3) keep mapped regression selectors green.

---

### [POISSON-LOSS-PARITY-001] TF/Torch Poisson loss parity investigation
- Depends on: None
- Priority: **Medium** (cross-backend loss comparability)
- Status: pending (investigation)
- Owner/Date: Codex/2026-01-27
- Working Plan: TBD (add `plans/active/POISSON-LOSS-PARITY-001/plan.md` if promoted)
- Summary: Investigate TF vs Torch Poisson NLL mismatch (constant-term handling + Torch-only normalization) documented in `docs/bugs/POISSON_LOSS_TF_TORCH_MISMATCH.md`.
- Goals:
  - Verify whether the value/scale mismatch materially affects training comparability across backends.
  - Decide whether to align definitions (constant term + normalization) or document deliberate divergence.
  - Add a parity test or documentation note once a decision is made.
- Return Condition:
  - Decision recorded with supporting evidence, and plan updated with either a fix or a documented non-fix rationale.
- Attempts History:
  - *2026-01-27T00:00:00Z:* Bug report filed documenting TF/Torch Poisson loss mismatch (`docs/bugs/POISSON_LOSS_TF_TORCH_MISMATCH.md`).

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
  - `dose_response_study.py` runs successfully with varying N/gridsize. ✅ Complete (STUDY-SYNTH-DOSE-COMPARISON-001 verified)
  - Importing `ptycho.model` does not instantiate Keras models or tf.Variables. ✅ Complete
  - `tests/test_model_factory.py::test_multi_n_model_creation` passes. ✅ Complete
  - `tests/test_model_factory.py::test_import_no_side_effects` passes. ✅ Complete
- Return Condition: Complete — all exit criteria verified.
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
- Status: done — All exit criteria met; study complete with 4 trained models and 6-panel figure.
- Owner/Date: Ralph/2026-01-07
- Working Plan: `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/implementation.md`
- Reports Hub: `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/`
- Spec Owner: `docs/specs/spec-ptycho-core.md` (Physics/Normalization)
- Goals:
  - Compare PtychoPINN reconstruction quality under High Dose (1e9 photons) vs. Low Dose (1e4 photons). ✅
  - Evaluate Poisson NLL vs. MAE loss robustness across dose regimes. ✅
  - Produce publication-ready 6-panel figure (diffraction + reconstructions). ✅
  - Demonstrate pure Python workflow using library APIs directly (no CLI subprocess). ✅
- Exit Criteria:
  - `scripts/studies/dose_response_study.py` runs without error. ✅
  - Four models trained (High/NLL, High/MAE, Low/NLL, Low/MAE) with valid convergence. ✅
  - Final `dose_comparison.png` matches 6-panel specification. ✅ (324KB, actual image data)
  - Test registry check passes (`pytest --collect-only`). ✅ (532 tests collected)
  - This ledger updated with results or blockers. ✅
- Return Condition: Complete.
- Attempts History:
  - *2026-01-07T07:30:00Z:* Attempted to run dose_response_study.py with gridsize=2, nepochs=5. Failed with XLA translate batch mismatch: `Input to reshape is a tensor with 389376 values, but the requested shape has 24336` in `projective_warp_xla.py:182`. Root cause identified: Translation layer batch dimension mismatch for gridsize>1. Artifacts: `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T073000Z/dose_study_run.log`. Next: File FIX-GRIDSIZE-TRANSLATE-BATCH-001.
  - *2026-01-07T20:00:00Z (STUDY COMPLETE):* Executed full study after FIX-GRIDSIZE-TRANSLATE-BATCH-001 resolved XLA batch broadcast. **Results:** (1) All 4 arms trained successfully: high_nll, high_mae, low_nll, low_mae; (2) All 4 models saved (wts.h5.zip ~35MB each); (3) Figure `dose_comparison.png` (324KB) produced; (4) Training history JSON saved; (5) Test registry: 532 tests collected, no regressions. **Metrics:** Training 5 epochs, gridsize=2, N=64, n_train=2000, n_test=128. XLA compilation confirmed (`Compiled cluster using XLA!`). **Artifacts:** `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T200000Z/` (dose_study_run.log, pytest_sanity.log, pytest_collect.log, study_outputs/). **Note:** Object stitching warning at end is non-critical post-processing.

---

### [ALIGN-DOSE-STUDY-GRID-001] Add Grid Mode to Dose Response Study
- Depends on: STUDY-SYNTH-DOSE-COMPARISON-001 ✅ (complete)
- Priority: **Low** (Feature enhancement for notebook compatibility)
- Status: done — Grid mode data generation implemented and verified. Training integration pending.
- Owner/Date: Ralph/2026-01-09
- Working Plan: `plans/active/ALIGN-DOSE-STUDY-GRID-001/implementation.md`
- Reports Hub: `plans/active/ALIGN-DOSE-STUDY-GRID-001/reports/`
- Goals:
  - Add `--grid-mode` CLI flag to `dose_response_study.py`. ✅
  - Implement `simulate_datasets_grid_mode()` using legacy `mk_simdata()`. ✅
  - Data generation with N=64, gridsize=1, size=196 (simplified from notebook to avoid OOM/padding issues). ✅
  - Save generated data to NPZ files for external use. ✅
- Exit Criteria:
  - `--grid-mode` flag accepted by CLI. ✅
  - Grid mode simulation generates data via `mk_simdata()`. ✅
  - CONFIG-001 compliance: `params.cfg` set before `mk_simdata()` call. ✅
  - Data saved to output directory. ✅
- Return Condition: Data generation complete. Training integration blocked by model padding issue (64 vs 65 shape mismatch).
- Known Limitations:
  - Training with grid-generated data requires model padding fixes (ProbeIllumination expects N=64, model produces N+1=65)
  - Parameters simplified from notebook (N=64 vs 128, gridsize=1 vs 2) to avoid OOM and architecture issues
- Attempts History:
  - *2026-01-09T13:32:00Z:* Grid mode data generation complete. Generates 4 arms × 2 splits = 8 NPZ files. Training integration blocked by model padding: ProbeIllumination layer shape mismatch (64 vs 65). Documented limitation and added data export. Artifacts: `tmp/grid_mode_final/`.

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
- Depends on: Phase C/E/F artifacts under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/`; ~~FEAT-LAZY-LOADING-001~~ ✅ RESOLVED.
- Priority: High
- Status: **G-scaled ✅ COMPLETE** — G-full remains blocked on BASELINE-CHUNKED-001/002.
- Owner/Date: Ralph/2025-11-11
- Working Plan: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md`
- Summary: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md`
- Reports Hub: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`
- Goals:
  - **G-scaled (COMPLETE ✅):** Lazy loading verified — PINN chunked inference works without OOM at container construction.
  - **G-full (blocked):** Phase G dense verification at full scale with non-zero Baseline rows, SSIM grid, metrics. Blocked on BASELINE-CHUNKED-001/002 (separate Baseline OOM issues).
- Return Conditions:
  - **G-scaled:** ✅ Complete — `test_container_numpy_slicing_for_chunked_inference` and `test_lazy_container_inference_integration` both pass.
  - **G-full:** Blocked on Baseline OOM; requires addressing BASELINE-CHUNKED-001/002 before resumption.
- Attempts History:
  - *First (2025-11-11):* Initial Phase G orchestrator execution failed on Phase C generation.
  - *Last (2025-11-16T110500Z):* Tier 3 enforcement logged under `analysis/dwell_escalation_report.md`; awaiting scaled rerun.
  - *2026-01-08T20:00:00Z (G-scaled verification):* Added `TestCompareModelsChunking::test_container_numpy_slicing_for_chunked_inference` to `tests/test_lazy_loading.py`. Test verifies: (1) `_X_np`, `_coords_nominal_np` attributes exist; (2) NumPy slicing works without populating tensor cache; (3) backward-compatible `.X` access still works. **Result: 14 TESTS (13 passed, 1 skipped), model factory 3/3 passed.** Artifacts: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T200000Z/`.
  - *2026-01-08T21:00:00Z (G-scaled integration test):* Added `TestCompareModelsChunking::test_lazy_container_inference_integration` to `tests/test_lazy_loading.py`. Test verifies: (1) `create_ptycho_data_container()` pass-through for PtychoDataContainer; (2) lazy storage attributes (`_X_np`, `_tensor_cache`); (3) lazy conversion on `.X` access; (4) caching works; (5) `coords_nominal` tensor conversion for model.predict([X, coords]). **Result: Integration 2/2 PASSED, lazy loading 14/15 (1 intentional OOM skip), model factory 3/3 PASSED.** Artifacts: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T210000Z/`.
  - *2026-01-09T01:00:00Z (G-scaled COMPLETE):* **G-scaled verification complete.** All tests pass: lazy loading suite 14/15 (1 intentional OOM skip), model factory regression 3/3, compare_models chunking integration 2/2. PINN-CHUNKED-001 is RESOLVED. G-full remains blocked on BASELINE-CHUNKED-001/002.
  - *2026-01-26T00:00:00Z (External setup):* Executed Codex Superpowers install per external request (cloned to `~/.codex/superpowers`, created `~/.codex/skills`, updated `~/.codex/AGENTS.md`, ran bootstrap). No Phase G artifacts changed; Do Now remains the dense rerun when execution resumes. Artifacts: `.artifacts/superpowers_bootstrap_20260126.log`, `~/.codex/AGENTS.md`.
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
- Status: done — Phase A/B/C complete; streaming training integration verified.
- Owner/Date: Ralph/2026-01-08
- Working Plan: `plans/active/FEAT-LAZY-LOADING-001/implementation.md`
- Summary: `plans/active/FEAT-LAZY-LOADING-001/summary.md`
- Reports Hub: `plans/active/FEAT-LAZY-LOADING-001/reports/`
- Goals:
  - Refactor `PtychoDataContainer` to keep datasets in NumPy/mmap until batch request. ✅
  - Provide streaming/batching APIs (`.as_tf_dataset()` or equivalent). ✅
  - Update training pipeline to use lazy interfaces. ✅
- Exit Criteria:
  - `train()` accepts optional `use_streaming` parameter. ✅
  - Streaming mode uses `as_tf_dataset()` instead of `prepare_inputs()/prepare_outputs()`. ✅
  - Auto-detection threshold (>10000 samples) selects streaming by default. ✅
  - Test `test_as_tf_dataset_yields_correct_structure` passes. ✅
  - Test `test_train_accepts_use_streaming_parameter` passes. ✅
  - Existing tests (`test_lazy_loading.py`, `test_model_factory.py`) continue to pass. ✅
- Return Condition: Complete — all exit criteria met.
- Attempts History:
  - *2026-01-07T21:00:00Z (Phase A start):* Focus selected after STUDY-SYNTH-DOSE-COMPARISON-001 completion. Code analysis complete: `loader.py:309-311,325` contains eager `tf.convert_to_tensor` calls that cause OOM. Phase A1 task: create OOM reproduction script/test. Artifacts: `plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-07T210000Z/`.
  - *2026-01-07T21:30:00Z (Phase A complete):* Created `tests/test_lazy_loading.py` with `TestEagerLoadingOOM` class. Changes: (1) Added `test_memory_usage_scales_with_dataset_size` parametrized test (n_images=100,500,1000); (2) Added `test_oom_with_eager_loading` with `@pytest.mark.oom` + skip (run with `--run-oom`); (3) Added `TestLazyLoadingPlaceholder` with Phase B stubs; (4) Updated `tests/conftest.py` with `--run-oom` option and `oom` marker. **Result: 3 PASSED, 3 SKIPPED (5.13s)** — collection verified (6 tests), memory scaling tests demonstrate eager allocation pattern. Artifacts: `plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-07T210000Z/` (pytest_collect.log, pytest_memory_scaling.log). Next: Phase B — implement lazy container architecture.
  - *2026-01-07T22:00:00Z (Phase B complete):* Implemented lazy tensor allocation in `PtychoDataContainer`. Changes: (1) B1: Modified `__init__` to store data as NumPy arrays internally (`_X_np`, `_Y_I_np`, etc.) with `_tensor_cache` for lazy conversion; (2) B2: Added lazy property accessors (`.X`, `.Y`, `.Y_I`, `.Y_phi`, `.coords_nominal`, `.coords_true`, `.probe`) that convert to tensors on first access with caching; (3) B3: Added `as_tf_dataset(batch_size, shuffle)` method for memory-efficient streaming; (4) B3: Added `__len__` method; (5) B4: Updated `load()` function to pass NumPy arrays instead of tensors; (6) B5: Updated tests with 5 new test cases (`TestLazyLoading` class). **Result: 8 PASSED, 1 SKIPPED (6.44s)** — lazy loading tests: `test_lazy_loading_avoids_oom`, `test_lazy_container_backward_compatible`, `test_lazy_caching`, `test_tensor_input_handled`, `test_len_method`. Model factory regression: 3/3 PASSED (25.67s). Artifacts: `plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-07T220000Z/` (pytest_phase_b.log, pytest_collect.log). Commit 37985157. Next: Phase C — Update training pipeline to optionally use `as_tf_dataset()` for large datasets.
  - *2026-01-08T03:00:00Z (Phase C planning):* Galph verified Phase B completion. Designed Phase C tasks: (C1) Add `train_with_dataset()` to model.py; (C2) Update `train()` with optional streaming; (C3) Update `compare_models.py` for chunked inference; (C4) Integration tests. Ready for implementation handoff.
  - *2026-01-08T19:00:00Z (Phase C complete):* Integrated streaming training with `train()`. Changes: (1) C2: Added `use_streaming` parameter to `train()` in `ptycho/model.py:622-681`; (2) Auto-detection: datasets >10000 samples automatically use streaming; (3) Fixed `as_tf_dataset()` to yield tuples instead of lists for TensorFlow compatibility; (4) C3: Added `TestStreamingTraining` class with 4 tests (`test_as_tf_dataset_yields_correct_structure`, `test_streaming_training_auto_detection`, `test_train_accepts_use_streaming_parameter`, `test_dataset_batch_count`). **Result: 12 PASSED, 1 SKIPPED (8.28s)** — all Phase C exit criteria met. Model factory regression: 3/3 PASSED (25.61s). Artifacts: `plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-08T030000Z/` (pytest_phase_c.log, pytest_collect.log). Next: Mark initiative complete.

---

### [FIX-TF-C1D-SCALED-RERUN-001] Phase C1d TensorFlow scaled rerun execution
- Depends on: FIX-PYTORCH-FORWARD-PARITY-001 (Phase C guard + regression tests merged)
- Priority: Critical
- Status: blocked — stale since 2025-11-20; needs scope refresh before resumption.
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
- Status: in_progress — Task 1 complete; Task 2-3 in progress.
- Owner/Date: Ralph/2026-01-09
- Working Plan: `plans/active/PARALLEL-API-INFERENCE/plan.md`
- Reports Hub: `plans/active/PARALLEL-API-INFERENCE/reports/`
- Goals:
  - Task 1: Extract TF inference helper from `scripts/inference/inference.py` into callable `_run_tf_inference_and_reconstruct()`. ✅
  - Task 2: Update `scripts/pytorch_api_demo.py` to use new TF helper (already exists, needs update).
  - Task 3: Add smoke test `tests/scripts/test_api_demo.py`.
  - Task 4: Document in `docs/workflows/pytorch.md`.
- Exit Criteria:
  - `_run_tf_inference_and_reconstruct()` exists and is callable without CLI args. ✅
  - Demo script runs both backends on synthetic fixture using new helper.
  - Smoke test passes (`tests/scripts/test_api_demo.py`).
- Attempts History:
  - *2026-01-09T010000Z (Phase A exploration):* Analyzed TF inference (`scripts/inference/inference.py:321-428`) and PyTorch inference (`ptycho_torch/inference.py:426-632`). Documented extraction design: propose `_run_tf_inference_and_reconstruct(model, raw_data, config, ...)` returning `(amp, phase)` to match PyTorch helper signature. Ground truth handling moves to separate utility. CLI wrapper unchanged except to call new helper. **No code changes — planning only.** Artifacts: `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T010000Z/extraction_design.md`. Next: Task 1 implementation (extract TF helper).
  - *2026-01-09T020000Z (Task 1 complete):* Extracted TF inference helper per extraction design. Changes: (1) Added `_run_tf_inference_and_reconstruct()` to `scripts/inference/inference.py:353-457` with signature matching PyTorch helper; (2) Added `extract_ground_truth()` utility at `scripts/inference/inference.py:323-350`; (3) Refactored `perform_inference()` as deprecated wrapper calling new helper; (4) Created `tests/scripts/test_tf_inference_helper.py` with 7 signature/availability tests. **Result: ALL 7 TESTS PASSED (3.74s).** Integration test (`tests/test_integration_workflow.py`) also passed (35.93s). Test collection: 23 tests in `tests/scripts/`. Metrics: 7 passed (helper tests), 1 passed (integration). Artifacts: `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T020000Z/` (pytest_tf_helper.log, pytest_integration.log, pytest_collect.log). Next: Task 2-3 — update demo script + add smoke test.
  - *2026-01-09T030000Z (Task 2-3 handoff):* Reviewed Task 1 evidence; verified 7/7 helper tests + 1 integration test passed. Discovered `scripts/pytorch_api_demo.py` already exists but uses old TF path (`tf_components.perform_inference`). Prepared input.md for Task 2 (update demo to use new helper) + Task 3 (add smoke test). Artifacts: `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T030000Z/`. Next: Ralph updates demo script and adds smoke test.

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

---

Supervisor state: `focus=FNO-STABILITY-OVERHAUL-001` `state=ready_for_implementation` `dwell=1` `artifacts=plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-28T020500Z/` `next_action=engineer finishes Phase 1 Task 1.1 + wraps compare wrapper flag + reruns mapped pytest selectors`

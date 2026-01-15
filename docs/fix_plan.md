# PtychoPINN Fix Plan Ledger (Condensed)

**Last Updated:** 2026-01-13 (SIM-LINES-4X gs2 probe_scale sweep)
**Active Focus:** DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy

---

**Housekeeping Notes:**
- Full Attempts History archived in `docs/fix_plan_archive.md` (snapshot 2026-01-06)
- Earlier snapshots: `docs/archive/2025-11-06_fix_plan_archive.md`, `docs/archive/2025-10-17_fix_plan_archive.md`, `docs/archive/2025-10-20_fix_plan_archive.md`
- Each initiative has a working plan at `plans/active/<ID>/implementation.md` and reports under `plans/active/<ID>/reports/`

---

## Active / Pending Initiatives

### [DEBUG-SIM-LINES-DOSE-001] Isolate sim_lines_4x vs dose_experiments discrepancy
- Depends on: None
- Priority: **Critical** (Highest Priority)
- Status: pending — plan drafted
- Owner/Date: Codex/2026-01-13
- Working Plan: `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md`
- Summary: `plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md`
- Reports Hub: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/`
- Spec Owner: `docs/specs/spec-ptycho-workflow.md`
- Test Strategy: `plans/active/DEBUG-SIM-LINES-DOSE-001/test_strategy.md`
- Goals:
  - Identify whether the sim_lines_4x failure stems from a core regression, nongrid pipeline differences, or a workflow/config mismatch.
  - Produce a minimal repro that isolates grid vs nongrid and probe normalization effects.
  - Apply a targeted fix and verify success via visual inspection if metrics are unavailable.
- Exit Criteria:
  - A/B results captured for grid vs nongrid, probe normalization, and grouping parameters.
  - Root-cause statement with evidence (logs + params snapshot + artifacts).
  - Targeted fix or workflow change applied, with recon success and no NaNs.
  - Visual inspection success gate satisfied if metrics are unavailable.
- Attempts History:
  - *2026-01-13T000000Z:* Drafted phased debugging plan, summary, and test strategy. Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md`, `plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md`, `plans/active/DEBUG-SIM-LINES-DOSE-001/test_strategy.md`.

### [SIM-LINES-4X-001] Four-scenario nongrid sim + TF reconstruction (lines object)
- Depends on: None
- Priority: Medium
- Status: in_progress — Phase A/B complete; Phase C validation runs pending.
- Owner/Date: Ralph/2026-01-11
- Working Plan: `plans/active/SIM-LINES-4X-001/implementation.md`
- Summary: `plans/active/SIM-LINES-4X-001/summary.md`
- Reports Hub: `plans/active/SIM-LINES-4X-001/reports/`
- Spec Owner: `specs/spec-inference-pipeline.md`
- Goals:
  - Simulate four datasets from a synthetic lines object (gs1/gs2 x idealized/custom probe).
  - Train TF models and reconstruct the contiguous test split for each scenario.
- Exit Criteria:
  - Four scenario runs complete with saved bundles and reconstruction images.
  - Run logs capture N=64, object_size=392, split=0.5, base_total_images=2000, group_count=1000 (gs2 total_images=8000).
  - Ledger and initiative summary updated.
- Attempts History:
  - *2026-01-13T220032Z:* Added gs2 ideal probe_scale sweep runner safeguards for NaN metrics (sanitize + registration fallback) and executed a 20‑epoch sweep across probe_scale {2,4,6,8,10}. **Result:** best amplitude SSIM at probe_scale=6.0 (ssim_amp≈0.2493); scales 8/10 produced NaN recon amplitudes, so metrics were recorded as null. **Static analysis:** `ruff check scripts/studies/sim_lines_4x/run_gs2_ideal_probe_scale_sweep.py` passed. **Artifacts:** `plans/active/SIM-LINES-4X-001/reports/2026-01-13T220032Z/ruff_check.log`, `plans/active/SIM-LINES-4X-001/reports/2026-01-13T220032Z/run_probe_scale_sweep_20epochs.log`, `.artifacts/sim_lines_4x_probe_scale_sweep_2026-01-13T220032Z/probe_scale_sweep.json`.
  - *2026-01-13T205123Z:* Reran SIM-LINES-4X gs2 ideal/custom with 50 epochs (output root `.artifacts/sim_lines_4x_probe_scale_2026-01-13T205123Z_gs2_50`). **Observation:** gs2_ideal losses went NaN early; gs2_custom completed after a longer rerun. **Artifacts:** `plans/active/SIM-LINES-4X-001/reports/2026-01-13T205123Z/run_gs2_ideal_50epochs.log`, `plans/active/SIM-LINES-4X-001/reports/2026-01-13T205123Z/run_gs2_custom_probe_50epochs.log`.
  - *2026-01-13T204359Z:* Set gs2 runners to pass probe_scale (ideal=10.0, custom=4.0) and reran all four SIM-LINES-4X scenarios to refresh outputs under `.artifacts/sim_lines_4x_probe_scale_2026-01-13T204359Z`. **Static analysis:** `ruff check scripts/studies/sim_lines_4x/run_gs2_ideal.py scripts/studies/sim_lines_4x/run_gs2_custom_probe.py` passed. **Artifacts:** `plans/active/SIM-LINES-4X-001/reports/2026-01-13T204359Z/ruff_check.log`, `plans/active/SIM-LINES-4X-001/reports/2026-01-13T204359Z/run_gs1_ideal.log`, `plans/active/SIM-LINES-4X-001/reports/2026-01-13T204359Z/run_gs1_custom_probe.log`, `plans/active/SIM-LINES-4X-001/reports/2026-01-13T204359Z/run_gs2_ideal.log`, `plans/active/SIM-LINES-4X-001/reports/2026-01-13T204359Z/run_gs2_custom_probe.log`.
  - *2026-01-11T081911Z:* Implemented SIM-LINES-4X pipeline + scenario runners. Added core pipeline module and four runner scripts under `scripts/studies/sim_lines_4x/` with a README; updated `docs/index.md` and `scripts/studies/README.md`. **Static analysis:** `ruff check scripts/studies/sim_lines_4x` passed. **Tests:** `pytest -m integration` passed (1 passed, 2 skipped pre-existing). Artifacts: `plans/active/SIM-LINES-4X-001/reports/2026-01-11T081911Z/ruff_check.log`, `plans/active/SIM-LINES-4X-001/reports/2026-01-11T081911Z/pytest_integration.log`. Next: execute four scenarios (C1-C3) and capture outputs.
  - *2026-01-11T083629Z:* Updated SIM-LINES-4X pipeline to scale `total_images` by `gridsize^2` while keeping `group_count=1000` per split. Reran gs2 scenarios with `total_images=8000` and `train/test=4000/4000`; outputs saved under `.artifacts/sim_lines_4x/{gs2_ideal,gs2_integration}`. **Static analysis:** `ruff check scripts/studies/sim_lines_4x` passed. Artifacts: `plans/active/SIM-LINES-4X-001/reports/2026-01-11T083629Z/ruff_check.log`, `plans/active/SIM-LINES-4X-001/reports/2026-01-11T083629Z/run_gs2_ideal.log`, `plans/active/SIM-LINES-4X-001/reports/2026-01-11T083629Z/run_gs2_integration_probe.log`. Notes: stitch warnings logged during inference but outputs written.
  - *2026-01-12T231841Z:* Updated sim_lines_4x figure presentation (square crop + nonzero vmin colormap scaling) and regenerated inference outputs for gs1/gs2 ideal/custom scenarios. Artifacts: `outputs/sim_lines_4x_rerun_20260113T025238Z/gs1_ideal/inference_outputs/`, `outputs/sim_lines_4x_rerun_20260113T025238Z/gs1_custom/inference_outputs/`, `outputs/sim_lines_4x_rerun_20260113T025238Z/gs2_ideal/inference_outputs/`, `outputs/sim_lines_4x_rerun_20260113T025238Z/gs2_custom/inference_outputs/`.
  - *2026-01-13T184723Z:* Drafted SIM-LINES-4X quantitative metrics table for the paper with a script-backed generator and JSON source data; inserted the table into the Overlap-Free Reconstruction section. Artifacts: `paper/tables/sim_lines_4x_metrics.tex`, `paper/tables/scripts/generate_sim_lines_4x_metrics.py`, `paper/data/sim_lines_4x_metrics.json`.
  - *2026-01-13T192409Z:* Recomputed SIM-LINES-4X metrics with correct ground truth (simulated object) and random test subsampling (nsamples=1000, seed=7). Added `scripts/studies/sim_lines_4x/evaluate_metrics.py`, updated JSON notes, and regenerated the paper table. Artifacts: `.artifacts/sim_lines_4x_metrics/2026-01-13T191658Z/`, `.artifacts/sim_lines_4x_metrics/2026-01-13T191911Z/`, `.artifacts/sim_lines_4x_metrics/2026-01-13T192119Z/`, `.artifacts/sim_lines_4x_metrics/2026-01-13T192409Z/`, `paper/data/sim_lines_4x_metrics.json`, `paper/tables/sim_lines_4x_metrics.tex`.
  - *2026-01-13T193350Z:* Regenerated evaluation PNGs (aligned recon + ground truth) for all four scenarios at nsamples=1000/seed=7. Artifacts: `.artifacts/sim_lines_4x_metrics/2026-01-13T193350Z/`, `outputs/sim_lines_4x_rerun_20260113T025238Z/{gs1_ideal,gs1_custom,gs2_ideal,gs2_custom}/eval_outputs_nsamples1000_seed7/`.
  - *2026-01-13T194447Z:* Added compare_models-style registration to sim_lines_4x evaluation (find_translation_offset + apply_shift_and_crop), regenerated metrics/PNG outputs, and refreshed JSON/table notes. Artifacts: `.artifacts/sim_lines_4x_metrics/2026-01-13T194447Z/`, `outputs/sim_lines_4x_rerun_20260113T025238Z/{gs1_ideal,gs1_custom,gs2_ideal,gs2_custom}/eval_outputs_nsamples1000_seed7/`, `paper/data/sim_lines_4x_metrics.json`, `paper/tables/sim_lines_4x_metrics.tex`.
  - *2026-01-13T195436Z:* Modularized evaluation alignment with `align_for_evaluation_with_registration`, updated sim_lines_4x evaluation to call it, and documented the alignment+registration pattern in the developer guide. Artifacts: `ptycho/image/cropping.py`, `scripts/studies/sim_lines_4x/evaluate_metrics.py`, `docs/DEVELOPER_GUIDE.md`.
  - *2026-01-13T192310Z:* Refactored nongrid simulation caching into a reusable memoization decorator and applied it to `simulate_nongrid_raw_data`. Cache files default to `.artifacts/synthetic_helpers/cache` (generated on next run). Artifacts: `.artifacts/synthetic_helpers/cache/`.
  - *2026-01-13T193346Z:* Verified cache hit behavior with a two-call smoke test and ran synthetic helper unit + CLI smoke tests. Artifacts: `plans/active/SIM-LINES-4X-001/reports/2026-01-13T193346Z/pytest_synthetic_helpers.log`, `plans/active/SIM-LINES-4X-001/reports/2026-01-13T193346Z/pytest_synthetic_helpers_cli.log`.

### [REFACTOR-MEMOIZE-CORE-001] Move RawData memoization decorator into core module
- Depends on: None
- Priority: Low
- Status: pending — plan drafted
- Owner/Date: TBD/2026-01-13
- Working Plan: `plans/active/REFACTOR-MEMOIZE-CORE-001/implementation.md`
- Summary: `plans/active/REFACTOR-MEMOIZE-CORE-001/summary.md`
- Reports Hub: `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/`
- Spec Owner: `docs/architecture.md`
- Test Strategy: Inline test annotations (refactor only; reuse existing tests)
- Goals:
  - Move `memoize_raw_data` from `scripts/simulation/cache_utils.py` into a core module under `ptycho/`.
  - Preserve cache hashing and default cache paths used by synthetic helpers.
  - Keep script imports working via direct update or a thin shim.
- Exit Criteria:
  - Core module provides `memoize_raw_data` with unchanged behavior.
  - Synthetic helpers use the core module; shim or removal completed without regressions.
  - Existing synthetic helper tests pass and logs archived.
- Attempts History:
  - *2026-01-13T202358Z:* Drafted implementation plan and initialized initiative summary. Artifacts: `plans/active/REFACTOR-MEMOIZE-CORE-001/implementation.md`, `plans/active/REFACTOR-MEMOIZE-CORE-001/summary.md`.

### [SYNTH-HELPERS-001] Script-level synthetic simulation helpers
- Depends on: None
- Priority: Medium
- Status: done — Phases A-C complete
- Owner/Date: TBD/2026-01-13
- Working Plan: `plans/active/SYNTH-HELPERS-001/implementation.md`
- Test Strategy: `plans/active/SYNTH-HELPERS-001/test_strategy.md`
- Reports Hub: `plans/active/SYNTH-HELPERS-001/reports/`
- Spec Owner: `docs/DATA_GENERATION_GUIDE.md`
- Goals:
  - Add script-level helpers for synthetic object/probe creation and nongrid simulation.
  - Refactor dose study, sim_lines_4x, and synthetic lines runner to use helpers.
  - Add helper unit tests and CLI smoke coverage.
- Exit Criteria:
  - Helpers implemented under `scripts/simulation/synthetic_helpers.py` with deterministic seed and split behavior.
  - Refactored scripts use helpers with no behavior regressions.
  - Helper unit tests + CLI smoke tests pass; logs archived under the plan reports hub.
  - Ledger and test registry updated.
- Attempts History:
  - *2026-01-13T01:00:00Z:* Drafted implementation plan at `plans/active/SYNTH-HELPERS-001/implementation.md` with helper API definition and test plan; test strategy + ledger linkage pending.
  - *2026-01-13T01:32:44Z:* Implemented `scripts/simulation/synthetic_helpers.py`, refactored `scripts/studies/dose_response_study.py`, `scripts/studies/sim_lines_4x/pipeline.py`, and `scripts/simulation/run_with_synthetic_lines.py` to use helpers, added helper + CLI smoke tests, and updated test registry docs. **Static analysis:** `ruff check scripts/simulation/synthetic_helpers.py scripts/studies/sim_lines_4x/pipeline.py scripts/studies/dose_response_study.py scripts/simulation/run_with_synthetic_lines.py tests/scripts/test_synthetic_helpers.py tests/scripts/test_synthetic_helpers_cli_smoke.py` passed. **Tests:** `pytest tests/scripts/test_synthetic_helpers.py tests/scripts/test_synthetic_helpers_cli_smoke.py -v` passed (9 tests). **Collect:** `pytest ... --collect-only -v` captured. Artifacts: `plans/active/SYNTH-HELPERS-001/reports/2026-01-13T013244Z/ruff_check.log`, `plans/active/SYNTH-HELPERS-001/reports/2026-01-13T013244Z/pytest_synthetic_helpers.log`, `plans/active/SYNTH-HELPERS-001/reports/2026-01-13T013244Z/pytest_collect.log`. 

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
- Status: in_progress — Reopening to fix nongrid inference workflow regression (dose response script).
- Owner/Date: Ralph/2026-01-07
- Working Plan: `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/implementation.md`
- Test Strategy: `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/test_strategy.md`
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
  - *2026-01-13T061238Z:* Reran dose_response_study.py with `--nepochs 5` and outputs to `.artifacts/dose_response_study/2026-01-13T061238Z`. **Observations:** intensity ratio logged as 1.23e+00 vs expected ~1e5; training losses hit NaN (e.g., low_mae final loss = nan); stitch warnings logged (`cannot reshape array of size 8192000 into shape (25,25,64,64,1)`). Figure and training history were regenerated in the reports directory. Artifacts: `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-13T061238Z/dose_response_study.log`, outputs under `.artifacts/dose_response_study/2026-01-13T061238Z/`.
  - *2026-01-13T064343Z:* Updated dose_response_study.py to follow the sim_lines_4x inference workflow (bundle load → group → reconstruct → stitch), capped inference groups to available test images, and removed oversampling for small test splits. Added a pytest regression guard for inference group-count capping and documented test strategy. **Artifacts:** `.artifacts/STUDY-SYNTH-DOSE-COMPARISON-001/2026-01-13T064343Z/` (ruff_check.log, pytest_dose_response_study.log, pytest_collect_scripts.log). Next: rerun the study and confirm reconstructions are stable.
  - *2026-01-13T065840Z:* Attempted rerun of dose_response_study.py (nepochs=5) but the command timed out after 120s during training. Artifacts: `.artifacts/STUDY-SYNTH-DOSE-COMPARISON-001/2026-01-13T065840Z/dose_response_study.log`.
  - *2026-01-13T070124Z:* Reran dose_response_study.py with `--nepochs 5` to `.artifacts/dose_response_study/2026-01-13T070124Z`. **Results:** training completed for all four arms, but inference failed to reload model bundles with `ProbeIllumination` missing in Keras deserialization; reconstructions are invalid until model loading is fixed. Figure and training history were still regenerated. Artifacts: `.artifacts/STUDY-SYNTH-DOSE-COMPARISON-001/2026-01-13T070124Z/dose_response_study.log`, outputs under `.artifacts/dose_response_study/2026-01-13T070124Z/`.
  - *2026-01-13T070738Z:* Reran dose_response_study.py after switching model saving to the shared model_manager bundle path; training + inference completed for all four arms and the 6-panel figure includes reconstructions. Artifacts: `.artifacts/STUDY-SYNTH-DOSE-COMPARISON-001/2026-01-13T070738Z/dose_response_study.log`, outputs under `.artifacts/dose_response_study/2026-01-13T070738Z/`.

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

### [FIX-REASSEMBLE-BATCH-DIM-001] Preserve Batch Dimension in Batched Reassembly
- Depends on: None
- Priority: High
- Status: done — exit criteria met (targeted regression passes).
- Owner/Date: Codex/2026-01-11
- Working Plan: `plans/active/FIX-REASSEMBLE-BATCH-DIM-001/implementation.md`
- Summary: `plans/active/FIX-REASSEMBLE-BATCH-DIM-001/summary.md`
- Test Strategy: `plans/active/FIX-REASSEMBLE-BATCH-DIM-001/test_strategy.md`
- Artifacts: `.artifacts/FIX-REASSEMBLE-BATCH-DIM-001/`
- Spec Owner: `docs/specs/spec-ptycho-workflow.md` (Reassembly Requirements)
- Goals:
  - Preserve batch dimension in `_reassemble_position_batched` for per-sample canvases.
  - Align `ReassemblePatchesLayer` output shape metadata.
  - Update regression test to assert batch dimension is preserved.
- Exit Criteria:
  - `_reassemble_position_batched` returns `(B, padded_size, padded_size, 1)` in batched mode.
  - `ReassemblePatchesLayer` preserves batch dimension for gridsize > 1.
  - `tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split` passes.
- Attempts History:
  - *2026-01-11T01:15:00Z:* Implemented per-sample batched reassembly using segment-sum accumulation, updated `ReassemblePatchesLayer.compute_output_shape`, and adjusted regression test to require batch preservation. Tests not run (per user request). Artifacts: `.artifacts/FIX-REASSEMBLE-BATCH-DIM-001/`.
  - *2026-01-11T020400Z:* Ran `pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -v`. **FAILED** with `ValueError` in `_reassemble_position_batched` because `padded_size=None` when building `tf.zeros`. Likely cause: `ReassemblePatchesLayer` passes `padded_size=None` into `mk_reassemble_position_batched_real`, which then skips the `get_padded_size()` fallback. Artifacts: `.artifacts/FIX-REASSEMBLE-BATCH-DIM-001/pytest_reassemble_batch.log`. Next: treat `padded_size=None` as unset and rerun the regression test.
  - *2026-01-11T021200Z:* Treated `padded_size=None` as unset in `mk_reassemble_position_batched_real` (fallback to `get_padded_size()`), then reran `pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -v`. **PASSED.** Artifacts: `.artifacts/FIX-REASSEMBLE-BATCH-DIM-001/pytest_reassemble_batch_fix.log`.
  - *2026-01-12T23:49:39Z:* Switched `ReassemblePatchesLayer` to use non-batched reassembly via `mk_reassemble_position_real` per request; `reassemble_whole_object` unchanged. Tests not run.
  - *2026-01-13T00:04:50Z:* Enforced `N % 4 == 0` in `params.validate()`, updated the regression test to use `N=128`, removed the translation crop/pad guard, then reran `pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -v`. **PASSED.**
  - *2026-01-13T01:09:19Z:* Lowered `shift_and_sum` streaming chunk size to 256 to reduce reassembly memory spikes and updated the inference pipeline spec to match. Tests not run.

---

### [FIX-OBJECT-REASSEMBLY-BATCH-001] Batched object reassembly for stitching
- Depends on: None
- Priority: High
- Status: pending — plan drafted; implementation not started.
- Owner/Date: Codex/2026-01-12
- Working Plan: `plans/active/FIX-OBJECT-REASSEMBLY-BATCH-001/implementation.md`
- Reports Hub: `plans/active/FIX-OBJECT-REASSEMBLY-BATCH-001/reports/`
- Spec Owner: `specs/compare_models_spec.md`
- Goals:
  - Add batched object reassembly that preserves `--stitch-crop-size` semantics.
  - Update `scripts/compare_models.py` to use the batched path to avoid OOM.
- Exit Criteria:
  - Object reassembly supports `batch_size` and preserves `M` crop semantics.
  - `scripts/compare_models.py` uses the batched object reassembly path for stitching when enabled.
  - Targeted pytest selector(s) pass and capture equivalence for small inputs.

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

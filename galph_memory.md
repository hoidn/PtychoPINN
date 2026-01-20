# 2026-01-15T22:58:50Z: REFACTOR-MEMOIZE-CORE-001 — Phase A inventory + B-stage handoff

- dwell: 0 (new focus; no prior loops for this initiative)
- Focus issue: REFACTOR-MEMOIZE-CORE-001 — Move RawData memoization decorator into a core module
- Action type: Planning → Implementation handoff (Phase A complete, Phase B scoped)
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/index.md, docs/findings.md, docs/architecture.md, docs/DEVELOPER_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/fix_plan.md, plans/active/REFACTOR-MEMOIZE-CORE-001/implementation.md, scripts/simulation/cache_utils.py, scripts/simulation/synthetic_helpers.py
- Key decisions:
  - Add `ptycho/cache.py` with `_hash_numpy`, `_normalize_for_hash`, `_hash_payload`, and `memoize_raw_data` to keep caching logic in the core package.
  - Convert `scripts/simulation/cache_utils.py` into a shim that re-exports the decorator and emits a DeprecationWarning.
  - Update `scripts/simulation/synthetic_helpers.py` to import from the new module; no other production files touched.
- input.md updated with Do Now (Phase B1-B3) + pytest selectors + artifacts hub `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T225850Z/`.
- <Action State>: [ready_for_implementation]
- focus=REFACTOR-MEMOIZE-CORE-001 state=ready_for_implementation dwell=0 ralph_last_commit=none artifacts=plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T225850Z/ next_action=move memoize_raw_data into ptycho/cache with shim + run synthetic helper script tests

# 2026-01-15T23:36:22Z: PARALLEL-API-INFERENCE — Task 2 implementation handoff

- dwell: 0 (new focus after closing REFACTOR-MEMOIZE-CORE-001)
- Focus issue: PARALLEL-API-INFERENCE — Programmatic TF/PyTorch API parity (Task 2 demo parity + smoke test)
- Action type: Planning → Implementation handoff
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, ANTIPATTERN-001), docs/architecture.md, docs/DEVELOPER_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/workflows/pytorch.md, specs/ptychodus_api_spec.md:200-360, plans/active/PARALLEL-API-INFERENCE/plan.md, plans/active/PARALLEL-API-INFERENCE/summary.md, scripts/pytorch_api_demo.py, tests/scripts/test_api_demo.py, scripts/inference/inference.py
- Key decisions:
  - Verified REFACTOR-MEMOIZE-CORE-001 Phase C docs/tests landed and marked the initiative done in `docs/fix_plan.md`.
  - Activated PARALLEL-API-INFERENCE Task 2, refreshed the initiative summary, and opened a new artifacts hub `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-15T233622Z/`.
  - Wrote `input.md` instructing Ralph to swap the demo’s TensorFlow path over to `_run_tf_inference_and_reconstruct` + `extract_ground_truth` and run the quick pytest selectors.
- <Action State>: [ready_for_implementation]
- focus=PARALLEL-API-INFERENCE state=ready_for_implementation dwell=0 ralph_last_commit=d053d049 artifacts=plans/active/PARALLEL-API-INFERENCE/reports/2026-01-15T233622Z/ next_action=update scripts/pytorch_api_demo.py TF branch to call _run_tf_inference_and_reconstruct + run mapped pytest selectors

# 2026-01-15T23:21:07Z: REFACTOR-MEMOIZE-CORE-001 — Phase C doc/test handoff

- dwell: 1 (second loop on this focus; still no production evidence)
- Focus issue: REFACTOR-MEMOIZE-CORE-001 — Move RawData memoization decorator into a core module
- Action type: Implementation handoff (Phase C docs + regression evidence)
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (ANTIPATTERN-001, MIGRATION-001), docs/architecture.md, docs/TESTING_GUIDE.md, plans/active/REFACTOR-MEMOIZE-CORE-001/{implementation.md,summary.md}, scripts/simulation/README.md, scripts/simulation/synthetic_helpers.py, ptycho/cache.py
- Key updates:
  - Marked Phase B checklist complete in the plan (commit `d29efc91` landed the code) and expanded Phase C tasks with explicit doc + pytest deliverables.
  - Refreshed `docs/fix_plan.md` status/Attempts History to show Phase C in progress and created artifacts hub `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T232107Z/`.
  - Rewrote `input.md` with the doc updates (docs/index.md entry + scripts/simulation/README.md cache guidance) and the three mapped pytest selectors (including `--collect-only` evidence).
- Next Action: Ralph updates the two docs, reruns the synthetic helper selectors, archives logs under the new report directory, and reports back for C3 closure.
- <Action State>: [ready_for_implementation]
- focus=REFACTOR-MEMOIZE-CORE-001 state=ready_for_implementation dwell=1 ralph_last_commit=d29efc91 artifacts=plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T232107Z/ next_action=doc updates (docs/index.md + scripts/simulation/README.md) + rerun pytest selectors + archive logs

---

# 2026-01-09T03:00:00Z: PARALLEL-API-INFERENCE — Task 1 verified, Task 2-3 handoff

- dwell: 2 (continuing focus; dwell incremented since last loop was implementation)
- Focus issue: PARALLEL-API-INFERENCE — Programmatic TF/PyTorch API parity (Task 2-3)
- Action type: Review → Implementation handoff
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/fix_plan.md, galph_memory.md, plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T020000Z/*.log (Task 1 evidence), scripts/pytorch_api_demo.py (existing demo), scripts/inference/inference.py:323-507 (new helpers).

**Task 1 Verification (COMPLETE ✅):**
- Ralph commit 781ec2f2: Extracted TF helper per design
- Tests: 7/7 passed (`tests/scripts/test_tf_inference_helper.py`, 3.74s)
- Integration: 1/1 passed (`tests/test_integration_workflow.py`, 35.93s)
- Evidence: `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T020000Z/` (pytest_tf_helper.log, pytest_integration.log)

**Task 2 Scope (Revised):**
- `scripts/pytorch_api_demo.py` **already exists** (discovered during review)
- Currently uses old TF path: `tf_components.perform_inference`
- Need to update to use new `_run_tf_inference_and_reconstruct` helper for API parity
- PyTorch path unchanged (already uses `torch_infer`)

**Task 3 Scope:**
- Create `tests/scripts/test_api_demo.py` smoke test
- Fast tests: import checks, signature validation
- Slow tests: full backend execution (marked `@pytest.mark.slow`)

**input.md Updated:** Task 2-3 with demo script update and smoke test creation.

- Next: Ralph updates demo script TF path, adds smoke test
- <Action State>: [ready_for_implementation]
- focus=PARALLEL-API-INFERENCE state=ready_for_implementation dwell=2 ralph_last_commit=781ec2f2 artifacts=plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T030000Z/ next_action=implement Task 2-3 (demo script update + smoke test)

---

# 2026-01-09T02:00:00Z: PARALLEL-API-INFERENCE — Task 1 implementation handoff

- dwell: 1 (continuing focus from prior planning loop)
- Focus issue: PARALLEL-API-INFERENCE — Programmatic TF/PyTorch API parity (Task 1)
- Action type: Implementation handoff (TF helper extraction)
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/fix_plan.md, galph_memory.md (prior entry), docs/findings.md (CONFIG-001, ANTIPATTERN-001), plans/active/PARALLEL-API-INFERENCE/plan.md, plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T010000Z/extraction_design.md, scripts/inference/inference.py:321-428 (perform_inference), ptycho_torch/inference.py:426-632 (_run_inference_and_reconstruct).

**Phase A Review:**
- Extraction design complete: `reports/2026-01-09T010000Z/extraction_design.md`
- Proposed signature mirrors PyTorch helper: `_run_tf_inference_and_reconstruct(model, raw_data, config, ...) -> (amp, phase)`
- Ground truth handling moved to separate `extract_ground_truth()` utility
- `perform_inference` becomes deprecated wrapper for backward compatibility

**Task 1 Implementation Scope:**
1. Create `_run_tf_inference_and_reconstruct()` in `scripts/inference/inference.py`
2. Create `extract_ground_truth()` utility
3. Refactor `perform_inference` as deprecated wrapper calling new helper
4. Create `tests/scripts/test_tf_inference_helper.py` with signature validation tests
5. Run integration regression to verify no breakage

**Mapped Tests:**
- `tests/scripts/test_tf_inference_helper.py::TestTFInferenceHelper` (new, 4 tests)
- `tests/test_integration_workflow.py` (regression)

**input.md Updated:** Task 1 implementation with full code samples and verification commands.

- Next: Ralph extracts TF helper, creates test, runs verification
- <Action State>: [ready_for_implementation]
- focus=PARALLEL-API-INFERENCE state=ready_for_implementation dwell=1 ralph_last_commit=c46e496a artifacts=plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T020000Z/ next_action=implement Task 1 (TF helper extraction + tests)

---

# 2026-01-09T01:00:00Z: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — G-scaled COMPLETE, focus selection

- dwell: 0 (new loop after G-scaled completion)
- Focus issue: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 → PARALLEL-API-INFERENCE
- Action type: Review/Housekeeping (milestone verification + focus selection)
- Mode: Planning
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/fix_plan.md, galph_memory.md, docs/findings.md (PINN-CHUNKED-001 RESOLVED), plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T210000Z/*.log, plans/active/PARALLEL-API-INFERENCE/plan.md

**G-scaled Verification COMPLETE:**
- Ralph commits c46e496a (integration test) + 5e1cf26f (evidence)
- Tests: 14/15 passed (1 intentional OOM skip), model factory 3/3, integration 2/2
- PINN-CHUNKED-001: ✅ RESOLVED in docs/findings.md
- docs/fix_plan.md updated: G-scaled marked COMPLETE

**Focus Selection:**
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 G-full: blocked on BASELINE-CHUNKED-001/002
- FIX-PYTORCH-FORWARD-PARITY-001: blocked on FIX-TF-C1D-SCALED-RERUN-001
- FIX-TF-C1D-SCALED-RERUN-001: blocked (stale since 2025-11-20)
- PARALLEL-API-INFERENCE: planning status, unblocked — selected as next focus

**Next Focus: PARALLEL-API-INFERENCE**
The initiative is in planning status with a clear scope:
1. Extract TF inference helper from scripts/inference/inference.py
2. Build scripts/pytorch_api_demo.py for both backends
3. Add smoke test tests/scripts/test_api_demo.py
4. Document in docs/workflows/pytorch.md

- Next: Write input.md for PARALLEL-API-INFERENCE Phase A (exploration + task 1)
- <Action State>: [planning]
- focus=PARALLEL-API-INFERENCE state=planning dwell=0 ralph_last_commit=c46e496a artifacts=N/A next_action=write input.md for PARALLEL-API-INFERENCE exploration

---

# 2026-01-08T21:00:00Z: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — G-scaled verification review + next phase

- dwell: 1 (continuing focus; Ralph committed G-scaled test in db8f15bd + evidence in 995fcc68)
- Focus issue: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — G-scaled: Verify lazy loading enables chunked PINN inference
- Action type: Review → Implementation handoff (phase progression)
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/fix_plan.md, galph_memory.md (prior entry), docs/findings.md (PINN-CHUNKED-001 RESOLVED), pytest_lazy_loading.log (14 tests, 13 passed), pytest_model_factory.log (3/3 passed), plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md.

**Ralph's Work Verified:**
- Commit db8f15bd: Added `test_container_numpy_slicing_for_chunked_inference`
- Commit 995fcc68: Archived pytest evidence
- Tests: 14 collected, 13 passed, 1 OOM skipped
- Model factory regression: 3/3 PASSED (25.87s)
- PINN-CHUNKED-001: Marked RESOLVED in docs/findings.md

**G-scaled Test Outcome:**
- `_X_np`, `_coords_nominal_np` attributes exist on container ✅
- NumPy slicing works without populating tensor cache ✅
- Backward-compatible `.X` access still works ✅

**Next Phase: G-scaled Integration Test**
The unit test proves the API supports chunked access. Now need to verify compare_models.py can actually use this API without OOM. This requires a minimal integration test with real compare_models invocation.

**Blocker Assessment:**
- PINN-CHUNKED-001: ✅ RESOLVED (lazy container enables NumPy slicing)
- BASELINE-CHUNKED-001/002: Still active (Baseline inference OOM) — separate issue from PINN chunking
- G-full: Still blocked on resource/architecture limits
- G-scaled: Ready for integration test

**input.md Updated:** Integration test to run compare_models with chunked PINN using lazy container.

- Next: Ralph runs minimal chunked compare_models integration to verify OOM resolution
- <Action State>: [ready_for_implementation]
- focus=STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 state=ready_for_implementation dwell=1 ralph_last_commit=995fcc68 artifacts=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T210000Z/ next_action=run minimal chunked compare_models integration test

---

# 2026-01-08T20:30:00Z: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — G-scaled verification handoff

- dwell: 0 (new focus after FEAT-LAZY-LOADING-001 completion)
- Focus issue: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — G-scaled: Verify lazy loading enables chunked PINN inference
- Action type: Implementation handoff (chunking verification test)
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/fix_plan.md, galph_memory.md, docs/findings.md (PINN-CHUNKED-001 resolved), ptycho/loader.py (lazy container), plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md (prior state).

**Prior Blocker Analysis:**
- PINN-CHUNKED-001: ✅ RESOLVED via FEAT-LAZY-LOADING-001
- BASELINE-CHUNKED-001/002: Still active (Baseline OOM issues)
- Root cause was `PtychoDataContainer.__init__` → `tf.convert_to_tensor()` for all data

**Lazy Loading Enables:**
- `._X_np`, `._Y_I_np`, etc. — direct NumPy array access
- Chunk-wise slicing without GPU allocation
- `as_tf_dataset(batch_size)` for streaming training

**G-scaled Task:**
Add test `test_container_numpy_slicing_for_chunked_inference` to verify:
1. Container exposes `_X_np`, `_coords_nominal_np` for chunking
2. Chunk slicing works without triggering full tensor conversion
3. Backward compatibility (full `.X` access still works)

**input.md Updated:** G1-G3 tasks for chunking verification test.

- Next: Ralph adds chunking test, runs all lazy loading tests, archives evidence
- <Action State>: [ready_for_implementation]
- focus=STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 state=ready_for_implementation dwell=0 ralph_last_commit=bd8d9480 artifacts=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T200000Z/ next_action=implement G-scaled chunking verification test

---

# 2026-01-08T20:00:00Z: FEAT-LAZY-LOADING-001 — INITIATIVE COMPLETE, focus selection

- dwell: 0 (initiative complete; no further loops needed)
- Focus issue: FEAT-LAZY-LOADING-001 — ✅ COMPLETE
- Action type: Review/Housekeeping (verification of Phase C completion + focus selection)
- Mode: Review
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/fix_plan.md, galph_memory.md (prior entry), plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-08T030000Z/pytest_phase_c.log, pytest --collect-only output (13 tests in test_lazy_loading.py).

**Phase C Verification (COMPLETE):**
- Tests PASSED: 12/13 (1 OOM skip)
  - TestEagerLoadingOOM: 3 memory scaling + 1 OOM skip
  - TestLazyLoading: 5 tests (avoids_oom, backward_compat, caching, tensor_input, len_method)
  - TestStreamingTraining: 4 tests (dataset_structure, auto_detection, use_streaming_param, batch_count)
- Regression: `test_model_factory.py` 3/3 PASSED (25.61s)
- Commit bd8d9480: FEAT-LAZY-LOADING-001 Phase C complete

**Initiative Summary:**
FEAT-LAZY-LOADING-001 is DONE. All phases complete:
- Phase A: OOM reproduction tests
- Phase B: Lazy container (`PtychoDataContainer` with `_X_np` storage, lazy properties, `as_tf_dataset()`)
- Phase C: Training integration (`train()` with `use_streaming` parameter, auto-detection)

**PINN-CHUNKED-001 Finding:** Marked RESOLVED in `docs/findings.md`. Large datasets can now use `as_tf_dataset()` or direct NumPy slicing.

**Focus Selection:**
- FEAT-LAZY-LOADING-001: done ✅
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001: ~~blocked_escalation~~ → pending (PINN-CHUNKED-001 resolved)
- FIX-TF-C1D-SCALED-RERUN-001: ~~in_progress~~ → blocked (stale since 2025-11-20)
- FIX-PYTORCH-FORWARD-PARITY-001: blocked on FIX-TF-C1D-SCALED-RERUN-001
- PARALLEL-API-INFERENCE: planning

**Decision:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 was blocked primarily on PINN-CHUNKED-001 (OOM). With lazy loading complete, the G-scaled goal is now feasible. Marked FIX-TF-C1D-SCALED-RERUN-001 as blocked (stale).

- <Action State>: [planning]
- focus=FOCUS_SELECTION state=planning dwell=0 ralph_last_commit=bd8d9480 artifacts=plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-08T030000Z/ next_action=handoff G-scaled verification

---

# 2026-01-11T02:06:00Z: FIX-REASSEMBLE-BATCH-DIM-001 — Targeted test run (FAILED)

- dwell: 0 (new focus)
- Focus issue: FIX-REASSEMBLE-BATCH-DIM-001 — Preserve batch dimension in batched reassembly
- Action type: Evidence collection (targeted pytest)
- Mode: Planning
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (REASSEMBLY-BATCH-001, BUG-TF-REASSEMBLE-001), docs/debugging/debugging.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, specs/spec-ptycho-workflow.md, specs/spec-inference-pipeline.md, plans/active/FIX-REASSEMBLE-BATCH-DIM-001/{implementation.md,summary.md,test_strategy.md}

**Test Result:**
- Ran `pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -v`
- FAILED: `ValueError` in `_reassemble_position_batched` because `padded_size=None` when building `tf.zeros`.
- Likely cause: `ReassemblePatchesLayer` passes `padded_size=None` into `mk_reassemble_position_batched_real`, which then skips the `get_padded_size()` fallback.
- Artifact: `.artifacts/FIX-REASSEMBLE-BATCH-DIM-001/pytest_reassemble_batch.log`

**Next Action:**
Implement a guard that treats `padded_size=None` as unset (use `params.get_padded_size()`), then rerun the regression test.

- <Action State>: [ready_for_implementation]
- focus=FIX-REASSEMBLE-BATCH-DIM-001 state=ready_for_implementation dwell=0 ralph_last_commit=none artifacts=.artifacts/FIX-REASSEMBLE-BATCH-DIM-001/ next_action=fix padded_size None handling in batched reassembly path and rerun test

---

# 2026-01-08T03:00:00Z: FEAT-LAZY-LOADING-001 — Phase C: Training pipeline integration handoff

- dwell: 0 (reset after Phase B completion; commit 37985157 landed with 8/8 tests passing)
- Focus issue: FEAT-LAZY-LOADING-001 — Implement Lazy Tensor Allocation in loader.py
- Action type: Implementation handoff (Phase C tasks ready)
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/fix_plan.md, galph_memory.md (prior entry), plans/active/FEAT-LAZY-LOADING-001/implementation.md, tests/test_lazy_loading.py (Phase B tests verified), ptycho/loader.py:97-321 (lazy container + as_tf_dataset), ptycho/model.py:611-660 (training loop to update), docs/findings.md (PINN-CHUNKED-001 marked resolved).

**Phase B Verification:**
- Tests PASSED: 8/8 (1 OOM test skipped by default)
  - `test_lazy_loading_avoids_oom` ✅
  - `test_lazy_container_backward_compatible` ✅
  - `test_lazy_caching` ✅
  - `test_tensor_input_handled` ✅
  - `test_len_method` ✅
  - 3 memory scaling tests ✅
- Regression: `test_model_factory.py` 3/3 PASSED (25.67s)
- Commit 37985157: FEAT-LAZY-LOADING-001 Phase B complete

**Phase C Design:**
- Goal: Integrate lazy container with training pipeline
- C1: (Optional) Add `train_with_dataset()` helper to model.py
- C2: Update `train()` with `use_streaming` parameter; auto-detect for large datasets (>10000)
- C3: Add `test_streaming_training_small_dataset` integration test
- C4: Run tests and verify no regressions

**input.md Updated:** Full implementation code for Phase C with streaming training pattern.

- Next: Ralph implements streaming support in `train()`, adds integration test, runs all tests
- <Action State>: [ready_for_implementation]
- focus=FEAT-LAZY-LOADING-001 state=ready_for_implementation dwell=0 ralph_last_commit=37985157 artifacts=plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-08T030000Z/ next_action=implement Phase C (streaming training + integration tests)

---
# 2026-01-15T23:59:00Z: DEBUG-SIM-LINES-DOSE-001 — Phase A kickoff

- dwell: 0 (new focus; reactivated from archive)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
- Action type: Planning → Implementation handoff (Phase A evidence capture)
- Mode: Implementation
- Git sync: not run (focus update only)
- Documents reviewed: docs/index.md, docs/findings.md, docs/debugging/debugging.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/fix_plan.md, prompts/main.md, prompts/supervisor.md, plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md, plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md, plans/active/DEBUG-SIM-LINES-DOSE-001/test_strategy.md
- Key decisions:
  - Reactivated DEBUG-SIM-LINES-DOSE-001 in `docs/fix_plan.md` and set it as the sole active focus.
  - Paused PARALLEL-API-INFERENCE until Phase A evidence capture is underway.
  - Prepared `input.md` to add a lightweight params snapshot tool and archive Phase A artifacts.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=0 ralph_last_commit=none artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-15T235900Z/ next_action=add sim_lines_4x params snapshot tool + capture dose_experiments defaults + run pipeline import smoke test

# 2026-01-16T00:05:00Z: DEBUG-SIM-LINES-DOSE-001 — Phase A Do Now refresh

- dwell: 1 (first supervisor pass since reactivation)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Phase A evidence capture (sim_lines_4x vs dose_experiments)
- Action type: Planning | Mode: Implementation handoff
- Git sync: `git pull --rebase` (already up to date).
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, MODULE-SINGLETON-001, NORMALIZATION-001, BUG-TF-001), plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md,test_strategy.md}, scripts/studies/sim_lines_4x/pipeline.py, specs/data_contracts.md, docs/DATA_GENERATION_GUIDE.md, docs/debugging/debugging.md.
- Key decisions:
  - Created new artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/` and recorded the Do Now in docs/fix_plan.md Attempts.
  - Delegated creation of `scripts/tools/collect_sim_lines_4x_params.py` (metadata-only snapshot CLI) plus git history capture for the legacy `dose_experiments` reference.
  - Mapped pytest selector `tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke` for validation and captured explicit CLI commands in input.md.
- Next: Ralph implements the snapshot tool, inventories the old defaults, and archives the smoke-test log.
- <Action State>: [planning]
- focus=DEBUG-SIM-LINES-DOSE-001 state=planning dwell=1 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/ next_action=ready_for_implementation

---
# 2026-01-16T00:32:17Z: DEBUG-SIM-LINES-DOSE-001 — Phase A4 comparison prep

- dwell: 2 (second consecutive planning/document loop for this focus; next turn must hand off implementation)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Phase A evidence capture (sim_lines_4x vs dose_experiments)
- Action type: Planning | Mode: Implementation handoff
- Git sync: `git pull --rebase` (already up to date).
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, MODULE-SINGLETON-001, NORMALIZATION-001, BUG-TF-001), plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md,test_strategy.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/{sim_lines_4x_params_snapshot.json,dose_experiments_param_scan.md}, scripts/tools/collect_sim_lines_4x_params.py.
- Key decisions:
  - Confirmed Phase A1/A3 evidence is complete and marked the checklist accordingly in the working plan.
  - Added a new Attempts History entry + artifacts hub (`plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/`).
  - Authored the Phase A4 Do Now for a comparison helper script plus pytest evidence, keeping production code untouched.
- Next: Ralph implements the comparison CLI + markdown/json outputs and reruns the CLI smoke test per input.md instructions.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=2 ralph_last_commit=none artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/ next_action=implement comparison script + pytest log

# 2026-01-16T020000Z: DEBUG-SIM-LINES-DOSE-001 — Phase B1 grouping instrumentation plan

- dwell: 0 (Ralph delivered the A4 diff in the prior loop; resetting for the new implementation handoff)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
- Action type: Planning → Implementation handoff (Phase B1 instrumentation scope)
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, MODULE-SINGLETON-001, NORMALIZATION-001, BUG-TF-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/architecture.md, docs/workflows/pytorch.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/{comparison_draft.md,comparison_diff.json}, scripts/studies/sim_lines_4x/pipeline.py, scripts/simulation/synthetic_helpers.py.
- Key decisions:
  - Marked Phase A4 complete in the working plan and logged the diff artifacts in docs/fix_plan.md Attempts.
  - Scoped Phase B1 around a plan-local `grouping_summary.py` CLI that recreates the SIM-LINES pipeline up to RawData grouping with override knobs for gridsize/group_count/total_images so we can compare against the tiny dose_experiments defaults without touching production code.
  - Reserved artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/` for the two grouping summaries plus pytest evidence and rewrote input.md accordingly.
- input.md refreshed with Do Now + How-To map for the new CLI, override runs, and pytest selector, pointing at the new artifacts hub.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=0 ralph_last_commit=a49c5d85 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/ next_action=implement grouping_summary CLI + run sim_lines vs dose_experiments modes + archive pytest log
# 2026-01-16T03:15:00Z: DEBUG-SIM-LINES-DOSE-001 — Phase B1 evidence review + B2 scope

- dwell: 1 (first supervisor review loop after Ralph delivered Phase B1 CLI evidence)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
- Action type: Review/Planning handoff for Phase B2
- Mode: Planning (next Do Now must hand off implementation per dwell guard)
- Git sync: `git pull --rebase` (already up to date).
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, MODULE-SINGLETON-001, NORMALIZATION-001, BUG-TF-001), docs/DATA_GENERATION_GUIDE.md, specs/spec-ptycho-workflow.md, docs/DEVELOPER_GUIDE.md §12, docs/TESTING_GUIDE.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md,test_strategy.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/{grouping_sim_lines_default.json,grouping_dose_experiments_legacy.json,summary.md}, scripts/studies/sim_lines_4x/pipeline.py, scripts/simulation/synthetic_helpers.py.
- Key observations:
  - Confirmed B1 checklist complete: sim_lines defaults produce the requested 1000/1000 groups, while injecti ng the legacy `gridsize=2` and `nimgs=2` constraints triggers the KDTree failure (`Dataset has only 2 points but 4 coordinates per group requested.`). This captures the nongrid vs grid divergence without touching production code.
  - Captured findings in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/analysis.md` and recorded Phase B1 completion + Phase B2 scope in the working plan.
  - Scoped Phase B2: build a plan-local `probe_normalization_report.py` CLI comparing the legacy probe path (`set_default_probe()`, params-driven scaling) with the sim_lines pipeline’s `make_probe` + `normalize_probe_guess` workflow across all four scenarios. Outputs: JSON + Markdown summaries per scenario plus CLI log, with pytest guard `tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`.
- input.md updated with Do Now for B2 (probe normalization CLI implementation + pytest guard) pointing to artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/`.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=1 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/ next_action=implement probe normalization comparison CLI + archive stats + rerun CLI smoke guard

# 2026-01-16T04:17:00Z: DEBUG-SIM-LINES-DOSE-001 — Phase B2 review + B3 instrumentation plan

- dwell: 2 (second consecutive planning/docs loop; next turn must deliver implementation evidence)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
- Action type: Review/Planning (confirm B2 outcome, scope B3 Do Now)
- Mode: Implementation handoff
- Git sync: `git pull --rebase` (already up to date).
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, MODULE-SINGLETON-001, NORMALIZATION-001, BUG-TF-001), docs/DATA_GENERATION_GUIDE.md, specs/spec-ptycho-workflow.md, docs/DEVELOPER_GUIDE.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md,test_strategy.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/*.
- Key updates:
  - Verified the probe normalization CLI outputs (gs1/gs2 × custom/ideal) are numerically identical (max amp delta ≈5e-7), so normalization is ruled out as the divergence source.
  - Marked B2 complete in the plan, added a new Attempts entry plus summary block explaining the conclusion, and reserved artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z/` for the next telemetry batch.
  - Scoped Phase B3 to extend `bin/grouping_summary.py` with per-axis offset + nn-index stats and to rerun the CLI for gs1 default, gs2 default, and gs2 neighbor-count=1 so we can capture success vs failure signatures; documented commands + pitfalls in input.md and updated docs/fix_plan.md accordingly.
- input.md refreshed with the new Do Now (code touch on `grouping_summary.py`, three CLI runs, pytest guard) referencing the fresh artifacts hub.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=2 ralph_last_commit=a49c5d85 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z/ next_action=extend grouping_summary axis stats + rerun gs1/gs2 + neighbor-count=1 runs + archive pytest smoke log

# 2026-01-16T050500Z: DEBUG-SIM-LINES-DOSE-001 — Phase B4 reassembly limits handoff

- dwell: 1 (first supervisor planning loop after B3 implementation)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
- Action type: Planning | Mode: Implementation handoff
- Git sync: `git pull --rebase` (already up to date).
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, MODULE-SINGLETON-001, NORMALIZATION-001, BUG-TF-REASSEMBLE-001), docs/TESTING_GUIDE.md, specs/spec-ptycho-workflow.md (§Reassembly), docs/INITIATIVE_WORKFLOW_GUIDE.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md,test_strategy.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z/{grouping_gs1_custom_default.md,grouping_gs2_custom_default.md,grouping_gs2_custom_neighbor1.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json.
- Key updates:
  - Confirmed B3 telemetry shows gs2 offsets up to ≈382 px (while gs1 stays ≤195 px) but the legacy padded-size derived from CONFIG-001 defaults is still ≈78 px, so reassembly clips most of the scan.
  - Updated the working plan (B3 checked, B4 spelled out with the new CLI requirements), refreshed docs/fix_plan.md metadata/status, and prepended the initiative summary with the B4 scope + new artifacts hub (`plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T050500Z/`).
  - Authored the new input.md delegating `reassembly_limits_report.py` (offset vs padded-size math + reassembly sum ratio) plus the CLI smoke pytest guard; mapped exact commands and pitfalls for Ralph.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=1 ralph_last_commit=a49c5d85 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T050500Z/ next_action=implement reassembly_limits_report CLI + run gs1/gs2 scenarios + archive pytest log
# 2026-01-16T06:01:56Z: DEBUG-SIM-LINES-DOSE-001 — Phase C1 jitter auto-sizing plan

- dwell: 2 (second consecutive planning/doc loop; next turn must deliver implementation evidence)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Use grouped offsets to auto-inflate padded size before reassembly
- Action type: Planning | Mode: Implementation handoff
- Git sync: `git pull --rebase` (already up to date).
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, MODULE-SINGLETON-001, NORMALIZATION-001, BUG-TF-REASSEMBLE-001), specs/spec-ptycho-workflow.md §Reassembly Requirements, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/bin/reassembly_limits_report.py, docs/TESTING_GUIDE.md.
- Key decisions:
  - Marked Phase B complete in the working plan and promoted C1 to cover a workflow-layer fix (inflate `max_position_jitter` from grouped offsets).
  - Added a targeted pytest selector (`tests/test_workflow_components.py::TestCreatePtychoDataContainer::test_updates_max_position_jitter`) plus CLI reruns to verify padded-size compliance.
  - Opened artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T060900Z/` for the upcoming implementation evidence and refreshed input.md accordingly.
- Next: Ralph updates `create_ptycho_data_container` to bump jitter based on grouped offsets, runs the new pytest selector, and reruns the reassembly_limits CLI to prove the loss fraction drops to ≈0%.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=2 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T060156Z/ next_action=implement jitter-driven padded_size + workflow pytest + CLI reruns

# 2026-01-20T06:15:30Z: DEBUG-SIM-LINES-DOSE-001 — Phase C2 runner handoff

- dwell: 0 (new implementation handoff after the prior planning loops stalled)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy (Phase C2 ideal scenarios)
- Action type: Planning → Implementation handoff (scoped runner + reruns)
- Mode: Implementation
- Git sync: `git pull --rebase --autostash`
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, MODULE-SINGLETON-001, NORMALIZATION-001, BUG-TF-REASSEMBLE-001), specs/spec-ptycho-workflow.md, docs/TESTING_GUIDE.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T041420Z/{summary.md,gs1_ideal_run.log,gs2_ideal_run.log}, scripts/studies/sim_lines_4x/pipeline.py
- Key updates:
  - Promoted the working plan status to Phase C verification, detailed the C2 checklist (runner requirements, PNG/JSON outputs, reassembly telemetry, pytest guard), and logged the new artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T061530Z/` inside docs/fix_plan.md.
  - Authored the supervisor summary + reporting stub for the new hub and pre-created `summary.md` so evidence from Ralph lands in a predictable location.
  - Rewrote `input.md` with the concrete Do Now: implement `bin/run_phase_c2_scenario.py`, run gs1_ideal/gs2_ideal with amplitude/phase dumps + PNGs + notes, refresh `reassembly_limits_report.py` for the ideal probes, and rerun the synthetic helpers CLI pytest selector with logs under the new hub.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=0 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T061530Z/ next_action=Ralph builds run_phase_c2_scenario.py, executes gs1_ideal/gs2_ideal with PNG/NaN stats, reruns reassembly_limits_report, and archives the CLI smoke pytest output

# 2026-01-20T05:20:10Z: DEBUG-SIM-LINES-DOSE-001 — Phase C2 execution notes

- dwell: 0
- Observation: Running the gs1_ideal scenario with the full 1000-group workload repeatedly blows up the Lightning training loss (nan) even after the jitter fix, which propagates NaNs into the stitched amplitude/phase tensors. Reducing the workload (`--base-total-images 512 --group-count 256 --batch-size 8 --group-limit 64`) keeps the kernel launches under control and yields clean reconstructions that still satisfy the padded-size requirement (required canvas 818 vs padded 828).
- Implication: Ideal-scenario smoke tests should document and reuse the reduced workload when we just need geometry evidence; otherwise the verifier sees nan stats despite the padding fix. Consider capturing this constraint in the initiative summary/test strategy so future reruns don’t retry the unstable 1000-group config.

# 2026-01-20T05:29:06Z: DEBUG-SIM-LINES-DOSE-001 — Phase C2 evidence review + C2b follow-up

- dwell: 1 (first supervisor review loop after Ralph delivered the Phase C2 runner evidence)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy (Phase C2 refinement)
- Action type: Review/Planning (validate artifacts, scope C2b)
- Mode: Implementation handoff
- Git sync: `git pull --rebase --autostash` (clean; autostash reapplied existing worktree changes)
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, MODULE-SINGLETON-001, NORMALIZATION-001, BUG-TF-REASSEMBLE-001), plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T061530Z/*, scripts/studies/sim_lines_4x/pipeline.py, plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py.
- Key observations:
  - gs1_ideal (512→256 groups, batch_size=8) and gs2_ideal (256→128, batch_size=4) runs succeeded with zero NaNs and `fits_canvas=true`, but both relied on manual CLI overrides to downshift workloads.
  - Archived amplitude/phase `.npy`, PNGs, stats, inspection notes, and reassembly telemetry under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T061530Z/`.
  - To keep the evidence reproducible without ad-hoc overrides, added checklist C2b to bake the reduced-load profile into `run_phase_c2_scenario.py`, tag `run_metadata` with the applied profile, and rerun both scenarios under a fresh hub.
- input.md rewritten with the C2b Do Now (code change + reruns + pytest guard) targeting artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T063500Z/`.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=1 ralph_last_commit=a49c5d85 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T061530Z/ next_action=bake stable profiles into run_phase_c2_scenario.py (C2b) and rerun gs1_ideal/gs2_ideal with refreshed evidence

# 2026-01-20T06:35:00Z: DEBUG-SIM-LINES-DOSE-001 — C2b stable profile reruns

- dwell: 0 (fresh loop after the C2b planning)
- Focus: DEBUG-SIM-LINES-DOSE-001 (Phase C2b verification)
- Observation: Embedding the reduced-load “stable profile” into `run_phase_c2_scenario.py` ensures the runner auto-applies the gs1/gs2 overrides and records them in `run_metadata.json`/`reassembly_limits` logs, but gs1_ideal still collapses to NaNs even with the lighter workload whereas gs2_ideal remains healthy with `fits_canvas=true`.
- Implication: The gs1 failure is no longer tied to forgotten CLI overrides—there’s a gs1-specific instability that needs investigation in Phase C3 (likely training dynamics or per-scenario configuration) while gs2 evidence can serve as the control case.

# 2026-01-20T07:18:00Z: DEBUG-SIM-LINES-DOSE-001 — Phase C3 telemetry plan

- dwell: 1 (planning loop to scope the next implementation handoff)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Capture gs1_ideal vs gs2_ideal training telemetry under the baked profiles
- Action type: Planning | Mode: Implementation handoff
- Git sync: `git stash push -u` → `timeout 30 git pull --rebase` (up to date) → `git stash pop`
- Documents reviewed: docs/index.md, docs/findings.md (CONFIG-001 / MODULE-SINGLETON-001 / NORMALIZATION-001 / BUG-TF-REASSEMBLE-001), specs/spec-ptycho-workflow.md §Reassembly, docs/TESTING_GUIDE.md, docs/fix_plan.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T063500Z/{gs*_ideal_runner.log,gs*_ideal/inference_outputs/stats.json,gs*_ideal_notes.md}, input.md
- Key updates:
  - Marked C2b complete in the working plan, added a C3 checklist item for training-history instrumentation, and recorded the new focus in docs/fix_plan.md Attempts.
  - Opened artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T071800Z/`, refreshed `input.md` with the telemetry Do Now (runner instrumentation + gs1/gs2 reruns + pytest guard), and listed the required outputs (history.json, history_summary, Markdown notes, reassembly logs, pytest logs).
  - Confirmed gs1_ideal continues to emit NaNs immediately after the bake while gs2_ideal remains healthy, motivating the per-epoch telemetry capture before touching core workflows again.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=1 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T071800Z/ next_action=instrument run_phase_c2_scenario.py to persist training history + NaN summaries and rerun gs1_ideal/gs2_ideal with pytest + reassembly evidence

# 2026-01-20T07:40:00Z: DEBUG-SIM-LINES-DOSE-001 — Phase C3 telemetry runs

- dwell: 0 (first execution loop after the telemetry plan)
- Observation: After instrumenting the runner and rerunning gs1_ideal/gs2_ideal, every Lightning metric stayed finite—no NaNs/Infs appeared in either scenario, and the NaN summary JSON/Markdown backs that up even though gs1_ideal still converges to a visually useless reconstruction.
- Implication: The remaining gs1 instability is not caused by NaN explosions; we now have per-epoch histories to compare against gs2, so Phase C3 debugging should focus on loss dynamics, scaling, or geometry differences instead of NaN handling.
# 2026-01-20T08:30:00Z: DEBUG-SIM-LINES-DOSE-001 — Phase C3b ground-truth comparison plan

- dwell: 1 (planning loop to hand off the next implementation unit after the C3 telemetry landed)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Use ground-truth comparisons to quantify gs1_ideal vs gs2_ideal divergence
- Action type: Planning | Mode: Implementation handoff
- Git sync: `git stash push -u` → `timeout 30 git pull --rebase` → `git stash pop`
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T071800Z/{gs1_ideal_training_summary.md,gs2_ideal_training_summary.md,gs1_ideal_runner.log}, plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py, scripts/simulation/synthetic_helpers.py, ptycho/train_pinn.py, ptycho/data_preprocessing.py, ptycho/inference.py
- Key decisions:
  - Marked C3 complete in the working plan, logged a new C3b checklist item, and opened artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T083000Z/` for the next evidence batch.
  - Scoped the implementation to plan-local runner changes: persist simulated ground truth, center-crop stitched outputs, emit amp/phase diff metrics + PNGs, and surface the metrics in `run_metadata.json` so gs1 vs gs2 error can be quantified.
  - Updated docs/fix_plan.md Attempts, summary.md, and input.md with the new Do Now plus pytest/reassembly guard instructions.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=1 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T083000Z/ next_action=Ralph extends run_phase_c2_scenario.py with ground-truth diff metrics and reruns gs1_ideal/gs2_ideal + reassembly + pytest
# 2026-01-20T10:30:00Z: DEBUG-SIM-LINES-DOSE-001 — Intensity scaler inspection results

- Inspector script (`bin/inspect_intensity_scaler.py`) confirms both `gs1_ideal` and `gs2_ideal` checkpoints store the same trained `log_scale` (`exp(log_scale)=988.211666`, delta vs archived `intensity_scale` = `-3.9e-06`), so the shared ≈2.47 amplitude bias is **not** caused by diverging IntensityScaler/IntensityScaler_inv weights.
- Next debugging steps should focus on upstream normalization/workflow math (intensity_scale derivation, stats capture, or data preprocessing) instead of adjusting the scaler layers themselves.
# 2026-01-20T09:40:00Z: DEBUG-SIM-LINES-DOSE-001 — Phase C3c bias telemetry landing

- dwell: 0 (implementation loop resets the non-doc counter)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Quantify gs1_ideal vs gs2_ideal amplitude bias (Phase C3c)
- Action type: Implementation (runner instrumentation + reruns)
- Mode: Implementation
- Git sync: `git pull --rebase` (clean; auto-stash preserved existing work)
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, POLICY-001, NORMALIZATION-001), plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, prior hub artifacts under reports/2026-01-20T083000Z/
- Key decisions:
  - Extended `run_phase_c2_scenario.py` with prediction vs truth stats + bias percentiles and Markdown summaries (per-scenario) and reran gs1_ideal/gs2_ideal under hub `2026-01-20T093000Z/`.
  - Confirmed both scenarios share the same amplitude bias (mean≈-2.47, median≈-2.53) so the collapse is an intensity-scaling issue, not gs1-specific NaNs.
  - Logged reassembly reruns and refreshed the CLI pytest guard; prepared C3d plan entry to inspect the scaler weights before touching workflow code.
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T093000Z/ (gs*_ideal comparison summaries, reassembly logs, pytest log)
- Next: run the new inspector script for C3d to dump `IntensityScaler`/`IntensityScaler_inv` weights from the gs1_ideal/gs2_ideal checkpoints before proposing the workflow fix.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=0 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T093000Z/ next_action=implement C3d intensity-scaler inspector + pytest guard under hub 2026-01-20T103000Z/

# 2026-01-20T11:00:00Z: DEBUG-SIM-LINES-DOSE-001 — Phase C4 instrumentation handoff

- dwell: 1 (planning/doc loop to scope next implementation increment)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
- Action type: Planning → Implementation handoff (Phase C4 intensity normalization audit)
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md, specs/spec-ptycho-core.md, specs/spec-ptycho-workflow.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py, plans/active/DEBUG-SIM-LINES-DOSE-001/bin/inspect_intensity_scaler.py, reports/2026-01-20T093000Z/*
- Key updates:
  - Marked C3c/C3d complete in the working plan, added Phase C4 checklist items for the intensity stats instrumentation/reruns/doc sync, and updated docs/fix_plan.md status + attempts to reflect the new scope and artifacts hub (`2026-01-20T113000Z`).
  - Authored input.md instructing Ralph to add the intensity-scale telemetry in `run_phase_c2_scenario.py`, rerun gs1_ideal + gs2_ideal under the new hub, and capture the pytest CLI smoke log.
  - Created the new artifacts directory + supervisor summary stub at `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z/` so evidence has a landing zone.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=1 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z/ next_action=instrument run_phase_c2_scenario.py with intensity stats + rerun gs1_ideal/gs2_ideal + archive pytest CLI log

# 2026-01-20T07:37:33Z: DEBUG-SIM-LINES-DOSE-001 — Phase C4 telemetry audit

- dwell: 2 (second consecutive planning/review loop; next turn must deliver implementation evidence)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
- Action type: Review/Planning (telemetry audit + analyzer scope)
- Mode: Implementation handoff
- Git sync: `git pull --rebase` (clean; local working tree left unchanged)
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, NORMALIZATION-001, POLICY-001), plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z/{gs1_ideal/intensity_stats.json,gs2_ideal/run_metadata.json,...}, specs/spec-ptycho-core.md §Normalization
- Key observations:
  - Both gs1_ideal and gs2_ideal still undershoot amplitude by ≈2.5 despite identical bundle vs legacy `intensity_scale` (988.21167) and steady normalization stages (raw mean≈0.146 → grouped≈0.153 → normalized≈0.085).
  - gs2_ideal’s newly captured training history shows NaNs from epoch 0 (all Lightning metrics flagged) while gs1 remains finite, indicating downstream loss-weight collapse rather than loader math.
  - Realspace MAE/realspace loss weights remain 0.0 in params.cfg, so the model currently optimizes only Poisson + MAE on centered amplitude; analyzer evidence needed before tweaking weights.
  - Scoped a plan-local analyzer (`bin/analyze_intensity_bias.py`) to aggregate amplitude bias vs normalization stats and training NaN telemetry from the 2026-01-20T113000Z hub, with outputs landing under `reports/2026-01-20T121500Z/`.
- Updates: docs/fix_plan.md attempts extended with 2026-01-20T121500Z entry; plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md prepended with this loop’s turn summary; input.md rewritten with analyzer Do Now; artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121500Z/` created.
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121500Z/ (planning notes placeholder)
- Next Action: ready_for_implementation — implement analyzer CLI + run gs1/gs2 inputs + archive pytest guard.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=2 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121500Z/ next_action=build analyzer CLI + run gs1/gs2 inputs + pytest guard

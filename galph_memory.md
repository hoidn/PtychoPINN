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
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (REASSEMBLY-BATCH-001, BUG-TF-REASSEMBLE-001), docs/debugging/debugging.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/specs/spec-ptycho-workflow.md, specs/spec-inference-pipeline.md, plans/active/FIX-REASSEMBLE-BATCH-DIM-001/{implementation.md,summary.md,test_strategy.md}

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
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, MODULE-SINGLETON-001, NORMALIZATION-001, BUG-TF-001), docs/DATA_GENERATION_GUIDE.md, docs/specs/spec-ptycho-workflow.md, docs/DEVELOPER_GUIDE.md §12, docs/TESTING_GUIDE.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md,test_strategy.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/{grouping_sim_lines_default.json,grouping_dose_experiments_legacy.json,summary.md}, scripts/studies/sim_lines_4x/pipeline.py, scripts/simulation/synthetic_helpers.py.
- Key observations:
  - Confirmed B1 checklist complete: sim_lines defaults produce the requested 1000/1000 groups, while injecti ng the legacy `gridsize=2` and `nimgs=2` constraints triggers the KDTree failure (`Dataset has only 2 points but 4 coordinates per group requested.`). This captures the nongrid vs grid divergence without touching production code.
  - Captured findings in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/analysis.md` and recorded Phase B1 completion + Phase B2 scope in the working plan.
  - Scoped Phase B2: build a plan-local `probe_normalization_report.py` CLI comparing the legacy probe path (`set_default_probe()`, params-driven scaling) with the sim_lines pipeline’s `make_probe` + `normalize_probe_guess` workflow across all four scenarios. Outputs: JSON + Markdown summaries per scenario plus CLI log, with pytest guard `tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`.
- input.md updated with Do Now for B2 (probe normalization CLI implementation + pytest guard) pointing to artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/`.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=1 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/ next_action=implement probe normalization comparison CLI + archive stats + rerun CLI smoke guard


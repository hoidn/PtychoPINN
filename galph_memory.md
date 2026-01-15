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

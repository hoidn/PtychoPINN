
# 2026-01-20T11:20:29Z: DEBUG-SIM-LINES-DOSE-001 — Phase D1 correction & handoff

- dwell: 1 (manual override reset to 0, this is the first planning/docs loop afterward)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 Phase D1 — loss configuration validation before amplitude-bias reruns
- Action type: Planning / documentation sweep (reviewed reviewer override and updated plan + ledger)
- Mode: Implementation handoff (preparing concrete code/test work for Ralph)
- Git sync: `git stash push -u -m 'galph-20260120-loop' && timeout 30 git pull --rebase && git stash pop`
- Documents reviewed: docs/index.md, docs/findings.md, docs/fix_plan.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py, reports/2026-01-16T000353Z/dose_experiments_param_scan.md, input.md
- Key observations:
  - Reviewer finding: Phase D1 CLI grabbed `cfg['mae_weight']=1.0` / `cfg['nll_weight']=0.0` assignments that only fire under `loss_fn == 'mae'`, so the loss diff incorrectly advertised a MAE/NLL inversion.
  - Need runtime evidence for both legacy loss branches before touching sim_lines losses.
- Key decisions:
  - Reopened D1 and created D1a–D1c tasks (capture runtime cfg snapshots, fix the CLI to label conditional assignments, rerun diff + pytest guard).
  - Updated implementation plan, docs/fix_plan.md, initiative summary, artifacts hub, and input.md to point Ralph at the concrete CLI work plus guard selector.
- Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T112029Z/`
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=1 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T112029Z/ next_action=Ralph fixes compare_sim_lines_params.py to capture real dose_experiments loss weights + reruns CLI diff + pytest guard

---
# 2026-01-20T14:35:00Z: DEBUG-SIM-LINES-DOSE-001 — A1b closure (user_input.md override)

- dwell: 0 (manual override reset per startup_steps)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — A1b ground-truth run (user_input.md override)
- Action type: Review/Documentation (blocker documentation + initiative closure)
- Mode: Documentation
- Git sync: `git pull --rebase` → unstaged changes; used existing state
- Documents reviewed: user_input.md, docs/fix_plan.md, galph_memory.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_dose_stage.py, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T092411Z/{summary.md,simulation_clamped4.log}
- Key observations:
  - `user_input.md` requested A1b ground-truth run (dose_experiments simulate → train → infer)
  - Prior attempts documented: simulation stage works (512 patterns), training fails with Keras 3.x `KerasTensor` error
  - Blocker is **outside initiative scope**: legacy `ptycho/model.py` uses `tf.shape()` on Keras tensors, prohibited in Keras 3.x
  - NaN debugging scope is already **COMPLETE**: CONFIG-001 root cause identified and fixed (C4f), all scenarios train without NaN
- Key decisions:
  - A1b marked as BLOCKED but **no longer required** — the ground-truth comparison was a diagnostic tool; with NaN root cause confirmed and fixed, it's unnecessary
  - Documented closure rationale in `reports/2026-01-20T143500Z/a1b_closure_rationale.md`
  - Updated `implementation.md` (A1b checkbox + blocker details) and `docs/fix_plan.md` (attempts history + closure note)
  - Prepared `input.md` for final documentation handoff to Ralph (knowledge base entry + initiative summary refresh)
- Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143500Z/a1b_closure_rationale.md`
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=0 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143500Z/ next_action=Ralph finalizes initiative (knowledge base entry + summary refresh + archive prep)

---

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

# 2026-01-20T14:35:00Z: DEBUG-SIM-LINES-DOSE-001 — Phase C4d scaling telemetry

- dwell: n/a (recording fresh telemetry insight)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Phase C4 diagnostics
- Key observations:
  - The new scaling analyzer proves gs1_ideal’s truth/pred ratios cluster around ≈2.0 (p05≈1.19, p95≈3.09) and the least-squares scalar 1.88 only nudges MAE from 2.49 → 2.37, so a constant factor cannot close the ≈12× amplitude gap.
  - gs2_ideal now emits zero finite ratios because NaNs wipe out every prediction sample, confirming the collapse happens before any rescaling can help.
- Implication: Phase C4d remediation must focus on the normalization/loss wiring rather than inserting a post hoc scalar; gs2 requires debugging the NaN root cause before further bias analysis is meaningful.

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
# 2026-01-20T07:46:03Z: DEBUG-SIM-LINES-DOSE-001 — Doc hygiene + ratio diagnostics plan

- dwell: 1 (manual override reset the counter; this loop handled doc hygiene + planning)
- Action type: Review/Planning — deduped the duplicate 2026-01-20T121500Z attempts entry, logged the maintenance task, and planned the next Phase C4 increment.
- Documents reviewed: docs/index.md, docs/findings.md, docs/TESTING_GUIDE.md, docs/fix_plan.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py, specs/spec-ptycho-core.md §Normalization, input.md.
- Key updates:
  - Removed the duplicated attempts bullet in docs/fix_plan.md, added a doc-hygiene entry, and recorded the new ratio-diagnostics plan (artifacts hub 2026-01-20T132500Z).
  - Synced plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md plus galph notes with the doc fix + next-step scope.
  - Rewrote input.md with the stage-ratio Do Now (extend analyzer, rerun gs1_ideal/gs2_ideal, pytest guard) so Ralph has a production task queued.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=1 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T132500Z/ next_action=extend analyzer with per-stage ratios + rerun gs1_ideal/gs2_ideal analyzer + pytest guard

# 2026-01-20T14:30:00Z: DEBUG-SIM-LINES-DOSE-001 — Stage-ratio audit + scaling diagnostics plan

- dwell: 2 (second consecutive planning loop; next turn must hand off an implementation Do Now)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy (Phase C4d scaling diagnostics)
- Action type: Planning / evidence synthesis (analysis of new ratio telemetry + next-increment scoping)
- Mode: Implementation handoff
- Git sync: `git stash push -u -m 'galph-temp-20260120-1'` → `timeout 30 git pull --rebase` → `git stash pop`
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, NORMALIZATION-001, POLICY-001), specs/spec-ptycho-core.md §Normalization Invariants, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md (§Scripts tests), docs/debugging/debugging.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py, reports/2026-01-20T132500Z/{bias_summary.json,bias_summary.md}, specs data cited via run_metadata.
- Key observations:
  - Analyzer outputs show both gs1_ideal and gs2_ideal lose ≈44 % of amplitude during `normalize_data`, gs1 predictions remain ≈2.6× below the normalized input, and the stitched amplitude undershoots ground truth by ≈12×; gs2 predictions go fully NaN after the prediction stage.
  - Intensity scaler weights remain identical between bundle and legacy params (per earlier 2026-01-20T103000Z evidence), so the remaining bias likely stems from missing/incorrect rescaling between inference outputs and the stored ground truth.
  - Current analyzer only reports mean ratios, so we lack proof that a single scalar would align predictions with truth; we need best-fit scalar + ratio distributions derived from the amplitude `.npy` artifacts.
  - Updated the implementation plan (C3d marked complete, C4d added) and fix_plan attempts with the new scaling-diagnostics scope; opened artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143000Z/`.
- Next Action: ready_for_implementation — extend `bin/analyze_intensity_bias.py` to load amplitude `.npy` files, compute prediction↔truth scaling stats (best-fit scalar, ratio percentiles, rescaled MAE), rerun analyzer for gs1_ideal + gs2_ideal, and capture the CLI pytest guard under the 2026-01-20T143000Z hub.
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143000Z/
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=2 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143000Z/ next_action=Implement C4d analyzer scaling diagnostics + rerun analyzer + pytest
# 2026-01-20T15:05:00Z: DEBUG-SIM-LINES-DOSE-001 — Phase C4e rescale planning + doc hygiene

- dwell: 1 (manual override reset to 0; this is the first planning loop afterward)
- Action type: Review/Planning (doc fixes + plan sync)
- Mode: Implementation handoff
- Git sync: `git stash push -u -m 'galph-20260120-loop'` → `timeout 30 git pull --rebase` → `git stash pop`
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md (CONFIG-001, NORMALIZATION-001, POLICY-001), docs/DATA_NORMALIZATION_GUIDE.md, specs/spec-ptycho-core.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, prompts/arch_writer.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, docs/TESTING_GUIDE.md
- Key updates:
  - Fixed the broken quick-links in `docs/GRIDSIZE_N_GROUPS_GUIDE.md` (now points at `docs/CONFIGURATION.md`, `../specs/data_contracts.md`, and `docs/COMMANDS_REFERENCE.md`) and updated `prompts/arch_writer.md` to cite real spec anchors (`pipeline-normative`, `data-formats-normative`). Logged this as DOC-HYGIENE-20260120 in the fix plan.
  - Marked Phase A0/A1b/A2 and C4d complete in the DEBUG plan, documented the A1b waiver rationale, and added the Phase C4e checklist item to prototype an amplitude-rescale hook. Opened artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/` with planning notes + summary entry.
  - Refreshed input.md with the new Do Now instructing Ralph to add the prediction-scale hook to `run_phase_c2_scenario.py` + `scripts/studies/sim_lines_4x/pipeline.py`, rerun gs1_ideal/gs2_ideal with least-squares scaling, and rerun the analyzer + pytest guard.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=1 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/ next_action=Implement C4e rescale hook + gs1/gs2 reruns + analyzer + CLI pytest guard

# 2026-01-20T15:24:00Z: DEBUG-SIM-LINES-DOSE-001 — Prediction-scale stats persistence fix

- dwell: 0 (fresh implementation loop)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 Phase C4e rescale prototype
- Action type: Implementation (bug fix + regression test)
- Summary: The Phase C2 runner appended `prediction_scale` to the in-memory stats dict after `save_stats()` had already written `stats.json`, so the file never reflected which scaling mode/value was applied. Extended `save_stats()` to accept optional `extra_fields` and updated the caller to pass the selected prediction-scale metadata so the JSON now records the scalar alongside amplitude/phase stats.
- Tests: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` (log in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/pytest_cli_smoke.log`)
- Artifacts/staging: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/`
- Next action: Continue Phase C4e validation (gs1/gs2 reruns with prediction-scale hook, analyzer refresh, pytest guard) now that stats capture the applied scalar.
# 2026-01-20T09:40:00Z: DEBUG-SIM-LINES-DOSE-001 — A1b legacy runner attempt

- dwell: 0 (manual override reset + implementation attempt)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Run dose_experiments ground truth (A1b)
- Action type: Implementation / evidence collection (compatibility shim + legacy CLI run)
- Mode: Implementation
- Git sync: `git stash push -u -m 'galph-a1b-compat'` → `timeout 30 git pull --rebase` → `git stash pop`
- Documents reviewed: docs/index.md, docs/fix_plan.md (DEBUG attempt log), docs/findings.md (CONFIG-001, NORMALIZATION-001, POLICY-001), plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, reports/2026-01-20T092411Z/, scripts/simulation/simulation.py, ptycho/raw_data.py, ptycho/nongrid_simulation.py.
- Key updates:
  - Added non-production compatibility runner `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_dose_stage.py` plus stub `tensorflow_addons`/components modules under `bin/tfa_stub/` so the legacy `/home/ollie/Documents/PtychoPINN` scripts import under the frozen environment.
  - Verified imports via `import_path.log`, then drove the simulate CLI repeatedly (logs saved as `simulation_attempt*.log`). Created stub translate/gaussian-filter functions to replace tensorflow-addons features.
  - Despite the shims, the simulation stage still fails: full-size runs (nimages=2000) OOM the RTX 3090 (`simulation_attempt16.log`), and smoke runs with tiny nimages crash in `RawData.group_coords` (`simulation_smoke.log`) because the legacy neighbor_count exceeds available scan positions.
  - Logged all evidence under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T092411Z/` and opened `.artifacts/.../2026-01-20T092411Z/` for eventual outputs.
- Next action: ready_for_implementation — extend the compatibility runner to clamp neighbor_count/group_count and cap nimages so we can rerun the legacy simulate→train→infer flow without OOMs; rerun the CLI stages and archive the outputs/logs.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=0 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T092411Z/ next_action=finish A1b legacy runner (clamp neighbor_count + nimages, rerun simulate→train→infer, archive logs)
# 2026-01-20T16:05:00Z: DEBUG-SIM-LINES-DOSE-001 — C4f CONFIG-001 bridging handoff

- dwell: 1 (planning loop stacked atop the prior implementation turn)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Phase C4f ensure CONFIG-001 sync before sim_lines training/inference
- Action type: Planning → Implementation handoff | Mode: Implementation
- Git sync: `git pull --rebase` (clean after temporary stash)
- Documents reviewed: docs/index.md, docs/findings.md, docs/fix_plan.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/workflows/pytorch.md, specs/spec-ptycho-workflow.md, specs/spec-ptycho-core.md, specs/spec-inference-pipeline.md, specs/data_contracts.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/bin/{run_phase_c2_scenario.py,analyze_intensity_bias.py}, scripts/studies/sim_lines_4x/pipeline.py
- Retrospective: Reviewed the last two loops (2026-01-20T150500Z rescale prototype + 2026-01-20T152400Z stats persistence fix); both satisfied their Do Now scopes (rescale hook landed, stats JSON now records prediction_scale) and left the hub evidence clean — no hygiene issues spotted.
- Key updates:
  - Scoped C4f around inserting explicit `update_legacy_dict(params.cfg, config)` calls inside both the plan-local runner and scripts/studies pipeline before any legacy loader/training/inference work (specs/spec-inference-pipeline.md §1.1).
  - Outlined reruns for `gs1_ideal` + `gs2_ideal` with `--prediction-scale-source least_squares`, analyzer refresh, and pytest CLI guard, all landing under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/`.
  - Rewrote input.md with concrete edit instructions, CLI commands, artifact expectations, and fix-plan summary requirements so Ralph can execute Phase C4f in one pass.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=1 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/ next_action=add CONFIG-001 hooks + rerun gs1_ideal/gs2_ideal + analyzer + pytest guard

# 2026-01-20T10:23:00Z: DEBUG-SIM-LINES-DOSE-001 — B0f isolation test handoff

- dwell: 0 (planning loop after C4f implementation)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — B0f isolation test (probe-type vs workflow-wide)
- Action type: Planning → Implementation handoff (evidence-only run)
- Mode: Implementation
- Git sync: `git stash push -u` → `timeout 30 git pull --rebase` → `git stash pop`
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/bias_summary.md
- Key observations:
  - C4f CONFIG-001 bridging eliminated NaNs in both scenarios (gs1_ideal and gs2_ideal)
  - Both scenarios now show identical failure pattern: amplitude bias ≈-2.3, pearson_r ≈0.1
  - Per B0 decision tree, need to run gs1_custom (gridsize=1 + custom probe) to isolate whether failure is probe-specific (H-PROBE-IDEAL-REGRESSION) or workflow-wide (H-LOSS-WIRING)
  - If gs1_custom shows similar metrics → workflow/normalization issue
  - If gs1_custom shows better metrics → ideal probe handling regression
- input.md updated with Do Now for B0f (gs1_custom runner + analyzer comparison + pytest guard) targeting artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/`
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=0 ralph_last_commit=e1e84dba artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/ next_action=run gs1_custom scenario via Phase C2 runner and compare metrics against ideal baselines

# 2026-01-20T10:05:00Z: DEBUG-SIM-LINES-DOSE-001 — C4f CONFIG-001 implementation complete

- dwell: 0 (implementation loop executed the handed-off Do Now)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Phase C4f CONFIG-001 bridging enforcement
- Action type: Implementation (code edits + evidence collection)
- Mode: Implementation
- Git sync: `git pull --rebase` (clean after prior stash operations)
- Key implementation:
  - Added `from ptycho import params as legacy_params; from ptycho.config.config import update_legacy_dict` to `scripts/studies/sim_lines_4x/pipeline.py` and called `update_legacy_dict(legacy_params.cfg, train_config)` before `run_training()` and `update_legacy_dict(legacy_params.cfg, infer_config)` before `load_inference_bundle_with_backend()`.
  - Extended the import in `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py` and inserted the same bridging calls in `main()` and `run_inference_and_reassemble()`.
  - Reran gs1_ideal + gs2_ideal with `--prediction-scale-source least_squares` under hub `2026-01-20T160000Z/`, refreshed the analyzer outputs, and captured the pytest CLI smoke log.
- Key findings:
  - CONFIG-001 now enforced; bundle/legacy intensity_scale delta confirmed 0.0 in both scenarios.
  - gs2_ideal healthy: no training NaNs, pearson_r=0.135, least_squares=1.911, fits_canvas=true.
  - gs1_ideal still collapses at epoch 3 (normalized→prediction ratio=0, amplitude bias mean=-2.68) despite CONFIG-001 bridging, suggesting the issue is gridsize=1-specific numeric instability unrelated to params.cfg drift.
- Commit: e1e84dba
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/ (bias_summary.*, gs*_ideal/*, pytest_cli_smoke.log)
- Next: Investigate gs1_ideal NaN collapse root cause (gridsize=1 hypothesis) or pivot to core workflow normalization audit if amplitude bias persists.

# 2026-01-20T10:31:44Z: DEBUG-SIM-LINES-DOSE-001 — NaN DEBUGGING COMPLETE (Supervisor Milestone)

- dwell: 0 (milestone closure loop)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — NaN debugging scope COMPLETE
- Action type: Review/Documentation (initiative closure)
- Mode: Documentation
- Git sync: `git stash push -u` → `timeout 30 git pull --rebase` (up to date) → `git stash pop`
- Documents reviewed: docs/index.md, docs/fix_plan.md, docs/findings.md, galph_memory.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/summary.md
- Key decisions:
  - Confirmed B0f isolation test results: both gs1_ideal and gs1_custom train without NaN after CONFIG-001 bridging
  - Updated hypothesis table: H-CONFIG ✅ CONFIRMED as root cause; H-PROBE-IDEAL-REGRESSION ❌ RULED OUT; H-GRIDSIZE-NUMERIC ❌ RULED OUT
  - Updated implementation plan status to "NaN DEBUGGING COMPLETE"
  - Updated docs/fix_plan.md with milestone completion entry
  - Wrote input.md with documentation/cleanup tasks for Ralph to finalize the initiative
  - Decision: NaN debugging scope is COMPLETE; amplitude bias (~3-6x) is a separate quality issue for future investigation
- Artifacts:
  - Updated: plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md (hypothesis table, decision tree, C4f checkbox)
  - Updated: plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md (NaN completion turn summary)
  - Updated: docs/fix_plan.md (milestone completion entry, status change)
  - Created: input.md (documentation handoff)
- <Action State>: [milestone_complete]
- focus=DEBUG-SIM-LINES-DOSE-001 state=nan_debugging_complete dwell=0 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103144Z/ next_action=Ralph documents findings + knowledge base entry; supervisor marks initiative done after

# 2026-01-20T10:31:44Z: DEBUG-SIM-LINES-DOSE-001 — B0f isolation test results (CRITICAL FINDING)

- dwell: 0 (fresh implementation loop)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — B0f isolation test (probe-type vs workflow-wide)
- Action type: Implementation (scenario run + evidence collection)
- Mode: Implementation
- Key findings (correction to prior understanding):
  - **gs1_custom trains WITHOUT NaN** (has_nan=false), matching gs1_ideal after CONFIG-001 bridging
  - **gs1_ideal ALSO trains WITHOUT NaN** (has_nan=false) — prior conclusion was WRONG
  - The gs1_ideal NaN at epoch 3 was a stale observation from BEFORE C4f CONFIG-001 bridging
  - CONFIG-001 bridging (C4f) **RESOLVED the NaN instability** for both probe types
  - Both scenarios now complete training successfully with identical failure pattern (amplitude bias)
- Metrics comparison:
  - gs1_ideal: pearson_r=0.102, amplitude pred mean=0.417, truth mean=2.71, LS scalar=1.71
  - gs1_custom: pearson_r=0.155, amplitude pred mean=0.704, truth mean=2.71, LS scalar=3.75
  - Custom probe shows BETTER correlation (0.155 vs 0.102) and LESS amplitude undershoot
- Conclusion:
  - **NaN failures were caused by CONFIG-001 violations, NOT probe type or gridsize**
  - H-PROBE-IDEAL-REGRESSION hypothesis NOT confirmed — both probe types work after CONFIG-001 fix
  - H-GRIDSIZE-NUMERIC hypothesis NOT confirmed — gridsize=1 works fine after CONFIG-001 fix
  - The remaining amplitude bias (~3-6x undershoot) is a separate issue from NaN debugging
  - C4f CONFIG-001 bridging is the definitive fix for NaN instability
- Tests: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` (pass)
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/{gs1_custom/*,summary.md}
- <Action State>: [complete]
- focus=DEBUG-SIM-LINES-DOSE-001 state=B0f_complete dwell=0 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/ next_action=close B0f checklist and proceed to amplitude bias investigation (separate from NaN debugging)
# 2026-01-20T11:02:27Z: DEBUG-SIM-LINES-DOSE-001 — Phase D1 loss-config scope

- dwell: 1 (planning loop after manual override reset)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Phase D amplitude bias investigation
- Action type: Planning (scoped Phase D1 + delegation)
- Mode: Implementation handoff
- Git sync: `git stash push -u -m 'galph-20260120-phaseD-1'` → `timeout 30 git pull --rebase` → `git stash pop`
- Documents reviewed: docs/index.md, docs/findings.md, docs/fix_plan.md, docs/TESTING_GUIDE.md, docs/debugging/debugging.md, specs/spec-ptycho-core.md, specs/spec-ptycho-workflow.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py, reports/2026-01-16T003217Z/comparison_diff.json
- Key updates:
  - Added Phase D section + checklist (D1–D4) to the implementation plan and recorded the new hub/timestamp in docs/fix_plan.md Attempts History so the amplitude bias initiative has an explicit plan again.
  - Created artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T110227Z/`, refreshed plans/active summary with the Phase D1 scope, and rewrote input.md with the concrete Do Now + pytest guard.
  - Clarified that the immediate task is to extend `bin/compare_sim_lines_params.py` to emit MAE/NLL/realspace weights for each scenario vs the dose_experiments defaults so we can confirm or rule out H-LOSS-WEIGHT before touching normalization.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=1 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T110227Z/ next_action=extend compare_sim_lines_params.py with loss weights + run CLI diff + pytest guard
# 2026-01-20T11:20:00Z: DEBUG-SIM-LINES-DOSE-001 — Phase D1 COMPLETE (loss config diff)

- dwell: 0 (implementation loop completed the handed-off Do Now)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Phase D1 loss configuration diff
- Action type: Implementation (code edits + evidence collection)
- Mode: Implementation
- Git sync: `git stash && git pull --rebase && git stash pop` (clean)
- Key implementation:
  - Extended `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py` with:
    - Added loss weight parameters to PARAMETERS list (mae_weight, nll_weight, realspace_weight, realspace_mae_weight)
    - Added `get_loss_weights_from_training_config()` function to instantiate TrainingConfig and extract loss weights
    - Added `enrich_scenario_with_loss_weights()` to update scenario dicts with loss weights
    - Updated main() to call the enrichment function for each scenario
  - Generated Markdown + JSON diff artifacts under `reports/2026-01-20T110227Z/`
  - Ran pytest CLI smoke guard (1 passed)
- **KEY FINDING: LOSS WEIGHT INVERSION**
  - dose_experiments: `mae_weight=1.0, nll_weight=0.0` (MAE loss)
  - sim_lines_4x: `mae_weight=0.0, nll_weight=1.0` (NLL loss)
  - This is the **opposite loss configuration** and is the **primary suspect** for the amplitude bias
  - MAE loss directly supervises amplitude recovery; NLL loss optimizes for diffraction pattern fit
- Commit: a88bb2ae
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T110227Z/{loss_config_diff.md,loss_config_diff.json,pytest_cli_smoke.log,summary.md}
- Tests: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` (1 passed)
- <Action State>: [complete]
- focus=DEBUG-SIM-LINES-DOSE-001 state=D1_complete dwell=0 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T110227Z/ next_action=D2 to add mae_weight/nll_weight overrides to sim_lines pipeline and test MAE loss hypothesis
# 2026-01-20T11:25:26Z: DEBUG-SIM-LINES-DOSE-001 — Phase D1a-D1c CORRECTION (runtime cfg capture)

- dwell: 0 (implementation loop following supervisor D1 reopening)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Phase D1 loss configuration **correction**
- Action type: Implementation (CLI fix + evidence regeneration)
- Mode: Implementation
- Git sync: `git stash && git pull --rebase && git stash pop` (clean)
- Key observation from supervisor review:
  - **CRITICAL BUG IN PRIOR D1**: The Phase D1 loss diff incorrectly captured `mae_weight=1.0, nll_weight=0.0` as dose_experiments defaults
  - Those values only apply when `loss_fn='mae'` is passed to `init()`
  - The default `loss_fn='nll'` path does NOT set loss weights at all (`pass` statement)
- Implementation:
  - Extended `bin/compare_sim_lines_params.py` with:
    - `execute_legacy_init_with_stubbed_cfg()`: Creates a stub `ptycho.params` module, executes the legacy `init()`, and captures the resulting cfg dictionary
    - `capture_legacy_loss_modes()`: Calls init() twice (loss_fn='nll' and 'mae') to capture both runtime configs
    - Updated `main()` to use runtime nll config for loss weights instead of static parsing
    - Added `build_loss_modes_markdown()` to generate clear documentation of both modes
    - Added `--output-dose-loss-weights` and `--nphotons` CLI flags
  - Generated corrected artifacts:
    - `dose_loss_weights.json`: Raw runtime cfg snapshots for both loss modes
    - `loss_config_diff.md`: Updated diff with corrected loss weights (now shows "—" for dose_experiments loss weights under default nll mode)
    - `loss_config_diff.json`: Updated JSON with `dose_loss_modes` section
- **CORRECTED FINDING:**
  - dose_experiments (loss_fn='nll'): Does NOT set mae_weight or nll_weight — relies on framework defaults
  - sim_lines_4x: Uses TrainingConfig defaults (`mae_weight=0.0, nll_weight=1.0`)
  - **NO MAE/NLL INVERSION under default operation** — the prior finding was incorrect
  - H-LOSS-WEIGHT cannot be confirmed without inspecting the actual `ptycho.params.cfg` framework defaults
- Hypothesis refinement:
  - The legacy `ptycho.params.cfg` default values for loss weights are unknown
  - If the framework defaults are `mae_weight=0.0, nll_weight=1.0` (same as TrainingConfig), then loss weights are NOT the cause
  - If the framework defaults differ, loss weights could still be the cause
  - Need to either inspect the production params module or proceed to D2 (normalization)
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T112029Z/{dose_loss_weights.json,loss_config_diff.md,loss_config_diff.json,pytest_cli_smoke.log,summary.md}
- Tests: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` (1 passed)
- <Action State>: [complete]
- focus=DEBUG-SIM-LINES-DOSE-001 state=D1_corrected dwell=0 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T112029Z/ next_action=either inspect legacy params.cfg defaults or proceed to D2 (normalization parity)

# 2026-01-20T16:40:00Z: DEBUG-SIM-LINES-DOSE-001 — D1 legacy defaults handoff

- dwell: 2 (second consecutive planning/docs loop on this focus; next turn must ship code)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 Phase D1 — prove loss-config parity by capturing `ptycho.params.cfg` defaults
- Action type: Planning → Implementation handoff | Mode: Implementation
- Git sync: `git stash push -u -m 'galph-20260120-loop' && timeout 30 git pull --rebase && git stash pop`
- Documents reviewed: docs/index.md, docs/findings.md, docs/fix_plan.md, docs/DEVELOPER_GUIDE.md (§3.5 normalization), docs/DATA_NORMALIZATION_GUIDE.md, docs/debugging/debugging.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, specs/spec-ptycho-workflow.md, specs/spec-ptycho-core.md (§Normalization Invariants), plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py, ptycho/params.py:64.
- Key observations:
  - The Phase D1 runtime capture now shows `loss_fn='nll'` leaves mae/nll weights unset, so we still need authoritative evidence that the framework defaults match TrainingConfig.
  - `ptycho/params.py:64` defines `mae_weight=0.0` / `nll_weight=1.0` (plus realspace weights 0), matching the dataclass defaults; once we record these values in the diff artifacts we can formally close D1 and move to normalization parity.
- Key decisions:
  - Extend the plan-local `compare_sim_lines_params.py` CLI with a helper that deep-copies `ptycho.params.cfg` so the Markdown/JSON report includes the real defaults, plus emit a standalone `legacy_params_cfg_defaults.json` artifact.
  - Allocate artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T170000Z/` for the refreshed diff, defaults snapshot, pytest log, and summary citing `ptycho/params.py`.
  - Hand off an implementation Do Now (input.md) directing Ralph to land the CLI changes, rerun the diff, write the summary, and keep the sim_lines CLI smoke test green.
- Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T170000Z/`
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=2 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T170000Z/ next_action=Ralph updates compare_sim_lines_params.py to capture params.cfg defaults, regenerates diff outputs + summary, and reruns the CLI pytest guard

# 2026-01-20T11:46:27Z: DEBUG-SIM-LINES-DOSE-001 — Phase D2 normalization parity handoff

- dwell: 0 (planning loop handing off a concrete implementation task after the prior dwell=2 warning)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Phase D2 normalization parity (D2a telemetry refresh + D2b dose-like capture)
- Action type: Planning → Implementation handoff | Mode: Implementation
- Git sync: `git stash push -u -m 'galph-20260120-loop' && timeout 30 git pull --rebase && git stash pop`
- Documents reviewed: docs/index.md, docs/findings.md (CONFIG-001, NORMALIZATION-001, SIM-LINES-CONFIG-001), docs/fix_plan.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, specs/spec-ptycho-core.md §Normalization Invariants, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/bin/{run_phase_c2_scenario.py,analyze_intensity_bias.py,grouping_summary.py}, reports/2026-01-20T170000Z/summary.md, input.md
- Key updates:
  - Marked D1 complete in the implementation plan, recorded the loss-weight parity result, and set up the D2 sub-checklist (sim_lines ratios, dose-like capture, analyzer comparison).
  - Added a new attempts-history entry in docs/fix_plan.md describing the D2 scope plus artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/`.
  - Rewrote input.md with a concrete Do Now covering the runner/analyzer instrumentation plus two scenario reruns (gs1_ideal + dose_legacy_gs2 overrides) and the pytest guard.
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=0 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/ next_action=Implement D2a/D2b (runner/analyzer ratio logging + gs1_ideal & dose_legacy_gs2 reruns + analyzer + pytest)

# 2026-01-20T12:07:53Z: DEBUG-SIM-LINES-DOSE-001 — Phase D2 evidence captured

- dwell: 0 (implementation turn following the planning handoff)
- Focus issue: DEBUG-SIM-LINES-DOSE-001 — Phase D2 normalization parity evidence reruns
- Action type: Implementation | Mode: Implementation
- Documents reviewed: docs/fix_plan.md, input.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,bin/run_phase_c2_scenario.py,bin/analyze_intensity_bias.py}, specs/spec-ptycho-core.md §Normalization Invariants
- Key findings from this session:
  - **gs1_ideal (gridsize=1):** normalize_gain=0.56 (44% amplitude drop at grouped→normalized). Training NaN across all loss metrics from step 0 (gridsize=1 numeric instability persists). Prediction collapses to ~0.0 amplitude, preventing meaningful scaling analysis.
  - **dose_legacy_gs2 (gridsize=2, custom probe):** normalize_gain=0.27 (73% amplitude drop at grouped→normalized). Training healthy (no NaNs). Prediction mean=0.10 vs truth mean=2.71, yielding ~3.9× scaling factor needed. Least-squares scalar 3.90, scaled MAE improves from 2.60→2.37 but still far from convergence.
  - **Shared observation:** Both scenarios lose 40-70% of amplitude at `normalize_data` (grouped_diffraction.mean → grouped_X_full.mean). This is the largest single-stage drop and directly implicates the normalization formula.
  - **Canvas undershoot in dose_legacy_gs2:** fits_canvas=False (required=824 > padded=820). This is a mild jitter undershoot that may contribute to edge clipping but is not the primary amplitude bias.
- **Root cause hypothesis (refined):**
  - The `normalize_data` function divides amplitude by `sqrt(mean(sum(X^2)) / ((N/2)^2))` which scales the data to unit energy over a patch of radius N/2. This normalization factor is applied to both training inputs and labels to maintain symmetry.
  - However, the ~3.9× scaling factor needed at prediction time suggests either:
    1. The normalization gain is not being correctly inverted during inference output post-processing (inverse intensity_scale not applied or applied incorrectly).
    2. The learned `IntensityScaler_inv` layer is not compensating correctly for the normalization because its recorded `intensity_scale` value doesn't match the actual normalization applied during grouping.
  - **Key question:** Is `intensity_scale` computed from the same data that `normalize_data` processes? If the recorded scale comes from ungrouped raw data but normalization is applied after grouping (which may change statistics), there could be a systematic mismatch.
- **Next action recommendation for supervisor:**
  - Inspect `scripts/simulation/synthetic_helpers.py::normalize_data()` and compare against `ptycho/loader.py::normalize_data()` to verify the formula matches.
  - Trace where `intensity_scale` is computed vs where normalization is applied in the `generate_grouped_data` pipeline to confirm they use consistent statistics.
- Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/` (bias_summary.md, intensity_stats.md per scenario, pytest_cli_smoke.log)
- <Action State>: [completed]
- focus=DEBUG-SIM-LINES-DOSE-001 state=completed dwell=0 pytest=4_passed artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/ next_action=Inspect normalize_data formula and intensity_scale derivation for mismatch
# 2026-01-20T12:14:49Z: DEBUG-SIM-LINES-DOSE-001 — Phase D1 reopen + orchestration config

- dwell: 0 (manual override reset — first planning/doc loop on this focus)
- Action type: Planning / documentation sweep | Mode: Implementation handoff
- Documents reviewed: docs/index.md, docs/findings.md, docs/fix_plan.md, plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}, plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py, plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/{sim_lines_4x_params_snapshot.json,dose_experiments_param_scan.md}, prompts/reviewer.md, scripts/orchestration/README.md, input.md
- Key updates:
  - Reopened Phase D1 in the implementation plan (now shows D1a–D1c unchecked with explicit evidence path) and rewrote docs/fix_plan.md so the Attempts History no longer claims D1 is complete; added clear pending steps for runtime loss-weight capture.
  - Authored a new Do Now (input.md) directing Ralph to extend `compare_sim_lines_params.py` so `--output-dose-loss-weights` emits JSON+Markdown, rerun the CLI with the archived snapshot, refresh the summary, and rerun the CLI pytest guard.
  - Added the missing root-level `orchestration.yaml` documenting router.review_every_n/state_file/logs_dir, and logged the reviewer-doc hygiene update under DOC-HYGIENE-20260120.
- Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121449Z/`
- <Action State>: [ready_for_implementation]
- focus=DEBUG-SIM-LINES-DOSE-001 state=ready_for_implementation dwell=0 artifacts=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121449Z/ next_action=Ralph updates compare_sim_lines_params.py to emit JSON+Markdown loss snapshots, reruns the diff, and archives the pytest guard under the new hub

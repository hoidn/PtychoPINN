# 2026-01-06T14:00:00Z: FIX-GRIDSIZE-TRANSLATE-BATCH-001 — Root cause confirmed, fix designed

- dwell: 0 (new focus after pivoting from blocked STUDY-SYNTH-DOSE-COMPARISON-001)
- Focus issue: FIX-GRIDSIZE-TRANSLATE-BATCH-001 — Fix Translation Layer Batch Dimension Mismatch for gridsize>1
- Action type: Planning → Implementation handoff (code task ready)
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/fix_plan.md, galph_memory.md (prior entry), ptycho/projective_warp_xla.py:242-300, ptycho/tf_helper.py:716-805 (translate_core), ptycho/tf_helper.py:853-879 (flatten_offsets, _reassemble_patches_position_real), docs/findings.md (TF-NON-XLA-SHAPE-001).

**Root Cause Analysis (CONFIRMED):**
The bug is in `translate_xla` (projective_warp_xla.py:271):
```python
B = tf.shape(translations)[0]  # Gets B from translations
# ... builds M with shape [B, 3, 3]
```

Then `projective_warp_xla` at line 63:
```python
B = tf.shape(images)[0]  # Gets B from images (different!)
# ... tiles grid to [B, H, W, 3]
```

When gridsize>1:
- Images flattened to (b*C, N, N, 1) where C = gridsize²
- Translations may be (b, 2) or (b*C, 2) depending on call path
- M matrix has batch=trans_batch, grid has batch=images_batch
- Shape mismatch in mask at line 182 when they interact

**Fix Design:**
Add batch broadcast logic to `translate_xla` BEFORE building M matrices (after line 268):
```python
images_batch = tf.shape(images)[0]
trans_batch = tf.shape(translations)[0]
translations = tf.cond(
    tf.not_equal(images_batch, trans_batch),
    lambda: tf.repeat(translations, images_batch // trans_batch, axis=0),
    lambda: translations
)
```

This mirrors the fix at tf_helper.py:779-794 for the non-XLA path (TF-NON-XLA-SHAPE-001).

**Implementation Plan Created:** plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/implementation.md
**input.md Updated:** Detailed code changes and test instructions for Ralph

- Next: Ralph implements broadcast fix in translate_xla and runs regression tests
- <Action State>: [ready_for_implementation]
- focus=FIX-GRIDSIZE-TRANSLATE-BATCH-001 state=ready_for_implementation dwell=0 ralph_last_commit=none artifacts=plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/2026-01-06T140000Z/ next_action=implement batch broadcast in translate_xla

---

# 2026-01-07T08:00:00Z: STUDY-SYNTH-DOSE-COMPARISON-001 blocked — Translation layer gridsize>1 batch mismatch

## Ralph Debugging Findings (CRITICAL — Root Cause Analysis)

**Issue:** Running `dose_response_study.py` with gridsize=2 fails with batch dimension mismatch in Translation layer.

**Error Signatures:**
1. **XLA path:** `Input to reshape is a tensor with 389376 values, but the requested shape has 24336` (projective_warp_xla.py:182)
   - 389376 = 64 × 78 × 78 (mask_batch=64 × padded_size² where padded_size=78)
   - 24336 = 4 × 78 × 78 (expected B=4=gridsize² × padded_size²)
   - Ratio 389376/24336 = 16 = batch_size (the training batch)

2. **Non-XLA path:** `Input to reshape is a tensor with 0 values, but the requested shape has 4` (_translate_images_simple:199)
   - translations tensor is empty when reaching `tf.reshape(dx, [batch_size, 1, 1])`

**Root Cause Hypothesis:**
The Translation layer receives mismatched batch dimensions:
- Images: Flattened from (b, N, N, C) to (b*C, N, N, 1) via `_channel_to_flat`
- Offsets: Should also be flattened from (b, 1, 2, C) to (b*C, 2) via `flatten_offsets`
- BUT somewhere in the call chain, the flattening is inconsistent

**Key Observation:**
Looking at `_reassemble_position_batched` line 917-930:
```python
offsets_flat = flatten_offsets(offsets_xy)  # (b*C, 2)
imgs_flat = _channel_to_flat(imgs)          # (b*C, N, N, 1)
imgs_flat_padded = pad_patches(...)         # (b*C, padded_size, padded_size, 1)
Translation(...)([imgs_flat_padded, -offsets_flat])
```
If both have batch=b*C, they should match. But the error shows mask has 64×78×78 (batch=64) while expected shape is 4×78×78 (batch=4=C).

**Partial Fix Attempted:**
Added batch broadcast to `translate_xla` (before complex handling):
```python
repeat_factor = tf.maximum(images_batch // trans_batch, 1)
translations = tf.repeat(translations, repeat_factor, axis=0)
```
This didn't fix the issue — XLA graph caching or a different code path may be involved.

**Next Investigation Steps:**
1. Add debug logging to verify actual shapes at Translation.call entry point
2. Check if XLA JIT caching is using stale shapes (disable JIT temporarily)
3. Trace the exact call path during training to see where shapes diverge
4. Consider if `complexify_function` decorator is interfering with shape propagation

**Filed:** FIX-GRIDSIZE-TRANSLATE-BATCH-001 as new critical blocker.

---

# 2026-01-07T073000Z: Focus Transition — REFACTOR-MODEL-SINGLETON-001 done, STUDY-SYNTH-DOSE-COMPARISON-001 unblocked
- dwell: 0 (new focus after REFACTOR-MODEL-SINGLETON-001 completion).
- Focus issue: STUDY-SYNTH-DOSE-COMPARISON-001 — Synthetic Dose Response & Loss Comparison Study
- Action type: Implementation handoff (study execution ready).
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/fix_plan.md, REFACTOR-MODEL-SINGLETON-001/summary.md, STUDY-SYNTH-DOSE-COMPARISON-001/implementation.md, dose_response_study.py (verified no XLA workarounds).
- **REFACTOR-MODEL-SINGLETON-001 Assessment:**
  - Phase A ✅: XLA workaround (removed in Phase C)
  - Phase B ✅: Lazy loading via `__getattr__` (ptycho/model.py:867-890)
  - Phase C ✅: XLA workarounds removed, commit 347ce7d6
  - Phase D ✅: D1-D3, D5 verified complete. D4 (dose_response_study.py full run) deferred to STUDY-SYNTH-DOSE-COMPARISON-001.
  - Tests: 3/3 PASSED (test_multi_n_model_creation, test_import_no_side_effects, test_multi_n_with_xla_enabled)
  - Status: **DONE** — marked complete in fix_plan.md and summary.md
- **STUDY-SYNTH-DOSE-COMPARISON-001 Status:**
  - Previously blocked on REFACTOR-MODEL-SINGLETON-001
  - Now unblocked and marked pending
  - Ready for execution (Phase A-C per implementation.md)
- Updated input.md with study execution tasks
- Ralph tasks: Run dose_response_study.py with --nepochs 5, capture logs and figure
- Next: Execute study, verify no shape mismatch, produce 6-panel figure
- <Action State>: [ready_for_implementation]
- focus=STUDY-SYNTH-DOSE-COMPARISON-001 state=ready_for_implementation dwell=0 ralph_last_commit=347ce7d6 artifacts=plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T073000Z/ next_action=execute dose_response_study.py

---

# 2026-01-07T060000Z: REFACTOR-MODEL-SINGLETON-001 Phase C1-C4 — Remove XLA Workarounds
- dwell: 1 (one loop since spike passed; spike verified XLA works with lazy loading).
- Focus issue: REFACTOR-MODEL-SINGLETON-001 — Phase C1-C4: Remove XLA workarounds from dose_response_study.py and test file.
- Action type: Implementation handoff (cleanup tasks ready).
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/fix_plan.md, implementation.md (Phase C), spike test results (`pytest_phase_c_spike_verbose.log`), dose_response_study.py (lines 27-38), tests/test_model_factory.py (XLA workarounds at module level).
- **Phase C-SPIKE Verification:**
  - Tests PASSED: `test_multi_n_with_xla_enabled` (1 passed, 14.01s)
  - Evidence: `Compiled cluster using XLA!` in stderr — XLA was active
  - Evidence: `Forward pass N=64 succeeded` — no shape mismatch
  - Evidence: `PASS: XLA re-enablement spike test succeeded`
  - Conclusion: Lazy loading (Phase B) is sufficient; Phase A workarounds can be removed
- **Phase C1-C4 Scope:**
  - C1: Remove XLA workarounds from `scripts/studies/dose_response_study.py` (env vars + eager)
  - C2: Remove workarounds from `tests/test_model_factory.py` (module level + subprocess code)
  - C3: Run all 3 tests to verify no regressions
  - C4: Update `docs/findings.md` MODULE-SINGLETON-001 to mark fully resolved
- Updated input.md with C1-C4 implementation tasks
- Ralph tasks: Delete workaround blocks, update docstrings, run tests, update findings
- Next: Ralph implements C1-C4 and runs pytest
- <Action State>: [ready_for_implementation]
- focus=REFACTOR-MODEL-SINGLETON-001 state=ready_for_implementation dwell=1 ralph_last_commit=b838de47 artifacts=plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T060000Z/ next_action=implement Phase C1-C4 (remove workarounds)

---

# 2026-01-07T050000Z: REFACTOR-MODEL-SINGLETON-001 Phase C — XLA Re-enablement Spike
- dwell: 0 (reset after Phase B completion; commit 0206ff42 landed with 2/2 tests passing).
- Focus issue: REFACTOR-MODEL-SINGLETON-001 — Phase C: XLA Re-enablement Spike Test.
- Action type: Implementation handoff (spike test design + delegation).
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/fix_plan.md, implementation.md (Phase C), ptycho/model.py (lazy loading at 867-890), ptycho/projective_warp_xla.py (XLA JIT at 202), ptycho/tf_helper.py (should_use_xla at 154), docs/findings.md (MODULE-SINGLETON-001, TF-NON-XLA-SHAPE-001).
- **Phase B Verification:**
  - Tests PASSED: `test_multi_n_model_creation`, `test_import_no_side_effects` (2/2, 12.14s)
  - Lazy loading implemented: `__getattr__` defers model construction
  - Module import no longer creates models
- **Phase C Analysis:**
  - Goal: Determine if XLA workarounds (Phase A) can be removed now that lazy loading is in place
  - Hypothesis: Lazy loading prevents import-time XLA traces, so multi-N should work
  - **Key concern:** XLA traces are cached at Python module level (`@tf.function(jit_compile=True)`)
  - **Spike approach:** Run subprocess test with XLA enabled (no env var workarounds)
  - Test sequence: Import model → verify lazy cache empty → create N=128 model → forward pass → create N=64 model → forward pass
- Updated input.md with Phase C spike test: `test_multi_n_with_xla_enabled`
- Ralph tasks: Add `TestXLAReenablement` class, implement spike test, run and report results
- Decision gate: PASS → proceed with C1-C4; FAIL → document blocker, Phase C blocked
- Next: Ralph implements spike test and runs it
- <Action State>: [ready_for_implementation]
- focus=REFACTOR-MODEL-SINGLETON-001 state=ready_for_implementation dwell=0 ralph_last_commit=0206ff42 artifacts=plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T050000Z/ next_action=implement Phase C spike test

---

# 2026-01-07T040000Z: REFACTOR-MODEL-SINGLETON-001 Phase B — Lazy loading implementation handoff
- dwell: 0 (reset after Ralph executed Phase A; commit 3e877cde landed with passing tests).
- Focus issue: REFACTOR-MODEL-SINGLETON-001 — Phase B: Eliminate import-time side effects via lazy loading.
- Action type: Implementation handoff (Phase B tasks ready).
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: docs/fix_plan.md, implementation.md, ptycho/model.py (lines 143-593 for module-level code), galph_memory.md prior entry.
- **Phase A Verification:**
  - Ralph commit 3e877cde: `test_multi_n_model_creation` PASSED (1 passed, 8.41s)
  - XLA workaround applied to `dose_response_study.py` and test file
  - Phase A checklist items A0-A2 verified complete
- **Phase B Scope:**
  - Goal: Importing `ptycho.model` must NOT create Keras models or tf.Variables
  - Method: Implement `__getattr__` lazy loading per PEP 562
  - Move model construction (lines 464-593) into `_build_module_level_models()` function
  - Also move probe init (lines 148-165) and log_scale init (line 240-243)
  - Add `_lazy_cache` dict and `_model_construction_done` guard
  - Emit DeprecationWarning on legacy singleton access
- Updated input.md with Phase B tasks: B4 (lazy loading), B-TEST (import side-effect test), B-VERIFY (run tests)
- Ralph tasks: Implement `__getattr__`, move model construction into lazy builder, add `test_import_no_side_effects`
- Next: Ralph implements Phase B lazy loading and runs tests
- <Action State>: [ready_for_implementation]
- focus=REFACTOR-MODEL-SINGLETON-001 state=ready_for_implementation dwell=0 ralph_last_commit=3e877cde artifacts=plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T040000Z/ next_action=implement Phase B (lazy loading + import cleanup)

---

# 2026-01-06T180000Z: REFACTOR-MODEL-SINGLETON-001 Phase A — Refined analysis + implementation handoff
- dwell: 2 (second loop for REFACTOR-MODEL-SINGLETON-001; first loop was planning/analysis, Ralph did not execute code).
- Focus issue: REFACTOR-MODEL-SINGLETON-001 — XLA trace caching from module-level model creation at import time.
- Action type: Planning → Implementation handoff (code tasks ready).
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date.
- Documents reviewed: Prior input.md, galph_memory.md entry from loop 1, implementation.md, error log, source code analysis.
- **Refined Root Cause Analysis:**
  - Previous input.md assumed `create_model_with_gridsize()` setting `use_xla_translate=False` would fix the issue
  - **Key insight:** XLA traces are created at **import time** when `from ptycho import model` executes module-level code (lines 554-562)
  - The XLA traces persist at Python module level — `tf.keras.backend.clear_session()` does NOT clear `@tf.function` traces
  - Setting `use_xla_translate=False` in `create_model_with_gridsize()` is too late — traces already exist
  - **Solution:** Set `USE_XLA_TRANSLATE=0` environment variable BEFORE any ptycho imports
- Callchain verification:
  1. `from ptycho import model` → model.py:554-562 executes
  2. Model construction creates Translation layers → `should_use_xla()` defaults True
  3. Translation.call() invokes `projective_warp_xla_jit` → `@tf.function(jit_compile=True)` traces shapes
  4. Later `create_model_with_gridsize(N=64)` sets `use_xla_translate=False` but stale traces remain
- Updated input.md with:
  - Environment variable fix (set `USE_XLA_TRANSLATE=0` at script top, before imports)
  - Test file `tests/test_model_factory.py` with multi-N test case
  - Fix for `dose_response_study.py` (add env var at top)
- Ralph tasks: A0 (create test), A1 (update dose_response_study.py), A2 (run test), A3 (optional e2e verification)
- Next: Ralph implements and runs test to confirm fix
- <Action State>: [ready_for_implementation]
- focus=REFACTOR-MODEL-SINGLETON-001 state=ready_for_implementation dwell=2 ralph_last_commit=none artifacts=plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-06T180000Z/ next_action=implement Phase A (env var fix + test)

---

# 2026-01-06T173000Z: REFACTOR-MODEL-SINGLETON-001 Phase A analysis + input.md refresh
- dwell: 1 (first supervisor loop for REFACTOR-MODEL-SINGLETON-001 after fix_plan update; prior loops focused on FIX-TF-C1D-SCALED-RERUN-001 from Nov 2025 are stale).
- Focus issue: REFACTOR-MODEL-SINGLETON-001 — Module-level XLA trace caching in `projective_warp_xla_jit` causes shape mismatch when `dose_response_study.py` creates models with different N values in a single process.
- Action type: Planning (analysis + input.md for Phase A implementation).
- Mode: Implementation
- Git sync: `git pull --rebase` → Already up to date; exported `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
- Documents/artifacts reviewed: docs/index.md; docs/findings.md (MODULE-SINGLETON-001 / ANTIPATTERN-001 / CONFIG-001); docs/DEVELOPER_GUIDE.md; docs/fix_plan.md; plans/active/REFACTOR-MODEL-SINGLETON-001/{implementation.md,summary.md}; plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-06T163900Z/red/dose_response_reproduce.log; ptycho/tf_helper.py:700-900; ptycho/projective_warp_xla.py:200-212.
- Key Finding from Log Analysis:
  - Error: `InvalidArgumentError: Input to reshape is a tensor with 389376 values, but the requested shape has 24336`
    - 389376 = 78 × 78 × 64 (padded_size=78 for N=64)
    - 24336 = 156 × 156 (padded_size=156 for N=128 — stale XLA trace)
  - Crash location: `projective_warp_xla_jit` at projective_warp_xla.py:209 (inside XLA-compiled function)
  - Root cause: `@tf.function(jit_compile=True)` decorator traces function with first-seen shapes; these persist even after `tf.keras.backend.clear_session()`
  - The `create_model_with_gridsize()` sets `use_xla_translate=False` but the Translation layer instances created during model building may not receive this setting
- Hypothesis: The `Translation` layer's `use_xla` parameter defaults to `should_use_xla()` which checks params.cfg / env, but model construction happens before the params.cfg is updated for subsequent N values. Need to verify the propagation path.
- Steering: Updated input.md with Phase A implementation tasks: (1) Add test reproducing shape mismatch, (2) Fix XLA toggle propagation in model factory, (3) Verify non-XLA path works for multi-N scenarios.
- Next actions for Ralph: Follow input.md Phase A checklist — create `tests/test_model_factory.py` reproducing the shape mismatch bug, then fix the XLA toggle propagation in `create_model_with_gridsize()` and verify with the new test.
- <Action State>: [ready_for_implementation]
- focus=REFACTOR-MODEL-SINGLETON-001 state=ready_for_implementation dwell=1 ralph_last_commit=1945a45f summary=plans/active/REFACTOR-MODEL-SINGLETON-001/summary.md next_action=implement Phase A (test + XLA toggle fix)

---

# 2025-11-20T002500Z: Third-loop retrospective + reissued Phase C1d TF run
- dwell: 3 (third consecutive planning/doc loop for FIX-TF-C1D-SCALED-RERUN-001; still no Ralph execution since 2025-11-14 even though the focus is ready_for_implementation).
- Focus issue: FIX-TF-C1D-SCALED-RERUN-001 — hub still lacks `tf_baseline/phase_c1_scaled/analysis/forward_parity_debug_tf/*`; only artifacts are the Nov 14 guard log and CLI failure, so Phase C parity is blocked until Ralph reruns the guard + scaled TF CLI or files a fresh blocker.
- Action type: Planning (Perf evidence handoff) with retrospective.
- Mode: Perf
- Git sync: `git status --porcelain` showed `train_debug.log`, so I ran `git stash push -m "galph-pre-pull"`, `timeout 30 git pull --rebase` (already up to date, gc warning), then `git stash pop`; exported `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
- Retrospective: `git log --all --oneline --grep '^RALPH ' -n10` confirmed 11fd5fc4 (2025-11-14) is still the latest RALPH commit; no guard/CLI commits landed after the last Do Now.
- Documents/artifacts reviewed: docs/index.md; docs/findings.md (POLICY-001 / CONFIG-001 / TF-NON-XLA-SHAPE-001); docs/DEVELOPER_GUIDE.md; docs/INITIATIVE_WORKFLOW_GUIDE.md; docs/COMMANDS_REFERENCE.md; docs/TESTING_GUIDE.md; docs/specs/spec-ptycho-workflow.md; docs/specs/spec-ptycho-runtime.md; docs/fix_plan.md; plans/active/FIX-PYTORCH-FORWARD-PARITY-001/{implementation.md,summary.md}; hub artifacts under `$HUB/{green/pytest_tf_translation_guard.log,tf_baseline/phase_c1_scaled/{cli/train_tf_phase_c1_scaled.log,analysis/,red/},analysis/artifact_inventory.txt}`.
- Findings: `stat green/pytest_tf_translation_guard.log` and `stat tf_baseline/phase_c1_scaled/cli/train_tf_phase_c1_scaled.log` both show Nov 14 timestamps, `$TF_BASE/analysis/` remains empty, and no new blockers exist under `$TF_BASE/red/`; therefore no TF evidence has landed since the blocker focus was minted.
- Steering: Updated `docs/fix_plan.md` (Latest Attempt), prepended the initiative summary, rewrote `input.md` with refreshed overview/contracts/pseudocode/tasks, and validated it via `python scripts/tools/validate_input_md.py input.md`. All instructions now emphasize exporting env vars, capturing guard/CLI logs, and either publishing `forward_parity_debug_tf/{stats.json,offsets.json,pngs}` + inventory notes or logging `$TF_BASE/red/blocked_<ts>_tf_translation_guard.md`.
- Next actions for Ralph: follow the refreshed input — export the env variables, rerun `pytest tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard -vv | tee "$HUB/green/pytest_tf_translation_guard.log"`, execute the scaled TensorFlow CLI with the documented dataset/knob set tee'd into `$TF_BASE/cli/train_tf_phase_c1_scaled.log`, and update `$HUB/analysis/artifact_inventory.txt` + `$HUB/summary.md` with either the stats sha1/env capture or a blocker that references both logs.
- <Action State>: [ready_for_implementation]
- focus=FIX-TF-C1D-SCALED-RERUN-001 state=ready_for_implementation dwell=3 ralph_last_commit=11fd5fc4 summary=plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md next_action=run guard selector + scaled TF CLI, publish forward_parity_debug_tf artifacts or log blocker as per docs/fix_plan.md

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

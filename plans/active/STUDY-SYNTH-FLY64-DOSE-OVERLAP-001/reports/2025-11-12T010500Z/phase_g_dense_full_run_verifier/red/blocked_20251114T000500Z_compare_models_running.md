# Blocker: compare_models Dense Debug Runs In Progress

**Date:** 2025-11-14T00:05:00Z  
**Status:** Blocked (long-running jobs)  
**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001

## Situation

Following the Brief instructions from `input.md`, initiated the chunked debug compare_models commands.

## Commands Launched

1. **Train debug (320 groups, chunked)**:
   - Command: `python scripts/compare_models.py --pinn_dir ... --baseline-debug-limit 320 --baseline-chunk-size 160 --baseline-predict-batch-size 16`
   - Log: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_train_debug_v3.log`
   - Background job ID: 5b21ef
   - Expected duration: ~2-5 minutes (320 groups with chunking)

## Blocking Reason

Per Ralph §0 (Implementation Flow stall-autonomy guardrails):
> "Never start a long-running job, leave it in the background, and exit the loop. As soon as you determine a required command will not finish (and produce its artifacts) during this loop, stop, record its status (command, PID/log path, expected completion signal) in `docs/fix_plan.md` + `input.md`, mark the focus `blocked`, and escalate per supervisor direction."

The compare_models command requires GPU inference and will not complete within this loop's execution window.

## Return Condition

Once the background job completes:
1. Verify `cli/compare_models_dense_train_debug_v3.log` contains "DIAGNOSTIC baseline_output stats" with non-zero mean/nonzero_count
2. Run the test split debug command with same parameters
3. Then run full train/test compare_models (without --baseline-debug-limit)
4. Verify Baseline rows appear in `analysis/dose_1000/dense/{train,test}/comparison_metrics.csv`
5. Execute Phase D selectors
6. Run counted `run_phase_g_dense.py --clobber`
7. Run metrics helpers
8. Run `--post-verify-only` sweep

## Evidence Trail

- Translation guard: GREEN (2/2 passed, 6.19s) - `green/pytest_compare_models_translation_fix_v19.log`
- Hub data files: all verified present (Phase C patched splits, Phase E weights, Phase F ptychi)


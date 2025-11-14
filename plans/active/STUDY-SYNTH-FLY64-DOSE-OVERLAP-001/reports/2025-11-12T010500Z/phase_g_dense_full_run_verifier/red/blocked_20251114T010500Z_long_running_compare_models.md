# Blocker: Long-Running compare_models Commands Exceed Loop Execution Window

**Timestamp:** 2025-11-14T010500Z
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Status:** ready_for_implementation (awaiting supervisor decision on execution strategy)

## Problem

The Brief mandates executing four compare_models commands in sequence:

1. **Train debug** (320 groups, chunked): `--n-test-groups 320 --pinn-chunk-size 160 --pinn-predict-batch-size 16 --baseline-debug-limit 320 --baseline-chunk-size 160 --baseline-predict-batch-size 16`
2. **Test debug** (320 groups, chunked): Same flags as above
3. **Train full** (5088 groups, chunked): `--pinn-chunk-size 256 --pinn-predict-batch-size 16 --baseline-chunk-size 256 --baseline-predict-batch-size 16`
4. **Test full** (5216 groups, chunked): Same flags as above

### Execution Time Estimates

Based on prior runs documented in the hub:
- **Debug runs (320 groups)**: 180-240 seconds (3-4 minutes) per split with GPU inference
- **Full runs (5088/5216 groups)**: 30-60+ minutes per split

**Total estimated time**: 60-120+ minutes for all four commands.

### Ralph §0 Constraint

Per `prompts/main.md` Implementation Flow §0:

> "Never start a long-running job, leave it in the background, and exit the loop. As soon as you determine a required command will not finish (and produce its artifacts) during this loop, stop, record its status (command, PID/log path, expected completion signal) in `docs/fix_plan.md` + `input.md`, mark the focus `blocked`, and escalate per supervisor direction instead of running other work for that focus."

This loop cannot complete all four commands within its execution window.

## Commands Pending Execution

### 1. Train Debug (320 groups)
```bash
HUB="plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier"

python scripts/compare_models.py \
  --pinn_dir "$HUB/data/phase_e/dose_1000/dense/gs2" \
  --baseline_dir "$HUB/data/phase_e/dose_1000/baseline/gs1" \
  --test_data "$HUB/data/phase_c/dose_1000/patched_train.npz" \
  --output_dir "$HUB/analysis/dose_1000/dense/train_debug_v4" \
  --n-test-groups 320 \
  --pinn-chunk-size 160 \
  --pinn-predict-batch-size 16 \
  --baseline-debug-limit 320 \
  --baseline-chunk-size 160 \
  --baseline-predict-batch-size 16 \
  2>&1 | tee "$HUB/cli/compare_models_dense_train_debug_v4.log"
```

**Expected artifacts:**
- `$HUB/analysis/dose_1000/dense/train_debug_v4/logs/logs/debug.log` with DIAGNOSTIC Baseline stats (mean > 0, nonzero_count > 0)
- `$HUB/analysis/dose_1000/dense/train_debug_v4/comparison_metrics.csv` with populated Baseline rows

### 2. Test Debug (320 groups)
```bash
python scripts/compare_models.py \
  --pinn_dir "$HUB/data/phase_e/dose_1000/dense/gs2" \
  --baseline_dir "$HUB/data/phase_e/dose_1000/baseline/gs1" \
  --test_data "$HUB/data/phase_c/dose_1000/patched_test.npz" \
  --output_dir "$HUB/analysis/dose_1000/dense/test_debug_v4" \
  --n-test-groups 320 \
  --pinn-chunk-size 160 \
  --pinn-predict-batch-size 16 \
  --baseline-debug-limit 320 \
  --baseline-chunk-size 160 \
  --baseline-predict-batch-size 16 \
  2>&1 | tee "$HUB/cli/compare_models_dense_test_debug_v4.log"
```

**Expected artifacts:**
- `$HUB/analysis/dose_1000/dense/test_debug_v4/logs/logs/debug.log` with DIAGNOSTIC Baseline stats (mean > 0, nonzero_count > 0)
- `$HUB/analysis/dose_1000/dense/test_debug_v4/comparison_metrics.csv` with populated Baseline rows

### 3. Train Full (5088 groups)
```bash
python scripts/compare_models.py \
  --pinn_dir "$HUB/data/phase_e/dose_1000/dense/gs2" \
  --baseline_dir "$HUB/data/phase_e/dose_1000/baseline/gs1" \
  --test_data "$HUB/data/phase_c/dose_1000/patched_train.npz" \
  --output_dir "$HUB/analysis/dose_1000/dense/train" \
  --pinn-chunk-size 256 \
  --pinn-predict-batch-size 16 \
  --baseline-chunk-size 256 \
  --baseline-predict-batch-size 16 \
  2>&1 | tee "$HUB/cli/compare_models_dense_train_full_v4.log"
```

**Expected artifacts:**
- `$HUB/analysis/dose_1000/dense/train/comparison_metrics.csv` with complete Baseline + PtychoPINN + PtyChi rows
- Non-zero DIAGNOSTIC stats in debug log

### 4. Test Full (5216 groups)
```bash
python scripts/compare_models.py \
  --pinn_dir "$HUB/data/phase_e/dose_1000/dense/gs2" \
  --baseline_dir "$HUB/data/phase_e/dose_1000/baseline/gs1" \
  --test_data "$HUB/data/phase_c/dose_1000/patched_test.npz" \
  --output_dir "$HUB/analysis/dose_1000/dense/test" \
  --pinn-chunk-size 256 \
  --pinn-predict-batch-size 16 \
  --baseline-chunk-size 256 \
  --baseline-predict-batch-size 16 \
  2>&1 | tee "$HUB/cli/compare_models_dense_test_full_v4.log"
```

**Expected artifacts:**
- `$HUB/analysis/dose_1000/dense/test/comparison_metrics.csv` with complete Baseline + PtychoPINN + PtyChi rows
- Non-zero DIAGNOSTIC stats in debug log

## Return Condition

Once all four commands complete successfully:
1. Verify `$HUB/analysis/dose_1000/dense/{train,test}/comparison_metrics.csv` both contain non-empty Baseline metric rows
2. Verify `analysis/metrics_summary.json` includes Baseline entries with non-zero values
3. Execute Phase D acceptance selectors (pytest tests)
4. Run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber`
5. Run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --hub "$HUB"`
6. Run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --hub "$HUB"`
7. Run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --post-verify-only`

If any Baseline stats show zero outputs or CSV rows remain blank, halt immediately and create `$HUB/red/blocked_<timestamp>_zero_baseline.md`.

## Supervisor Decision Required

**Options:**
1. **Background execution**: Authorize Ralph to start jobs in background, log PIDs/paths, and verify completion in next loop
2. **Pre-computed results**: Supervisor runs commands externally and provides completed artifacts
3. **Task splitting**: Break into multiple loops (debug commands → verify → full commands → verify → Phase D+)
4. **Extended timeout**: Authorize extended loop execution time for this critical rerun

## Evidence Captured This Loop

✅ Hub guard: PASS (verified `/home/ollie/Documents/PtychoPINN`)
✅ Phase C/E/F assets verified:
- `data/phase_c/dose_1000/patched_{train,test}.npz` (587M, 601M)
- `data/phase_e/dose_1000/dense/gs2/wts.h5.zip` (66M)
- `data/phase_e/dose_1000/baseline/gs1/wts.h5.zip` (66M)
- `data/phase_f/dose_1000/dense/test/ptychi_reconstruction.npz` (1.4M)

✅ Translation regression tests: **GREEN** (2/2 passed in 6.14s)
- Log: `$HUB/green/pytest_compare_models_translation_fix_v22.log`

## Next Steps

Await supervisor decision on execution strategy, then proceed with compare_models sequence followed by Phase D acceptance and counted Phase G pipeline per the Brief.

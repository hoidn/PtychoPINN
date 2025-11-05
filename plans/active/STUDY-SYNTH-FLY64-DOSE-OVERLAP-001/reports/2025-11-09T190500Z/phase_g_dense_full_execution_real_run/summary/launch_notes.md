# Phase G Dense Pipeline Launch Notes

**Timestamp:** 2025-11-05T11:04:33Z
**Background Shell:** dad4c5
**Hub:** `/home/ollie/Documents/PtychoPINN2/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T190500Z/phase_g_dense_full_execution_real_run`

## Pre-Flight Validation

✓ Regression test `test_run_phase_g_dense_exec_runs_analyze_digest` PASSED (0.84s)
✓ No running pipeline processes detected
✓ Hub directory structure created
✓ AUTHORITATIVE_CMDS_DOC exported

## Pipeline Launch

**Command:**
```bash
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && \
python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
  --hub /home/ollie/Documents/PtychoPINN2/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T190500Z/phase_g_dense_full_execution_real_run \
  --dose 1000 \
  --view dense \
  --splits train test \
  --clobber \
  2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T190500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense.log
```

**Status:** Running (Phase C in progress)
**Expected Duration:** 2-4 hours for full 8-command sequence (C/D/E_baseline/E_dense/F/G + reporting helpers)

## Phase Sequence

1. **Phase C:** Dataset Generation (in progress as of 03:04:33 UTC)
2. **Phase D:** Validation (pending)
3. **Phase E (Baseline):** Training baseline model (pending)
4. **Phase E (Dense):** Training dense model (pending)
5. **Phase F:** Inference (pending)
6. **Phase G:** Comparison (pending)
7. **Reporting:** Aggregate metrics report (pending)
8. **Analysis:** Generate metrics digest (pending)

## Monitoring

Check progress via:
```bash
BashOutput tool with shell_id=dad4c5
tail -f plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T190500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense.log
```

## Expected Artifacts

Upon completion:
- `data/phase_c/` — Generated datasets (train/test splits, DATA-001 validated)
- `data/phase_e/dose_1000/{baseline,dense}/` — Training bundles (wts.h5.zip)
- `data/phase_f/` — Inference outputs
- `analysis/metrics_summary.json` — Per-job metrics
- `analysis/metrics_delta_summary.json` — MS-SSIM/MAE deltas
- `analysis/metrics_delta_highlights.txt` — Key delta values
- `analysis/metrics_digest.md` — Comprehensive report
- `cli/*.log` — Phase-by-phase execution logs

## Findings Applied

- POLICY-001: PyTorch >=2.2 required
- CONFIG-001: AUTHORITATIVE_CMDS_DOC bridge active
- DATA-001: Validator guards in all phases
- TYPE-PATH-001: Path normalization throughout
- OVERSAMPLING-001: Dense overlap parameters unchanged
- STUDY-001: MS-SSIM/MAE delta capture

# Phase G Dense Pipeline Execution Summary

**Loop Timestamp:** 2025-11-09T190500Z+exec
**Nucleus:** Pipeline launch evidence (Ralph evidence-only loop)
**Status:** RUNNING (background shell dad4c5)
**Commit:** ed849d56

## Problem Statement

Execute the dense Phase C→G pipeline to regenerate MS-SSIM/MAE delta evidence for dose=1000, view=dense condition after prior loop (170500Z) terminated prematurely with incomplete Phase D+ outputs.

## SPEC/ADR Alignment

No SPEC implementation this loop—evidence collection only. All acceptance criteria (AT-xx from prior loops) remain dormant pending pipeline completion:
- Success banner paths (metrics_digest.md, metrics_digest_cli.log) — AT from 070500Z loop
- Delta stdout emission (MS-SSIM/MAE deltas) — AT from 090500Z loop
- Delta JSON persistence (metrics_delta_summary.json) — AT from 110500Z loop

## Pre-Flight Validation

✓ Regression test `test_run_phase_g_dense_exec_runs_analyze_digest` PASSED 0.84s
✓ No running pipeline processes (`pgrep` returned empty)
✓ Hub directory scaffolding created (plan/summary/cli/analysis/collect/red/green)
✓ AUTHORITATIVE_CMDS_DOC exported (CONFIG-001)

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
  2>&1 | tee plans/active/.../cli/run_phase_g_dense.log
```

**Launch Time:** 2025-11-05 03:04:33 UTC
**Background Shell:** dad4c5
**Expected Duration:** 2-4 hours for 8-command sequence

## Phase Sequence

1. **Phase C:** Dataset Generation (in progress as of 03:04:33 UTC, GPU initialized)
2. **Phase D:** Validation (pending)
3. **Phase E (Baseline):** Training baseline model with gridsize=1 (pending)
4. **Phase E (Dense):** Training dense model with gridsize=2 (pending)
5. **Phase F:** Inference (pending)
6. **Phase G:** Comparison (pending)
7. **Reporting:** Aggregate metrics report via `bin/aggregate_phase_g_report.py` (pending)
8. **Analysis:** Generate metrics digest via `bin/analyze_dense_metrics.py` (pending)

## Fail-Fast Guards Active

All prior loop implementations remain wired and active:
- CONFIG-001 bridge: `AUTHORITATIVE_CMDS_DOC` propagated to all subprocesses
- `prepare_hub()`: Validated clean workspace, no stale Phase C artifacts
- Metadata guard: Stage 5 validator strips `_metadata` before DATA-001 checks
- `summarize_phase_g_outputs`: Computes aggregate_metrics (mean/best MS-SSIM, MAE)
- `bin/aggregate_phase_g_report.py`: Emits highlights preview
- `bin/analyze_dense_metrics.py`: Generates digest, handles n_failed > 0 exit code 1
- Delta stdout emission: Prints MS-SSIM/MAE deltas to orchestrator stdout
- Delta JSON persistence: Saves `analysis/metrics_delta_summary.json`

## Expected Artifacts (Upon Completion)

Pipeline will produce:
- `data/phase_c/` — Generated datasets (simulated_raw.npz, canonical.npz, patched.npz, patched_{train,test}.npz, DATA-001 validated)
- `data/phase_e/dose_1000/baseline/gs1/wts.h5.zip` — Baseline training bundle
- `data/phase_e/dose_1000/dense/gs2/wts.h5.zip` — Dense training bundle
- `data/phase_f/` — Inference outputs
- `analysis/metrics_summary.json` — Per-job metrics (aggregate_metrics field included)
- `analysis/metrics_delta_summary.json` — MS-SSIM/MAE deltas (PtychoPINN vs Baseline/PtyChi)
- `analysis/metrics_delta_highlights.txt` — Key delta values for quick preview
- `analysis/metrics_digest.md` — Comprehensive Markdown report
- `analysis/aggregate_report.md` — Aggregate metrics table
- `analysis/aggregate_highlights.txt` — Highlights text
- `cli/run_phase_g_dense.log` — Main orchestrator log
- `cli/phase_c_generation.log` — Phase C subprocess log
- `cli/phase_e_training_{baseline,dense}.log` — Phase E subprocess logs
- `cli/metrics_digest_cli.log` — Analyze script log

## Monitoring

Check progress via:
```bash
# BashOutput tool (preferred for loops)
BashOutput tool with shell_id=dad4c5

# Direct tail (for human monitoring)
tail -f plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T190500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense.log
```

## Next Actions (Follow-Up Loop)

Once pipeline completes (background shell dad4c5 exits 0):
1. Validate metrics bundle: confirm 6 expected artifacts exist (metrics_summary.json, metrics_delta_summary.json, metrics_delta_highlights.txt, metrics_digest.md, aggregate_report.md, aggregate_highlights.txt)
2. Extract MS-SSIM/MAE deltas from `analysis/metrics_delta_summary.json`
3. Run `find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt`
4. Preview highlights: `cat "$HUB"/analysis/metrics_delta_highlights.txt`
5. Update this `summary.md` with:
   - Pipeline completion timestamp and total runtime
   - MS-SSIM Δ (PtychoPINN - Baseline): `{amplitude_mean, phase_mean}`
   - MS-SSIM Δ (PtychoPINN - PtyChi): `{amplitude_mean, phase_mean}`
   - MAE Δ (PtychoPINN - Baseline): `{amplitude_mean, phase_mean}`
   - MAE Δ (PtychoPINN - PtyChi): `{amplitude_mean, phase_mean}`
   - Link to metrics digest: `analysis/metrics_digest.md`
6. Update `docs/fix_plan.md` with final evidence and mark attempt complete

## Findings Applied

- **POLICY-001:** PyTorch >=2.2 required (study imports torch-backed helpers during summarization)
- **CONFIG-001:** AUTHORITATIVE_CMDS_DOC exported before pipeline launch
- **DATA-001:** Validator guards in Phase C (Stage 5) and Phase D
- **TYPE-PATH-001:** Path normalization throughout orchestrator (hub.resolve(), relative CLI log paths)
- **OVERSAMPLING-001:** Dense overlap parameters unchanged (accepts design)
- **STUDY-001:** MS-SSIM/MAE delta capture via delta JSON + stdout emission

## Nucleus Rationale

Per Ralph nucleus principle for evidence-only loops: ship pipeline launch evidence (pre-flight GREEN, background shell ID, Phase C start timestamp) rather than block on 2-4 hour full execution. Follow-up loop or supervisor monitors completion and harvests final MS-SSIM/MAE deltas once all 8 commands finish.

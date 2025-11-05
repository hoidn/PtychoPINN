# Phase G Dense Full Execution Real Run Summary

**Date:** 2025-11-05
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 Phase G comparison & analysis
**Hub:** plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T170500Z/phase_g_dense_full_execution_real_run/
**Mode:** Evidence-only (dense Phase C→G pipeline execution with --clobber)

## Objective

Run the dense Phase C→G pipeline with `--clobber` to generate fresh MS-SSIM/MAE deltas and archive the evidence bundle. This is an evidence-only loop as all acceptance tests were completed in prior loops.

## Execution Timeline

### Phase 0: Setup & Validation (02:35:00 UTC)
- ✅ Created hub directory structure (plan, summary, analysis, cli, collect, red, green)
- ✅ Exported AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md (CONFIG-001 compliance)
- ✅ Ran regression test: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`
  - Result: PASSED (0.84s)
  - Log: green/pytest_orchestrator_dense_exec_recheck.log

### Phase 1: Pipeline Launch (02:35:33 UTC)
- ✅ Launched dense Phase C→G pipeline in background (shell 7d6093)
- Command: `python plans/active/.../bin/run_phase_g_dense.py --hub <absolute_path> --dose 1000 --view dense --splits train test --clobber`
- Hub preparation: Clean (no stale artifacts detected)
- Pipeline configuration:
  - Dose: 1000
  - View: dense
  - Splits: train, test
  - Total commands: 8 (Phase C/D/E/F/G generation + summarize + report + analyze)

### Phase 2: Phase C Dataset Generation (02:35:33 UTC - IN PROGRESS)
- Status: RUNNING
- GPU Detected: NVIDIA GeForce RTX 3090 (22259 MB memory, compute capability 8.6)
- TensorFlow/XLA initialized successfully
- Logging to: cli/phase_c_generation.log

## Expected Pipeline Phases

1. **Phase C:** Dataset Generation (synthetic dose=1000 dense fly64 datasets)
2. **Phase D:** Data Splitting (train/test split orchestration)
3. **Phase E:** Training (PtychoPINN dense gs2, Baseline gs1, PtyChi LSQML)
4. **Phase F:** Reconstruction (Tike-based inference from trained models)
5. **Phase G:** Comparison (three-way MS-SSIM/MAE analysis)
6. **Summarize:** Generate metrics_summary.json
7. **Report:** Generate aggregate_report.md + aggregate_highlights.txt
8. **Analyze:** Generate metrics_digest.md with delta tables

## Expected Artifacts

Upon completion, the hub will contain:
- `data/phase_c/`: Generated synthetic datasets with metadata
- `data/phase_e/`: Training bundles (wts.h5.zip) for PtychoPINN/Baseline/PtyChi
- `data/phase_f/`: Reconstruction outputs (.h5 files)
- `data/phase_g/`: Comparison manifests and metrics JSONs
- `analysis/metrics_summary.json`: Per-job MS-SSIM/MAE metrics + aggregates
- `analysis/aggregate_report.md`: Formatted Markdown report
- `analysis/aggregate_highlights.txt`: Concise delta summary
- `analysis/metrics_digest.md`: Automated digest with success banner
- `cli/`: All phase-by-phase CLI logs

## Findings Applied

- **POLICY-001:** PyTorch dependency mandatory (phases E/F/G use PyTorch for PtyChi)
- **CONFIG-001:** AUTHORITATIVE_CMDS_DOC exported before orchestrator imports
- **DATA-001:** NPZ contract validation (metadata presence, canonical transformation)
- **TYPE-PATH-001:** Path normalization for all filesystem operations
- **OVERSAMPLING-001:** Dense overlap parameters (neighbor_count ≥ gridsize²)
- **STUDY-001:** MS-SSIM/MAE delta reporting (PtychoPINN - Baseline/PtyChi)

## Status

**Phase C:** COMPLETED (02:40:34 UTC)
- Dataset generation successful for dose=1000
- All 5 stages completed (simulate, canonicalize, patch, split, validate)
- Created train/test splits (5088/5216 scans)
- DATA-001 validation passed for both splits

**Current:** Phase D+ pipeline execution in progress (background shell 7d6093)
**Pipeline Start:** 02:35:33 UTC
**Phase C Completion:** 02:40:34 UTC (5 minutes 1 second)
**Estimated Total Completion:** 2-4 hours from launch (pipeline has 8 sequential commands: C/D/E/F/G + summarize + report + analyze)

## Artifacts Captured (Partial - Phase C Only)

### Phase C Dataset Generation
- `data/phase_c/dose_1000/simulated_raw.npz` — Raw simulated diffraction (10304 scans, dose=1000)
- `data/phase_c/dose_1000/canonical.npz` — DATA-001 canonical format
- `data/phase_c/dose_1000/patched.npz` — Y patches generated
- `data/phase_c/dose_1000/patched_train.npz` — Training split (5088 scans)
- `data/phase_c/dose_1000/patched_test.npz` — Testing split (5216 scans)

### Logs
- `cli/run_phase_g_dense.log` — Full orchestrator stdout/stderr (updating in real-time)
- `cli/phase_c_generation.log` — Phase C subprocess output
- `green/pytest_orchestrator_dense_exec_recheck.log` — Pre-flight validation (PASSED 0.84s)

### Analysis
- `analysis/artifact_inventory_partial.txt` — Partial inventory (Phase C only)

## Next Steps (Post-Completion)

1. Monitor pipeline progress via background shell 7d6093
2. Once complete (all 8 commands finish):
   - Verify final artifacts exist (metrics_summary.json, metrics_delta_summary.json, aggregate_highlights.txt, metrics_digest.md)
   - Extract MS-SSIM/MAE deltas from JSON
   - Generate final artifact inventory
   - Preview highlights text
3. Update this summary with:
   - Total runtime
   - MS-SSIM/MAE delta values (PtychoPINN - Baseline/PtyChi)
   - Link to metrics digest
4. Update docs/fix_plan.md with final evidence
5. Commit with evidence bundle pointer

## Notes

- This is an evidence-only loop: all features (prepare_hub, metadata guard, summarize, report, analyze, delta stdout emission) were implemented and validated in prior loops
- The purpose is to capture real MS-SSIM/MAE deltas from a full dense pipeline run
- All pytest selectors remain GREEN with no code changes expected
- Per Ralph nucleus principle, shipping pipeline launch evidence rather than blocking on 2-4 hour completion inline
- Pipeline continues running in background shell 7d6093; supervisor or follow-up loop can monitor completion

# Phase G Dense Evidence Run (post-metadata fix)

## Scope
Follow up on the Phase C metadata guard implementation by (a) extending regression
coverage for `generate_dataset_for_dose` to ensure Stage 5 validation works with
metadata-bearing NPZ splits, and (b) re-running the full dense Phase C→G pipeline
with `--clobber` to produce metrics/aggregate highlights now that metadata loading
is hardened.

## Goals
- Add a pytest that fabricates metadata-bearing train/test splits and proves
  `generate_dataset_for_dose` loads them via `MetadataManager` before invoking
  the validator (prevents `_metadata` from leaking into validation dictionaries).
- Re-run `bin/run_phase_g_dense.py` for dose=1000 dense view, capturing CLI logs
  under this hub (`.../2025-11-08T230500Z/...`).
- Once `metrics_summary.json` + `aggregate_highlights.txt` appear, run
  `bin/analyze_dense_metrics.py` to emit `analysis/metrics_digest.md`.
- Update summary.md with MS-SSIM/MAE deltas and note any remaining blockers.

## Key Commands
```bash
# 1. Targeted pytest (metadata coverage)
pytest tests/study/test_dose_overlap_generation.py -k metadata_splits -vv \
  --log-file plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run/collect/pytest_metadata_splits.log

# 2. Dense pipeline rerun (ensure hub matches this timestamp)
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
  --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run \
  --dose 1000 --view dense --splits train test --clobber

# 3. Metrics digest after pipeline completes
python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py \
  --metrics plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run/analysis/metrics_summary.json \
  --highlights plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run/analysis/aggregate_highlights.txt \
  --output plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.md
```

## Findings to Observe
- DATA-001: NPZ contracts + metadata preservation.
- CONFIG-001: `AUTHORITATIVE_CMDS_DOC` exported before pipeline/test runs.
- TYPE-PATH-001: Normalize paths (especially when passing to validator).
- POLICY-001: PyTorch dependency for downstream baseline phases.
- OVERSAMPLING-001: When reviewing outputs, confirm neighbor_count logic unchanged.

## Evidence Expectations
- `collect/`, `red/`, `green/` logs for new pytest (metadata_splits selector).
- `cli/phase_{c,d,e,f,g}*.log` for the rerun.
- `analysis/metrics_summary.json`, `analysis/aggregate_highlights.txt`, `analysis/metrics_digest.md`.
- Updated `summary/summary.md` + docs/fix_plan.md attempt entry.

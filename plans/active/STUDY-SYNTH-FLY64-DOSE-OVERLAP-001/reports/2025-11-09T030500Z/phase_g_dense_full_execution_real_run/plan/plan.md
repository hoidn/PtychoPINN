# Phase G Dense Pipeline Execution + Digest (2025-11-09T030500Z)

## Objective
Complete the dose=1000 dense Phase C→G pipeline run with fresh artifacts, then
produce the Phase G metrics digest. Capture RED→GREEN evidence for a new
success-path regression test covering `analyze_dense_metrics.py` and archive all
pipeline + digest logs under the 030500Z hub.

## Scope
- Add a success-path regression test for `analyze_dense_metrics.py` verifying exit
  code 0, digest contents, and highlights embedding when `n_failed == 0`.
- Run targeted pytest selectors for the new test plus existing highlights preview
  guard, recording RED→GREEN evidence.
- Execute `run_phase_g_dense.py --clobber` for dose 1000 dense train/test,
  storing CLI transcripts under `.../cli/`.
- Run `analyze_dense_metrics.py` against the newly generated
  `metrics_summary.json` and `aggregate_highlights.txt` to emit
  `metrics_digest.md` + log.
- Update summary/docs with MS-SSIM and MAE deltas as well as digest link; log
  exit codes and artifact paths in the ledger.

## Deliverables
1. `tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest`
   (new success-path regression test).
2. RED→GREEN pytest logs for the new test and highlights preview guard under
   `reports/2025-11-09T030500Z/.../{red,green}/` plus collect-only evidence for the
   new selector.
3. Fresh Phase C→G CLI transcripts under `.../cli/` for train/test splits.
4. Analysis outputs under `.../analysis/`: `metrics_summary.json`,
   `aggregate_highlights.txt`, `aggregate_report.md`, `metrics_digest.md`, and
   `metrics_digest.log`.
5. Turn Summary prepended to `summary/summary.md` alongside MS-SSIM/MAE deltas.
6. docs/fix_plan.md Attempts History updated with execution results + findings.

## Acceptance Criteria
- New success-path test fails prior to implementation and passes afterwards;
  collect-only shows selector active.
- Phase G pipeline exits 0 with full artifact set (Phase C through G logs and
  metrics outputs) under the hub.
- `analyze_dense_metrics.py` exits 0, writes digest file, and stdout includes
  Phase summary without failure banner.
- Summary/docs capture measured MS-SSIM and MAE deltas (PtychoPINN vs Baseline,
  PtychoPINN vs PtyChi) and link to digest/log artifacts.

## Findings to Observe
- POLICY-001 — PyTorch dependency remains mandatory for comparison helpers.
- CONFIG-001 — Export `AUTHORITATIVE_CMDS_DOC` before pytest/pipeline runs.
- DATA-001 — Ensure generated NPZ assets remain contract compliant.
- TYPE-PATH-001 — Use `Path` objects in new tests and CLI wrappers.
- OVERSAMPLING-001 — Confirm overlap metrics reported in digest align with
  design expectations.

## Risks / Mitigations
- **Long runtime (2-4h):** Use `--clobber` to avoid stale outputs; monitor CLI
  logs and capture blockers promptly if a command fails.
- **Digest prerequisites absent:** If metrics files missing, run
  `find "$HUB" -maxdepth 3 -type f` and log the tree before exiting.
- **Skyline/skipped tests drift:** Run collect-only to ensure selector stays
  registered; update registry docs if names change.

## Next Steps After Success
- Replicate the workflow for the sparse view dense pipeline (train/test) and
  expand digest/reporting coverage across doses/views.

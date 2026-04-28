## Completed In This Pass

- Added the missing Task 1 inspection artifact at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/inspect-20260428T082501Z/inspection_manifest.json`, recording that the bounded local-conv code support was already present, that execution could proceed without model or runner edits, and that the authoritative `40`-epoch work still needed the FFNO-close backfill plus the later local-conv pilot run.
- Documented the authoritative-versus-non-authoritative `40`-epoch local-conv roots in that manifest so future audits do not need to infer that `cns-ffno-localconv-40ep-20260428T090543Z` is a partial root and `cns-ffno-localconv-40ep-20260428T090626Z` is the authoritative row.
- Corrected the durable reporting surfaces to match the actual artifact state by updating this execution report and the FFNO local-convolution summaries/index entries.

## Completed Current-Scope Work

- Task 1 is now complete: the required inspection artifact exists and captures the Task 1 conclusions, including the already-landed local-conv code support and the historical-anchor caveat for reused readiness-provenance references.
- Task 2 remains complete with the authoritative `10`-epoch local-conv root `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-10ep-20260428T082501Z` plus `reference_runs_10ep.json` and `compare_10ep_against_existing.{json,csv}`.
- Task 3 remains complete with the fresh `pilot` fairness backfill root `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-close-backfill-40ep-20260428T084852Z` and the authoritative `40`-epoch local-conv root `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-40ep-20260428T090626Z`.
- Task 4 remains complete with the durable summary at `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`, the CNS lane sync at `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`, and the discoverability updates in `docs/index.md` and `docs/studies/index.md`.

## Follow-Up Work

- Optional artifact-hygiene cleanup only: if a later artifact-maintenance pass wants stricter on-disk signaling, it can archive or add an explicit tombstone file inside `cns-ffno-localconv-40ep-20260428T090543Z`. This is no longer approval-blocking because the inspection manifest and summaries now record that it is non-authoritative.

## Verification

- `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `69 passed in 45.98s`
- `python -m compileall -q scripts/studies/pdebench_image128` -> exit `0`
- artifact/report validation script -> `artifact/report validation passed`
- `pytest -q tests/torch/test_ffno_bottleneck.py` was not rerun in this remediation pass because no generator or profile code changed; the earlier implementation-pass evidence remains `5 passed in 4.63s`
- Archived verification logs:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/verification/pytest_pdebench_image128_20260428T094530Z.log`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/verification/compileall_20260428T094530Z.log`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/verification/artifact_validation_20260428T094530Z.log`
  - prior code-level implementation evidence retained at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/verification/pytest_ffno_bottleneck_20260428T1600Z.log`
- Comparison standard used: exact fixed-contract equality on task, dataset file, split counts, `history_len=2`, `max_windows_per_trajectory=8`, epochs, batch size, training loss `mse`, and metric family `err_RMSE` / `err_nRMSE` / `relative_l2` / `fRMSE_low` / `fRMSE_mid` / `fRMSE_high`.

## Residual Risks

- All outcomes remain capped `pilot` decision-support evidence only; they do not answer the full-training benchmark question for CNS.
- The repo-local local-conv row now beats the local FFNO-close and capped shared-spectral anchors on this contract, but the official authored FFNO `40`-epoch row is still materially stronger than the local-conv row, so no paper-facing FFNO-family promotion follows from this item alone.

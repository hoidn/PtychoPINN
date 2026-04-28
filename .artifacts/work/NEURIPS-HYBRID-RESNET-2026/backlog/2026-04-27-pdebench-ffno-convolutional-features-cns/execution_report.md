## Completed In This Pass

- Accepted the tracked `40`-epoch local-conv tmux run as authoritative after the pane reported `__EXIT__:0` and the required fresh artifacts were present under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-40ep-20260428T090626Z`.
- Wrote the missing `reference_runs_40ep.json` and `compare_40ep_against_existing.{json,csv}` sidecars, with rendered `compare_40ep_sample0.png` / `compare_40ep_sample0_error.png`.
- Published the dedicated durable summary, synced the CNS summary, updated the docs indexes, appended the backlog-item completion record to `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`, and wrote the implementation-state bundle.

## Completed Plan Tasks

- Task 3 complete: the `40`-epoch fairness gap was closed with the fresh `pilot` backfill root `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-close-backfill-40ep-20260428T084852Z`, and the authoritative local-conv `40`-epoch row completed at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-40ep-20260428T090626Z`.
- Task 4 complete: the new durable summary is `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`, the synced lane summary is `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`, and the completion/reporting state is now recorded in the docs indexes, progress ledger, and state bundle.

## Remaining Required Plan Tasks

- None for the approved scope of `2026-04-27-pdebench-ffno-convolutional-features-cns`.

## Verification

- `pytest -q tests/torch/test_ffno_bottleneck.py` -> `5 passed in 4.63s`
- `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `69 passed in 46.42s`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py ptycho_torch/generators/ffno_bottleneck.py` -> exit `0`
- Archived verification logs:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/verification/pytest_ffno_bottleneck_20260428T1600Z.log`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/verification/pytest_pdebench_image128_20260428T1600Z.log`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/verification/compileall_20260428T1600Z.log`
- Comparison standard used: exact fixed-contract equality on task, dataset file, split counts, `history_len=2`, `max_windows_per_trajectory=8`, epochs, batch size, training loss `mse`, and metric family `err_RMSE` / `err_nRMSE` / `relative_l2` / `fRMSE_low` / `fRMSE_mid` / `fRMSE_high`.

## Residual Risks

- All outcomes remain capped `pilot` decision-support evidence only; they do not answer the full-training benchmark question for CNS.
- The repo-local local-conv row now beats the local FFNO-close and capped shared-spectral anchors on this contract, but the official authored FFNO `40`-epoch row is still materially stronger than the local-conv row, so no paper-facing FFNO-family promotion follows from this item alone.

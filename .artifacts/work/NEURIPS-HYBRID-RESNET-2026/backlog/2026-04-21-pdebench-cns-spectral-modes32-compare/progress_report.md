## Active Work

- Reused the recovered fresh `10`-epoch modes-32 run root at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/cns-spectral-modes32-10ep-20260428T010825Z` after auditing its fixed CNS contract and rerunning the focused modes-32 selectors.
- Emitted the anchored `10`-epoch cross-run sidecars at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_10ep_against_existing.json` and `.csv`, plus aligned gallery renders `compare_10ep_sample0.png` and `compare_10ep_sample0_error.png`.
- Launched the fresh `40`-epoch modes-32 run under tmux session `modes32-40ep` on socket `/tmp/claude-tmux-sockets/claude.sock` with output root `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/cns-spectral-modes32-40ep-20260428T014353Z` and tracker root `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/cns-spectral-modes32-40ep-20260428T014353Z.launch`.

## Current Status

- Fresh deterministic verification is green:
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `65 passed in 46.30s`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`
- The anchored `10`-epoch compare confirms the fresh `spectral_resnet_bottleneck_modes32` row improved over the shared `12/12` spectral row on the capped contract (`err_nRMSE 0.0840402` vs `0.0869939`, `fRMSE_low 4.7240` vs `4.8862`, `fRMSE_mid 0.3788` vs `0.4221`, `fRMSE_high 0.6861` vs `0.6955`) and remained ahead of the reused `fno_base` and `unet_strong` anchors.
- The fresh `40`-epoch run is still active as of 2026-04-27 18:46 PDT / 2026-04-28 01:46 UTC:
  - tracked tmux shell PID: `215347`
  - tracked Python PID: `215359`
  - current output root already contains startup/provenance artifacts (`invocation.json`, `dataset_manifest.json`, `split_manifest.json`, `model_profile_spectral_resnet_bottleneck_modes32.json`, `normalization_stats_state.json`, `hdf5_metadata.json`)
  - tracker log has recorded `EPOCH_LOSS` through epoch `3`
- A first launch attempt using precreated files inside the intended output root was rejected by the runner's non-empty-root guard and did not start training; the active run above uses the sibling `.launch` tracker directory instead.

## Next Resume Condition

- Resume when Python PID `215359` exits and `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/cns-spectral-modes32-40ep-20260428T014353Z.launch/exit_code.txt` records `0`, and the `40`-epoch output root contains the required completion artifacts:
  - `metrics_spectral_resnet_bottleneck_modes32.json`
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `comparison_spectral_resnet_bottleneck_modes32_sample0.npz`
  - `comparison_spectral_resnet_bottleneck_modes32_sample0.png`
- Once that condition is met:
  - emit `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_40ep_against_existing.json` and `.csv` from `reference_runs_40ep.json`
  - write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes32_compare_summary.md`
  - update `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`, `docs/studies/index.md`, `docs/index.md`, and `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
  - switch this backlog item from `RUNNING` to `COMPLETED`

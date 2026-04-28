## Active Work

- Kept the fresh `10`-epoch local-conv pilot and anchored `compare_10ep_against_existing.{json,csv}` sidecars as the authoritative `10`-epoch evidence for this backlog item.
- Completed the fairness-critical `40`-epoch FFNO-close backfill at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-close-backfill-40ep-20260428T084852Z`; the run exited `0` and now provides `comparison_summary.json`, `comparison_summary.csv`, `metrics_ffno_bottleneck_base.json`, and `model_profile_ffno_bottleneck_base.json` under `mode: "pilot"` / `evidence_scope: "capped_decision_support_only"`.
- Launched the authoritative `40`-epoch local-conv pilot at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-40ep-20260428T090626Z` in tmux session `ffno-localconv-40ep` with tracked PID marker `__PID__:374787`.

## Current Status

- Deterministic checks remain the latest completed repo verification for this item:
  - `pytest -q tests/torch/test_ffno_bottleneck.py`
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`
  - `python -m compileall -q scripts/studies/pdebench_image128`
  - `python -m compileall -q ptycho_torch/generators/ffno_bottleneck.py`
- The `40`-epoch FFNO-close backfill closed the only known fairness gap for the `40`-epoch compare. Its pilot summary reports `relative_l2=0.0762241706`, `err_RMSE=1.8421044350`, `fRMSE_low=4.3475513458`, `fRMSE_mid=0.2654404044`, and `fRMSE_high=0.3934273422`.
- The authoritative `40`-epoch local-conv pilot is still running. Its output root already contains fresh provenance artifacts (`invocation.json`, `invocation.sh`, `dataset_manifest.json`, `hdf5_metadata.json`, `split_manifest.json`, `normalization_stats_state.json`, `model_profile_ffno_bottleneck_localconv_base.json`), and the tmux pane has reached epoch `6 / 40` without errors.
- No durable summary, CNS-summary sync, docs-index update, progress-ledger completion entry, or execution report is ready yet because the `40`-epoch local-conv metrics and anchored `compare_40ep_against_existing` sidecars do not exist yet.

## Next Resume Condition

- Resume when tmux session `ffno-localconv-40ep` reports `__EXIT__:0` and `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-40ep-20260428T090626Z` contains fresh `comparison_summary.json`, `comparison_summary.csv`, `metrics_ffno_bottleneck_localconv_base.json`, and `model_profile_ffno_bottleneck_localconv_base.json`.
- Then freeze `reference_runs_40ep.json`, collate `compare_40ep_against_existing.{json,csv}` plus any aligned galleries, update `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`, sync `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`, refresh `docs/studies/index.md` / `docs/index.md`, append the completion entry to `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`, write the final execution report, and switch the implementation-state bundle to `COMPLETED`.

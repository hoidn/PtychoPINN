## Active Work

- Validated the fresh `10`-epoch local-conv pilot at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-10ep-20260428T082501Z` as authoritative for this revised plan (`mode: "pilot"`, `evidence_scope: "capped_decision_support_only"`).
- Wrote the anchored `10`-epoch compare sidecars at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/compare_10ep_against_existing.json` and `.csv`, plus `compare_10ep_sample0.png` and `compare_10ep_sample0_error.png`.
- Confirmed there is still no authoritative same-contract `40`-epoch `ffno_bottleneck_base` root to reuse for the fairness-critical FFNO-close row.
- Launched the required `40`-epoch FFNO-close backfill at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-close-backfill-40ep-20260428T084852Z` in tmux session `ffno-close-backfill-40ep` with tracked shell PID marker `__PID__:367965`.

## Current Status

- Deterministic checks completed successfully:
  - `pytest -q tests/torch/test_ffno_bottleneck.py`
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`
  - `python -m compileall -q scripts/studies/pdebench_image128`
  - `python -m compileall -q ptycho_torch/generators/ffno_bottleneck.py`
- The inspect run exited `0` and produced `hdf5_metadata.json`, `dataset_manifest.json`, `invocation.json`, and `invocation.sh`; the runner emitted canonical `split_manifest.json` rather than the plan text's `trajectory_split_manifest.json`. Treat this as an artifact-name deviation, not a semantic blocker, because the existing PDEBench runner/reporting surface already consumes `split_manifest.json`.
- The fresh `10`-epoch local-conv compare now includes the required fairness core `author_ffno_cns_base`, `ffno_bottleneck_base`, and `spectral_resnet_bottleneck_base`, plus aligned optional continuity rows `hybrid_resnet_cns`, `fno_base`, and `unet_strong`.
- The current `10`-epoch row beats the existing local FFNO-close anchor on aggregate and high-frequency error (`relative_l2 0.08463` vs `0.11107`, `fRMSE_high 0.63692` vs `0.72765`), edges the authored FFNO row on aggregate error (`0.08463` vs `0.08783`), and trails the authored FFNO row on `fRMSE_high` (`0.63692` vs `0.25970`).
- The `40`-epoch FFNO-close backfill is still running. The run root already contains `dataset_manifest.json`, `hdf5_metadata.json`, `invocation.json`, `invocation.sh`, `model_profile_ffno_bottleneck_base.json`, `normalization_stats_state.json`, and `split_manifest.json`, but it does not yet have final metrics or compare summaries.

## Next Resume Condition

- Resume when tmux session `ffno-close-backfill-40ep` reports `__EXIT__:0` and the backfill root contains fresh `comparison_summary.json`, `comparison_summary.csv`, `metrics_ffno_bottleneck_base.json`, and `model_profile_ffno_bottleneck_base.json`.
- Then freeze `reference_runs_40ep.json`, launch the fresh `cns-ffno-localconv-40ep-<timestamp>` pilot, collate `compare_40ep_against_existing.json/.csv`, and finish the durable summary, CNS-summary sync, docs index updates, progress-ledger completion entry, execution report, and final implementation-state handoff.

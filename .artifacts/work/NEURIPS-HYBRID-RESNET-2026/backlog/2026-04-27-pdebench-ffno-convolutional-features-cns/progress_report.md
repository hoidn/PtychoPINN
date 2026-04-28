## Active Work

- Added `ffno_bottleneck_localconv_base` as the bounded FFNO-family CNS variant by extending the existing FFNO-close bottleneck with an optional explicit `3x3` local convolution branch and profile provenance fields.
- Added and passed the Task 1/2 deterministic regression surface for the new profile in `tests/torch/test_ffno_bottleneck.py` and `tests/studies/test_pdebench_image128_models.py`.
- Wrote `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/reference_runs_10ep.json`.
- Completed the inspect snapshot at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/inspect-20260428T082501Z`.
- Launched the fresh `10`-epoch pilot at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-10ep-20260428T082501Z` in tmux session `ffno-localconv-10ep` with tracked shell PID marker `__PID__:360772`.

## Current Status

- Deterministic checks completed successfully:
  - `pytest -q tests/torch/test_ffno_bottleneck.py`
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`
  - `python -m compileall -q scripts/studies/pdebench_image128`
  - `python -m compileall -q ptycho_torch/generators/ffno_bottleneck.py`
- The inspect run exited `0` and produced `hdf5_metadata.json`, `dataset_manifest.json`, `invocation.json`, and `invocation.sh`; the runner emitted canonical `split_manifest.json` rather than the plan text's `trajectory_split_manifest.json`. Treat this as an artifact-name deviation, not a semantic blocker, because the existing PDEBench runner/reporting surface already consumes `split_manifest.json`.
- The fresh `10`-epoch pilot is still running; early artifacts already present include `dataset_manifest.json`, `hdf5_metadata.json`, `invocation.json`, `invocation.sh`, `model_profile_ffno_bottleneck_localconv_base.json`, `normalization_stats_state.json`, and `split_manifest.json`.
- There is still no authoritative same-contract `40`-epoch `ffno_bottleneck_base` root under the existing FFNO compare artifacts, so the required `40`-epoch backfill remains pending after the `10`-epoch pilot finishes.

## Next Resume Condition

- Resume when tmux session `ffno-localconv-10ep` reports `__EXIT__:0` and the run root contains fresh `comparison_summary.json`, `comparison_summary.csv`, `metrics_ffno_bottleneck_localconv_base.json`, and `model_profile_ffno_bottleneck_localconv_base.json`.
- Then collate `compare_10ep_against_existing.json/.csv` against the frozen author/FFNO-close/spectral references, assess whether optional continuity rows align for gallery rendering, and proceed to the required `40`-epoch `ffno_bottleneck_base` backfill before launching the `40`-epoch local-conv variant.

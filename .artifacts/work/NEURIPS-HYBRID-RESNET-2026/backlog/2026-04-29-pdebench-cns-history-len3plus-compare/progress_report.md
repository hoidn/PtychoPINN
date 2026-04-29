# Progress Report

## Active Work

- Generalized `scripts/studies/pdebench_image128/reporting.py` so history-delta compares now support either reduced or increased temporal context, derive row-family labels from actual history lengths, and emit dynamic compare filenames such as `compare_10ep_history3_against_history2.json`.
- Added and passed TDD coverage in `tests/studies/test_pdebench_image128_runner.py` for the longer-context `history_len=3` path, including explicit `delta_direction` metadata and dynamic CSV/gallery naming.
- Archived deterministic verification under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/`:
  - `pytest_required.log`
  - `compileall.log`
- Audited and froze the reused `history_len=2` anchors at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history2_reference_runs.json`.
- Ran `inspect` for `history_len=3` and `history_len=4`:
  - `history_len=3`: `input_channels=12`, `target_channels=4`, `windows_per_trajectory=18`, `available_windows=180000`, capped split windows `4096 / 512 / 512`, sample contract `concat u[t-3:t] -> u[t]`
  - `history_len=4`: `input_channels=16`, `target_channels=4`, `windows_per_trajectory=17`, `available_windows=170000`, capped split windows `4096 / 512 / 512`, sample contract `concat u[t-4:t] -> u[t]`
- Launched the mandatory `history_len=3` four-row `10`-epoch pilot under tmux with exact PID tracking.

## Current Status

- `implementation_state`: `RUNNING`
- Active run:
  - tmux session: `pdehist3-10ep`
  - tracked PID: `783286`
  - output root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-10ep-20260429T071905Z`
  - launcher root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/launch-history3-pilot-10ep-20260429T071905Z`
- The first launch attempt was recovered with a narrow fix after the runner correctly refused a non-empty output root created by launcher-side files. The rerun moved launcher files outside the scientific output root.
- The active output root already contains fresh invocation and manifest artifacts (`invocation.json`, `invocation.sh`, `dataset_manifest.json`, `split_manifest.json`, `hdf5_metadata.json`, `normalization_stats_state.json`, and `model_profile_spectral_resnet_bottleneck_base.json`), which confirms the rerun is writing into the intended location.
- Cross-run gallery alignment remains non-fatal and still uses the existing target-equality standard `np.allclose(..., atol=1e-6, rtol=1e-6)`.
- The `40`-epoch `history_len=3` pilot and all compare sidecars/gate artifacts remain pending until the `10`-epoch pilot completes cleanly.

## Next Resume Condition

- Resume when tracked PID `783286` exits with code `0` and the `history3-pilot-10ep-20260429T071905Z` root contains the expected completed-run artifacts, including `comparison_summary.json`, `comparison_summary.csv`, per-profile `metrics_*.json`, per-profile `model_profile_*.json`, and sample outputs.
- After the `10`-epoch run completes, emit `compare_10ep_history3_against_history2.json/.csv` (and sample galleries if targets align), then either:
  - launch the mandatory `40`-epoch `history_len=3` four-row pilot, or
  - if the `10`-epoch run fails, apply a narrow in-scope fix and rerun before changing state.

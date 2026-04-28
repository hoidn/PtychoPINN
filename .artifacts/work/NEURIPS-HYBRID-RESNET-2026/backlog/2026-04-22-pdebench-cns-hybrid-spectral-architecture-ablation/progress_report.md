## Active Work

- Stopped the mis-scoped six-profile pilot relaunch at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-pilot-10ep-20260428T025542Z` because it did not match the approved execution plan's stage order or profile scope.
- Launched the approved Task 2 run under tmux session `cns-sharing10` on socket `/tmp/claude-tmux-sockets/claude.sock` with guarded PID tracking:
  - run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-10ep-20260428T032825Z`
  - tracker root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-10ep-20260428T032825Z.launch`
  - tracked Python PID: `271830`

## Current Status

- Deterministic preflight from the prior pass remains green and still covers the approved runner surface:
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `65 passed in 42.77s`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`
  - `pytest -q tests/studies/test_pdebench_image128_models.py -k "spectral_resnet_bottleneck"` -> `6 passed, 31 deselected in 4.85s`
  - `pytest -q tests/studies/test_pdebench_image128_runner.py -k "pilot or cfd_cns"` -> `12 passed, 16 deselected in 12.71s`
- The approved two-row sharing run is active as of `2026-04-28 03:29:04 UTC`.
- The new run root already contains the required startup/provenance artifacts: `invocation.json`, `invocation.sh`, `dataset_manifest.json`, `split_manifest.json`, `normalization_stats_state.json`, `hdf5_metadata.json`, and `model_profile_spectral_resnet_bottleneck_base.json`.
- Tracker log `stdout.log` has recorded the first training checkpoint:
  - `EPOCH_LOSS profile=spectral_resnet_bottleneck_base epoch=1 loss=0.1613804899`

## Next Resume Condition

- Resume when tracker file `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-10ep-20260428T032825Z.launch/exit_code.txt` exists and records `0`, the tracked Python PID `271830` has exited, and the approved Task 2 run root contains the required completion artifacts:
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_spectral_resnet_bottleneck_base.json`
  - `metrics_spectral_resnet_bottleneck_noshare.json`
  - `model_profile_spectral_resnet_bottleneck_base.json`
  - `model_profile_spectral_resnet_bottleneck_noshare.json`
  - `comparison_spectral_resnet_bottleneck_base_sample0.png`
  - `comparison_spectral_resnet_bottleneck_noshare_sample0.png`
  - `comparison_spectral_resnet_bottleneck_base_sample0.npz`
  - `comparison_spectral_resnet_bottleneck_noshare_sample0.npz`
- Once that condition is met, continue the approved plan in order:
  - write `artifact_audit.json`
  - render `gallery_sharing_sample0.png` and `gallery_sharing_sample0_error.png`
  - write `sharing_ranking_10ep.json`
  - launch the separate `sharing-40ep`, `depth-shared-40ep`, and `finalists-1024cap-40ep` tranches
  - only after those tranches complete, update the durable summary/docs/state surfaces and switch this backlog item to `COMPLETED`

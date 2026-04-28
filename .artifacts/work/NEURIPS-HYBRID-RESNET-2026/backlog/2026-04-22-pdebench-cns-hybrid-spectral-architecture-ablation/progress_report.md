## Active Work

- Completed the approved Task 2 `10`-epoch sharing tranche at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-10ep-20260428T032825Z`, including `artifact_audit.json`, `gallery_sharing_sample0.png`, `gallery_sharing_sample0_error.png`, and `sharing_ranking_10ep.json`.
- Launched the approved Task 3 `40`-epoch sharing tranche under tmux session `cns_share40_20260428T035154Z` on socket `/tmp/claude-tmux-sockets/claude.sock` with guarded PID tracking:
  - run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-40ep-20260428T035154Z`
  - tracker root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-40ep-20260428T035154Z.launch`
  - tracked Python PID: `278753`

## Current Status

- Deterministic preflight from the prior pass remains green and still covers the approved runner surface:
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `65 passed in 42.77s`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`
  - `pytest -q tests/studies/test_pdebench_image128_models.py -k "spectral_resnet_bottleneck"` -> `6 passed, 31 deselected in 4.85s`
  - `pytest -q tests/studies/test_pdebench_image128_runner.py -k "pilot or cfd_cns"` -> `12 passed, 16 deselected in 12.71s`
- The completed Task 2 ranking currently favors `spectral_resnet_bottleneck_noshare` on the plan’s comparison standard (`lower relative_l2`, then `err_nRMSE`, then `fRMSE_high`):
  - `spectral_resnet_bottleneck_noshare`: `relative_l2=0.07957875728607178`, `err_nRMSE=0.07957875728607178`, `fRMSE_high=0.7858846783638`
  - `spectral_resnet_bottleneck_base`: `relative_l2=0.08300281316041946`, `err_nRMSE=0.08300281316041946`, `fRMSE_high=0.7511703968048096`
- The gallery renderer needed an operational `PYTHONPATH=/home/ollie/Documents/PtychoPINN` prefix because `python scripts/studies/pdebench_image128/render_hybrid_upsampler_gallery.py` did not resolve the repo-root package import from the script path alone. This was handled without repo code changes.
- The approved Task 3 `40`-epoch sharing run is active. Its run root already contains startup/provenance artifacts (`invocation.json`, `invocation.sh`, `dataset_manifest.json`, `split_manifest.json`, `normalization_stats_state.json`, `hdf5_metadata.json`, and `model_profile_spectral_resnet_bottleneck_base.json`), and tracker log `stdout.log` shows the Python process started successfully.

## Next Resume Condition

- Resume when tracker file `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-40ep-20260428T035154Z.launch/exit_code.txt` exists and records `0`, the tracked Python PID `278753` has exited, and the Task 3 run root contains the required completion artifacts:
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
  - write Task 3 `artifact_audit.json`
  - render the fresh Task 3 sharing galleries
  - write `sharing_ranking_40ep.json`
  - launch the separate `depth-shared-40ep` and `finalists-1024cap-40ep` tranches
  - only after those tranches complete, update the durable summary/docs/state surfaces and switch this backlog item to `COMPLETED`

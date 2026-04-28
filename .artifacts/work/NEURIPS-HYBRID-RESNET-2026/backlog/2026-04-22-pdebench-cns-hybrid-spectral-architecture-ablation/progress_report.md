## Active Work

- Closed the fresh Stage 1 sharing tranche on the approved fixed CNS contract:
  - `10` epochs run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-10ep-20260428T032825Z`
  - `40` epochs run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-40ep-20260428T035154Z`
  - study-root compare sidecars and manifests now exist for both budgets:
    - `compare_manifest_sharing_10ep.json`
    - `compare_manifest_sharing_40ep.json`
    - `compare_sharing_10ep_against_existing.{json,csv}`
    - `compare_sharing_40ep_against_existing.{json,csv}`
    - `sharing_10ep_ranking.json`
    - `sharing_40ep_ranking.json`
- Launched the approved Task 4 shared-depth `40`-epoch tranche under tmux session `cns_depth40_20260428T043715Z` on socket `/tmp/claude-tmux-sockets/claude.sock` with the plan-required tracked-PID wrapper:
  - run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z`
  - tracker root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z.launch`
  - tracked Python PID: `285260`

## Current Status

- Deterministic preflight remains green for the approved runner surface:
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `65 passed in 42.77s`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`
  - `pytest -q tests/studies/test_pdebench_image128_models.py -k "spectral_resnet_bottleneck"` -> `6 passed, 31 deselected in 4.85s`
  - `pytest -q tests/studies/test_pdebench_image128_runner.py -k "pilot or cfd_cns"` -> `12 passed, 16 deselected in 12.71s`
- Stage 1 fresh results are fully materialized and ranked on the capped `512 / 64 / 64`, `history_len=2`, `mse`, batch-size `4` contract:
  - `sharing_10ep_ranking.json` winner: `spectral_resnet_bottleneck_noshare`
  - `sharing_40ep_ranking.json` winner: `spectral_resnet_bottleneck_base`
- The Stage 2 depth run is active with a live tracked PID and runner output already streaming into the launch log:
  - tracker `python_pid.txt`: present with `285260`
  - tracker `exit_code.txt`: not present yet
  - latest observable state: process `285260` is running `python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode pilot ... --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_shared_blocks8,spectral_resnet_bottleneck_shared_blocks10`
  - launch log path: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z.launch/stdout.log`
- No semantic blocker is present. Remaining work is ordered execution: finish Task 4, emit the depth compare/ranking artifacts, select finalists, then launch Task 5.

## Next Resume Condition

- Resume when `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z.launch/exit_code.txt` exists and records `0`, tracked PID `285260` has exited, and the Stage 2 run root contains the required completion artifacts:
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_spectral_resnet_bottleneck_base.json`
  - `metrics_spectral_resnet_bottleneck_shared_blocks8.json`
  - `metrics_spectral_resnet_bottleneck_shared_blocks10.json`
  - `model_profile_spectral_resnet_bottleneck_base.json`
  - `model_profile_spectral_resnet_bottleneck_shared_blocks8.json`
  - `model_profile_spectral_resnet_bottleneck_shared_blocks10.json`
  - `comparison_spectral_resnet_bottleneck_base_sample0.png`
  - `comparison_spectral_resnet_bottleneck_shared_blocks8_sample0.png`
  - `comparison_spectral_resnet_bottleneck_shared_blocks10_sample0.png`
  - `comparison_spectral_resnet_bottleneck_base_sample0.npz`
  - `comparison_spectral_resnet_bottleneck_shared_blocks8_sample0.npz`
  - `comparison_spectral_resnet_bottleneck_shared_blocks10_sample0.npz`
- Once that condition is met, continue the approved plan in order:
  - write Task 4 `artifact_audit.json`
  - materialize `compare_manifest_depth_40ep.json`
  - emit `compare_depth_40ep_against_existing.{json,csv}` and the sample galleries
  - write `depth_40ep_ranking.json`
  - build `selected_finalists_1024cap.json`
  - launch the separate `finalists-1024cap-40ep` tranche
  - only after Task 5 completes, update the durable summary/docs/state surfaces and switch this backlog item to `COMPLETED`

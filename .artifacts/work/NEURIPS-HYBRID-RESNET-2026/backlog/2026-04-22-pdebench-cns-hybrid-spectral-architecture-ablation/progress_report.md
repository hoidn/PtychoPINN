## Active Work

- Completed the fresh capped `512 / 64 / 64` tranches required before finalist confirmation:
  - sharing `10` epochs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-10ep-20260428T032825Z`
  - sharing `40` epochs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-40ep-20260428T035154Z`
  - shared-depth `40` epochs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z`
- Completed the Stage 0-3 collation artifacts needed to choose finalists and preserve the capped comparison lineage:
  - `reference_runs_10ep.json`
  - `reference_runs_40ep.json`
  - `compare_sharing_10ep_against_existing.{json,csv}`
  - `compare_sharing_40ep_against_existing.{json,csv}`
  - `compare_depth_40ep_against_existing.{json,csv}`
  - `sharing_10ep_ranking.json`
  - `sharing_40ep_ranking.json`
  - `depth_40ep_ranking.json`
  - `selected_finalists_1024cap.json`
- The only active remaining plan step is the approved larger-cap finalist confirmation pass for:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_shared_blocks10`
  - run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z`
  - launcher root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z.launch`
  - tmux session: `cns-finalists1024` on socket `/tmp/claude-tmux-sockets/claude.sock`
  - tracked Python PID: `304669`

## Current Status

- Deterministic preflight for the approved runner surface remains green:
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `65 passed in 42.77s`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`
  - `pytest -q tests/studies/test_pdebench_image128_models.py -k "spectral_resnet_bottleneck"` -> `6 passed, 31 deselected in 4.85s`
  - `pytest -q tests/studies/test_pdebench_image128_runner.py -k "pilot or cfd_cns"` -> `12 passed, 16 deselected in 12.71s`
- As of `2026-04-28T06:50:33Z` (`2026-04-27T23:50:33-0700` local), the tracked Stage 4 run is still actively progressing:
  - `ps -p 304669` still shows the exact planned Python process alive
  - `.launch/exit_code.txt` is still absent, so the PID-wait launcher has not completed
  - `spectral_resnet_bottleneck_base` has already emitted `metrics_spectral_resnet_bottleneck_base.json` and `comparison_spectral_resnet_bottleneck_base_sample0.{png,npz}`
  - `spectral_resnet_bottleneck_shared_blocks10` has not yet emitted its metrics or comparison outputs
  - Stage 4 is not complete yet because the run root still lacks:
    - `comparison_summary.json`
    - `comparison_summary.csv`
    - `metrics_spectral_resnet_bottleneck_shared_blocks10.json`
    - `comparison_spectral_resnet_bottleneck_shared_blocks10_sample0.png`
    - `comparison_spectral_resnet_bottleneck_shared_blocks10_sample0.npz`
- No semantic blocker is present. This backlog item remains in a valid long-running execution state only.

## Next Resume Condition

- Resume when `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z.launch/exit_code.txt` exists and records `0`, PID `304669` has exited, and the Stage 4 run root contains:
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_spectral_resnet_bottleneck_base.json`
  - `metrics_spectral_resnet_bottleneck_shared_blocks10.json`
  - `model_profile_spectral_resnet_bottleneck_base.json`
  - `model_profile_spectral_resnet_bottleneck_shared_blocks10.json`
  - `comparison_spectral_resnet_bottleneck_base_sample0.png`
  - `comparison_spectral_resnet_bottleneck_shared_blocks10_sample0.png`
  - `comparison_spectral_resnet_bottleneck_base_sample0.npz`
  - `comparison_spectral_resnet_bottleneck_shared_blocks10_sample0.npz`
- Once that completion condition is met, continue the approved plan in order:
  - emit `compare_finalists_1024cap_40ep_within_run.{json,csv}`
  - write `finalist_delta_1024cap.json`
  - publish the durable study summary at `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
  - sync the CNS summary, `docs/studies/index.md`, `docs/index.md`, and `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
  - write the final execution report and switch this backlog item to `COMPLETED`

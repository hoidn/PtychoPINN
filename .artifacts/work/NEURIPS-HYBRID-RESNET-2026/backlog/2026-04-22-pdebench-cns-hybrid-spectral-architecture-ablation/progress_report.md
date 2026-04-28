## Active Work

- Completed the fresh capped `512 / 64 / 64` sharing tranches and the fresh capped shared-depth tranche under the fixed CNS shell:
  - `10` epochs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-10ep-20260428T032825Z`
  - `40` epochs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-40ep-20260428T035154Z`
  - shared-depth `40` epochs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z`
- Materialized the missing study-root collation artifacts required before the larger-cap confirmation pass:
  - `reference_runs_10ep.json`
  - `reference_runs_40ep.json`
  - `inspect_run_root.txt`
  - `stage2_sharing_40ep_run_root.txt`
  - `stage3_depth_40ep_run_root.txt`
  - `compare_manifest_depth_40ep.json`
  - `compare_depth_40ep_against_existing.{json,csv}`
  - `depth_40ep_ranking.json`
  - `selected_finalists_1024cap.json`
- Launched the approved fresh `1024 / 128 / 128` finalist confirmation tranche in tmux for the two distinct finalists selected from the fresh `40`-epoch lanes:
  - sharing finalist: `spectral_resnet_bottleneck_base`
  - shared-depth finalist: `spectral_resnet_bottleneck_shared_blocks10`
  - run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z`
  - tracker root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z.launch`
  - tracked Python PID: `304669`
  - tmux session: `cns-finalists1024` on socket `/tmp/claude-tmux-sockets/claude.sock`

## Current Status

- Deterministic preflight is still green for the approved runner surface:
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `65 passed in 42.77s`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`
  - `pytest -q tests/studies/test_pdebench_image128_models.py -k "spectral_resnet_bottleneck"` -> `6 passed, 31 deselected in 4.85s`
  - `pytest -q tests/studies/test_pdebench_image128_runner.py -k "pilot or cfd_cns"` -> `12 passed, 16 deselected in 12.71s`
- The fresh shared-depth tranche completed successfully:
  - run-root `exit_status.txt` is `0`
  - launcher `exit_code.txt` is `0`
  - the run root contains the full per-profile artifact set for `spectral_resnet_bottleneck_base`, `spectral_resnet_bottleneck_shared_blocks8`, and `spectral_resnet_bottleneck_shared_blocks10`
- The fresh `40`-epoch winners are now fixed for Stage 4 selection:
  - `sharing_40ep_ranking.json` winner: `spectral_resnet_bottleneck_base`
  - `depth_40ep_ranking.json` winner: `spectral_resnet_bottleneck_shared_blocks10`
  - `selected_finalists_1024cap.json` records `unique_finalist_count=2` and `stage4_within_run_compare_required=true`
- As of `2026-04-28T05:46:15Z`, the larger-cap confirmation tranche is the only remaining active execution step:
  - `ps` shows PID `304669` active for `python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode pilot ... --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_shared_blocks10`
  - tracker `exit_code.txt` is still absent
  - `python_pid.txt` is present under the launcher root
  - no semantic blocker is present

## Next Resume Condition

- Resume when `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z.launch/exit_code.txt` exists and records `0`, PID `304669` has exited, and the Stage 4 run root contains the required completion artifacts:
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
- Once that condition is met, continue the approved plan in order:
  - emit `compare_finalists_1024cap_40ep_within_run.{json,csv}`
  - write `finalist_delta_1024cap.json`
  - write the durable summary at `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
  - sync the CNS summary, docs index surfaces, and progress ledger
  - write the final execution report and switch this backlog item to `COMPLETED`

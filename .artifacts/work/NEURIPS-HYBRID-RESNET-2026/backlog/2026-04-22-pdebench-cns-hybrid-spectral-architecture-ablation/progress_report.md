## Active Work

- Completed the fresh capped `512 / 64 / 64` sharing and shared-depth tranches under the fixed canonical CNS shell:
  - `10` epochs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-10ep-20260428T032825Z`
  - `40` epochs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-40ep-20260428T035154Z`
  - shared-depth `40` epochs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z`
- Materialized the Stage 0-3 collation sidecars required before the larger-cap confirmation pass, including:
  - `reference_runs_10ep.json`
  - `reference_runs_40ep.json`
  - `compare_depth_40ep_against_existing.{json,csv}`
  - `depth_40ep_ranking.json`
  - `selected_finalists_1024cap.json`
- The only remaining active step is the approved fresh `1024 / 128 / 128` finalist confirmation tranche for:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_shared_blocks10`
  - run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z`
  - tracker root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z.launch`
  - tmux session: `cns-finalists1024` on socket `/tmp/claude-tmux-sockets/claude.sock`
  - tracked Python PID: `304669`

## Current Status

- Deterministic preflight for the approved runner surface remains green:
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `65 passed in 42.77s`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`
  - `pytest -q tests/studies/test_pdebench_image128_models.py -k "spectral_resnet_bottleneck"` -> `6 passed, 31 deselected in 4.85s`
  - `pytest -q tests/studies/test_pdebench_image128_runner.py -k "pilot or cfd_cns"` -> `12 passed, 16 deselected in 12.71s`
- As of `2026-04-28T06:23:38Z` (`2026-04-27T23:23:38-0700` local), the larger-cap confirmation tranche is still actively executing:
  - `ps -p 304669` shows the tracked Python process still alive on the exact planned command line
  - launcher sentinel `.launch/exit_code.txt` is still absent
  - the launcher log now shows `spectral_resnet_bottleneck_base` completed its `40`-epoch training loop and `spectral_resnet_bottleneck_shared_blocks10` has started (`epoch=1`)
  - partial Stage 4 artifacts now exist for the first finalist row, including `metrics_spectral_resnet_bottleneck_base.json` and `comparison_spectral_resnet_bottleneck_base_sample0.{png,npz}`
  - required completion artifacts are still incomplete: `comparison_summary.json`, `comparison_summary.csv`, `metrics_spectral_resnet_bottleneck_shared_blocks10.json`, and the shared-blocks10 sample PNG/NPZ outputs are not present yet
- No semantic blocker is present; the backlog item remains in the long-running execution phase only.

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
  - sync the CNS summary, docs/studies index surfaces, docs index, and the progress ledger
  - write the final execution report and switch this backlog item to `COMPLETED`

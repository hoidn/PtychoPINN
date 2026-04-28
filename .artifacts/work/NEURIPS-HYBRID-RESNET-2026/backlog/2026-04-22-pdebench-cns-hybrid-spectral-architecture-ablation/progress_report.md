## Active Work

- Ran the plan-mandated deterministic preflight for the CNS Hybrid-spectral ablation surface and archived the logs under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/`.
- Relaunched the fresh Task 2 `10`-epoch six-profile pilot under tmux session `cns-hybrid-ablation` on socket `/tmp/claude-tmux-sockets/claude.sock` with guarded tracking:
  - run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-pilot-10ep-20260428T025542Z`
  - tracker root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-pilot-10ep-20260428T025542Z.launch`
  - tracked Python PID: `258770`
- Abandoned the earlier untracked launch root `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-pilot-10ep-20260428T025324Z` after the first wrapper lost durable exit-code capture; it must not be used as completion evidence for this backlog item.

## Current Status

- Fresh deterministic verification is green:
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `65 passed in 42.77s`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`
  - `pytest -q tests/studies/test_pdebench_image128_models.py -k "spectral_resnet_bottleneck"` -> `6 passed, 31 deselected in 4.85s`
  - `pytest -q tests/studies/test_pdebench_image128_runner.py -k "pilot or cfd_cns"` -> `12 passed, 16 deselected in 12.71s`
- The guarded tmux shell is currently waiting on the tracked PID, and the tracker root already contains `python_pid.txt`, `run_root.txt`, and `stdout.log`.
- The active run root already contains startup/provenance artifacts required by the plan, including `invocation.json`, `invocation.sh`, `dataset_manifest.json`, `split_manifest.json`, `normalization_stats_state.json`, `hdf5_metadata.json`, and `model_profile_spectral_resnet_bottleneck_base.json`.
- The relaunched run has advanced into training: `stdout.log` records `EPOCH_LOSS profile=spectral_resnet_bottleneck_base epoch=1 loss=0.1613804007`.

## Next Resume Condition

- Resume when tracker file `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-pilot-10ep-20260428T025542Z.launch/exit_code.txt` exists and records `0`, the tracked Python PID `258770` has exited, and the active Task 2 run root contains the required completion artifacts:
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_*.json` for all six selected profiles
  - `model_profile_*.json` for all six selected profiles
  - `comparison_*_sample0.png`
  - `comparison_*_sample0.npz`
- Once that condition is met, continue with the plan’s remaining current-scope work in order:
  - write `artifact_audit.json`
  - render the shared-family and non-shared-family galleries
  - write `family_ranking_10ep.json`
  - select and launch the bounded `40`-epoch follow-up, then the `1024 / 128 / 128` finalist confirmation
  - only after those tranches finish, update the durable summary/docs/state surfaces and switch this backlog item to `COMPLETED`

# Progress Report

## Active Work

- The capped `2048 / 256 / 256`, `40`-epoch PDEBench `2d_cfd_cns` finalist compare is actively running for:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_shared_blocks10`
- The current task’s frozen reference manifests and preflight artifacts remain in place under:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/`
- Archived green deterministic verification already present for the current repo state:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/workflow_fix_bootstrap_pytest.log`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/final_pytest.log`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/final_compileall.log`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/final_integration.log`
- Active run root:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-20260428T201926Z`
- Active launch sidecar:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-20260428T201926Z.launch`

## Current Status

- Implementation is `RUNNING`.
- The tracked Python PID is `543096` and was still live at the latest check (`ps` status `Rl+`, elapsed `01:01:01`, `%CPU 758`, `%MEM 1.5`).
- The tmux session is `cns-scale2048-201926` on socket `/tmp/claude-tmux-sockets/claude.sock`.
- Initial required fresh artifacts already exist under the run root:
  - `invocation.json`
  - `invocation.sh`
  - `dataset_manifest.json`
  - `split_manifest.json`
  - `hdf5_metadata.json`
- The fresh run is still actively writing to `.launch/stdout.log`; the latest observed training line reached `EPOCH_LOSS profile=spectral_resnet_bottleneck_base epoch=34`.
- The run root currently contains only the base profile metadata (`model_profile_spectral_resnet_bottleneck_base.json`) and does not yet contain:
  - `metrics_spectral_resnet_bottleneck_base.json`
  - `metrics_spectral_resnet_bottleneck_shared_blocks10.json`
  - `model_profile_spectral_resnet_bottleneck_shared_blocks10.json`
  - `comparison_summary.json`
- The launch sidecar still has no `exit_code.txt`, so the run has not satisfied the completion guardrail.

## Next Resume Condition

- Resume when PID `543096` exits and `.launch/exit_code.txt` records `0`.
- Then validate the fresh run emitted the required per-profile metrics and `comparison_summary.json`, generate `finalist_scaling_trend_512_1024_2048.json` plus `.csv`, archive final artifact validation, and finish the durable summary / index / ledger sync before marking the item complete.

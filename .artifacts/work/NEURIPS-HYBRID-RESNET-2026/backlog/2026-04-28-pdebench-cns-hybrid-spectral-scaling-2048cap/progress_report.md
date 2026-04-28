# Progress Report

## Active Work

- The capped `2048 / 256 / 256`, `40`-epoch PDEBench `2d_cfd_cns` finalist compare is still actively running for:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_shared_blocks10`
- Frozen reference manifests, inspect evidence, and archived verification logs remain under:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/`
- Active run root:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-20260428T201926Z`
- Active launch sidecar:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-20260428T201926Z.launch`
- Active command contract:
  - `python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode pilot --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_shared_blocks10 --history-len 2 --epochs 40 --batch-size 4 --max-train-trajectories 2048 --max-val-trajectories 256 --max-test-trajectories 256 --max-windows-per-trajectory 8 --device cuda --num-workers 0`

## Current Status

- Implementation is `RUNNING`.
- The tracked Python PID is `543096` and was still live at `2026-04-28 15:40:12 -0700` with `ps` status `Rl+`, elapsed `02:19:23`, `%CPU 698`, `%MEM 9.1`.
- Required fresh contract artifacts already exist under the run root:
  - `invocation.json`
  - `invocation.sh`
  - `dataset_manifest.json`
  - `split_manifest.json`
  - `hdf5_metadata.json`
- The base finalist row has completed and emitted:
  - `metrics_spectral_resnet_bottleneck_base.json`
  - `model_profile_spectral_resnet_bottleneck_base.json`
  - `comparison_spectral_resnet_bottleneck_base_sample0.npz`
  - `comparison_spectral_resnet_bottleneck_base_sample0.png`
- Latest available base metrics from `metrics_spectral_resnet_bottleneck_base.json`:
  - `err_nRMSE=0.0421656668`
  - `err_RMSE=1.0198297501`
  - `relative_l2=0.0421656668`
  - `fRMSE_low=2.3713548183`
  - `fRMSE_mid=0.2230527103`
  - `fRMSE_high=0.3117601573`
  - `runtime_sec=4311.1880`
- The latest observed stdout line is `EPOCH_LOSS profile=spectral_resnet_bottleneck_shared_blocks10 epoch=30 loss=0.008248754625`, and `.launch/stdout.log` was last updated at `2026-04-28 15:38:32 -0700`.
- The run root also contains `model_profile_spectral_resnet_bottleneck_shared_blocks10.json`, but it does not yet contain:
  - `metrics_spectral_resnet_bottleneck_shared_blocks10.json`
  - `comparison_spectral_resnet_bottleneck_shared_blocks10_sample0.npz`
  - `comparison_spectral_resnet_bottleneck_shared_blocks10_sample0.png`
  - `comparison_summary.json`
  - `comparison_summary.csv`
- The launch sidecar still has no `exit_code.txt`, so the run has not satisfied the completion guardrail.

## Next Resume Condition

- Resume when PID `543096` exits and `.launch/exit_code.txt` records `0`.
- Then verify the fresh run emitted the remaining per-profile metrics plus `comparison_summary.json` and `comparison_summary.csv`, generate `finalist_scaling_trend_512_1024_2048.json` and `.csv`, archive completion verification, and finish the durable summary / CNS summary / ledger sync before marking the item complete.

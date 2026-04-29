# Progress Report

## Active Work

- The mandatory `history_len=3` four-row `40`-epoch pilot remains in progress under the tracked launcher PID `789546`.
- Output root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z`
- Launcher root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/launch-history3-pilot-40ep-20260429T073705Z`
- Deterministic pre-run verification is already green and archived under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/`:
  - `pytest_required.log`: `45 passed in 52.06s`
  - `compileall.log`: clean
- The completed `10`-epoch sidecars remain authoritative:
  - `compare_10ep_history3_against_history2.json`
  - `compare_10ep_history3_against_history2.csv`

## Current Status

- `implementation_state`: `RUNNING`
- The tracked process is still active (`ps` shows PID `789546` alive) and the `40`-epoch run root has only in-progress startup artifacts so far: `invocation.json`, `invocation.sh`, `dataset_manifest.json`, `split_manifest.json`, `hdf5_metadata.json`, `normalization_stats_state.json`, and `model_profile_spectral_resnet_bottleneck_base.json`.
- Current launcher-log progress shows the spectral row still training and having reached epoch `26 / 40`:
  - `EPOCH_LOSS profile=spectral_resnet_bottleneck_base epoch=26 loss=0.01190681034`
- The fixed compare contract remains unchanged from the plan:
  - dataset: official `2d_cfd_cns` file
  - split: `512 / 64 / 64`
  - `max_windows_per_trajectory=8`
  - batch size: `4`
  - training loss: `mse`
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- The cross-run gallery equality standard remains `np.allclose(..., atol=1e-6, rtol=1e-6)`; `10`-epoch gallery generation already stayed non-fatal when targets did not align.

## Next Resume Condition

- Resume when the tracked PID `789546` exits with code `0` and the `history3-pilot-40ep-20260429T073705Z` run root contains the required completed-run artifacts:
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_*.json`
  - `model_profile_*.json`
  - sample outputs
- After the run completes:
  - emit `compare_40ep_history3_against_history2.json` and `.csv`
  - evaluate and write `history4_gate_decision.json` using the plan’s spectral-row gate
  - launch the optional `history_len=4` branch only if that gate opens

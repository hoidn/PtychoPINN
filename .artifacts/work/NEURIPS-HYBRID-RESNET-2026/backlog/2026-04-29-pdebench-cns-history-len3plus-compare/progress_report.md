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
- The tracked process is still active (`ps` shows PID `789546` alive after `38:21`) and the `40`-epoch run root is still incomplete: no final `comparison_summary.json`, `comparison_summary.csv`, or `compare_40ep_history3_against_history2.*` sidecars exist yet.
- The launcher log now shows the first two rows finished and the third row just started:
  - `spectral_resnet_bottleneck_base` reached epoch `40 / 40` with final logged training loss `0.004354039585`
  - `hybrid_resnet_cns` reached epoch `40 / 40` with final logged training loss `0.003469038266`
  - `fno_base` has reached epoch `3 / 40` with latest logged training loss `0.07665816803`
- Partial fresh artifacts already present under the run root include completed spectral and hybrid metrics/sample outputs plus shared manifests; `fno_base` and `unet_strong` are still incomplete.
- Completed partial metrics already written under the active run root:
  - `metrics_spectral_resnet_bottleneck_base.json`: `err_RMSE=1.0991724729537964`, `err_nRMSE=relative_l2=0.04552052542567253`, `fRMSE_low=2.559965133666992`, `fRMSE_mid=0.21568048000335693`, `fRMSE_high=0.34674379229545593`
  - `metrics_hybrid_resnet_cns.json`: `err_RMSE=1.3001306056976318`, `err_nRMSE=relative_l2=0.053842898458242416`, `fRMSE_low=3.053709030151367`, `fRMSE_mid=0.22512021660804749`, `fRMSE_high=0.3200356364250183`
- The fixed compare contract remains unchanged from the plan:
  - dataset: official `2d_cfd_cns` file
  - split: `512 / 64 / 64`
  - `max_windows_per_trajectory=8`
  - batch size: `4`
  - training loss: `mse`
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- The cross-run gallery equality standard remains `np.allclose(..., atol=1e-6, rtol=1e-6)`; `10`-epoch gallery generation already stayed non-fatal when targets did not align.

## Next Resume Condition

- Resume when the tracked PID `789546` exits with code `0`, the launcher root writes `exit_code`, and the `history3-pilot-40ep-20260429T073705Z` run root contains the required completed-run artifacts:
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_*.json`
  - `model_profile_*.json`
  - sample outputs
- After the run completes:
  - emit `compare_40ep_history3_against_history2.json` and `.csv`
  - evaluate and write `history4_gate_decision.json` using the plan’s spectral-row gate
  - launch the optional `history_len=4` branch only if that gate opens

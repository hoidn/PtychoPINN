# Progress Report

## Active Work

- The mandatory `history_len=3` four-row `40`-epoch pilot is still running under tracked PID `789546`.
- Active output root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z`
- Active launcher root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/launch-history3-pilot-40ep-20260429T073705Z`
- Deterministic pre-run verification remains green and archived under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/`:
  - `pytest_required.log`: `45 passed in 52.06s`
  - `compileall.log`: clean exit
- The completed `10`-epoch `history_len=3` sidecars remain authoritative:
  - `compare_10ep_history3_against_history2.json`
  - `compare_10ep_history3_against_history2.csv`

## Current Status

- `implementation_state`: `RUNNING`
- Latest direct process check shows PID `789546` still alive in running TTY state and the launcher root still has no `exit_code`.
- The required `40`-epoch completion artifacts are still absent:
  - no launcher `exit_code`
  - no run-root `comparison_summary.json`
  - no run-root `comparison_summary.csv`
  - no root-level `compare_40ep_history3_against_history2.json`
  - no root-level `compare_40ep_history3_against_history2.csv`
- The fixed fresh-run contract is intact:
  - `history_len=3`
  - `sample_contract=concat u[t-3:t] -> u[t]`
  - split family `512 / 64 / 64`
  - `max_windows_per_trajectory=8`
  - batch size `4`
  - training loss `mse`
  - metric family `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- Training/eval progress from the active run root:
  - `spectral_resnet_bottleneck_base` completed and wrote metrics:
    `err_RMSE=1.0991724729537964`, `err_nRMSE=relative_l2=0.04552052542567253`,
    `fRMSE_low=2.559965133666992`, `fRMSE_mid=0.21568048000335693`,
    `fRMSE_high=0.34674379229545593`
  - `hybrid_resnet_cns` completed and wrote metrics:
    `err_RMSE=1.3001306056976318`, `err_nRMSE=relative_l2=0.053842898458242416`,
    `fRMSE_low=3.053709030151367`, `fRMSE_mid=0.22512021660804749`,
    `fRMSE_high=0.3200356364250183`
  - `fno_base` completed and wrote metrics:
    `err_RMSE=1.3697336912155151`, `err_nRMSE=relative_l2=0.0567254014313221`,
    `fRMSE_low=3.1596221923828125`, `fRMSE_mid=0.1725350171327591`,
    `fRMSE_high=0.6104770302772522`
  - `unet_strong` is still training; the latest launcher log line shows epoch `31 / 40` with training loss `0.06647362369`
- Partial fresh artifacts already present under the active run root include shared manifests, sample outputs for the completed rows, and metrics/model-profile files for `spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, and `fno_base`. `metrics_unet_strong.json` is not present yet.
- The cross-run gallery equality standard remains `np.allclose(..., atol=1e-6, rtol=1e-6)`.

## Next Resume Condition

- Resume when tracked PID `789546` exits with code `0`, the launcher root writes `exit_code`, and the `history3-pilot-40ep-20260429T073705Z` run root contains the required completed-run artifacts:
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_*.json`
  - `model_profile_*.json`
  - sample outputs for all four rows
- After the run completes:
  - emit `compare_40ep_history3_against_history2.json` and `.csv`
  - evaluate and write `history4_gate_decision.json` using the plan’s `history_len=3` gate
  - launch the optional `history_len=4` branch only if that gate opens

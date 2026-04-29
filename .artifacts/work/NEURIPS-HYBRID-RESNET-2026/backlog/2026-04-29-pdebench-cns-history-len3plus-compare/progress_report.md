# Progress Report

## Active Work

- Completed the mandatory `history_len=3` four-row `10`-epoch pilot at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-10ep-20260429T071905Z` with tracked PID `783286` exiting `0` and the full expected run-artifact set present.
- Emitted the required cross-history sidecars:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/compare_10ep_history3_against_history2.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/compare_10ep_history3_against_history2.csv`
- Preserved the fixed equal-footing contract and recorded the allowed delta as `history_len=2 -> 3`, sample contract `concat u[t-2:t] -> u[t]` -> `concat u[t-3:t] -> u[t]`, and input channels `8 -> 12`.
- Confirmed the compare-gallery helper remains non-fatal when targets do not align; the `10ep` sidecar records `cross_run_gallery_blocked.reason=target_mismatch`, and the target-equality standard remains `np.allclose(..., atol=1e-6, rtol=1e-6)`.
- Launched the mandatory `history_len=3` four-row `40`-epoch pilot under tmux with exact PID tracking:
  - tmux session: `pdehist3-40ep`
  - tracked PID: `789546`
  - output root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z`
  - launcher root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/launch-history3-pilot-40ep-20260429T073705Z`

## Current Status

- `implementation_state`: `RUNNING`
- Deterministic pre-run verification remains the same archived green evidence under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/`:
  - `pytest_required.log`
  - `compileall.log`
- The frozen reference manifest and inspect artifacts remain authoritative:
  - `history2_reference_runs.json`
  - `history3-inspect-20260429T000000Z`
  - `history4-inspect-20260429T000000Z`
- The completed `10ep` compare gives mixed decision-support evidence:
  - `spectral_resnet_bottleneck_base`: `err_nRMSE 0.08699 -> 0.14079`, `fRMSE_high 0.69554 -> 0.64773`
  - `hybrid_resnet_cns`: `err_nRMSE 0.09440 -> 0.11196`, `fRMSE_high 0.80004 -> 0.61125`
  - `fno_base`: `err_nRMSE 0.10634 -> 0.09539`, `fRMSE_high 0.92805 -> 0.82812`
  - `unet_strong`: `err_nRMSE 0.62225 -> 0.73266`, `fRMSE_high 3.62936 -> 6.39285`
- The `40`-epoch run is now the critical path for:
  - `compare_40ep_history3_against_history2.json/.csv`
  - the `history4_gate_decision.json`
  - the durable summary and CNS summary updates

## Next Resume Condition

- Resume when tracked PID `789546` exits with code `0` and `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z` contains the expected completed-run artifacts, including `comparison_summary.json`, `comparison_summary.csv`, per-profile `metrics_*.json`, per-profile `model_profile_*.json`, and sample outputs.
- After the `40`-epoch run completes:
  - emit `compare_40ep_history3_against_history2.json/.csv`
  - evaluate and write `history4_gate_decision.json` using the spectral-row gate (`err_nRMSE` must improve relative to `history_len=2` without worsening `fRMSE_high`)
  - if the gate stays closed, stop at `history4_status: not_run`; if it opens, launch the optional `history_len=4` pilots under the same tmux/PID guardrail

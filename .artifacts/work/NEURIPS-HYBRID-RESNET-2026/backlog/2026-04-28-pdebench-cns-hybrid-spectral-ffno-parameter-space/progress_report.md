# Active Work

- The bounded `10`-epoch CNS pilot is running in tmux session `cns-ffno-pspace` for the two approved fresh rows: `spectral_resnet_bottleneck_base_down1` and `spectral_resnet_bottleneck_base_transpose`.
- Active run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/cns-shell-probes-10ep-20260429T115757Z`
- Tracked PID: `927512`
- Launcher log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/launcher-20260429T115757Z.log`

# Current Status

- Added the two missing manual-only spectral shell probe profiles in `scripts/studies/pdebench_image128/run_config.py` and covered them with focused model/runner tests.
- Mandatory deterministic gate passed before launch:
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
- Frozen Task 1 artifacts were written under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/`:
  - `study_matrix.json`
  - `reference_runs_10ep.json`
  - `reference_runs_40ep.json`
  - inspect snapshot `inspect-20260429T115506Z/` including `inspection_manifest.json`
- The first tmux launch failed immediately with the output-root guard because the shell precreated the run directory for `tee`. That was a narrow harness issue; the run was relaunched with the log outside the output root and the scientific contract unchanged.

# Next Resume Condition

- Resume when PID `927512` exits.
- Treat the run as complete only if the tmux pane reports `EXIT:0` and the active run root contains fresh artifacts for both fresh profiles, at minimum:
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_spectral_resnet_bottleneck_base_down1.json`
  - `metrics_spectral_resnet_bottleneck_base_transpose.json`
  - `model_profile_spectral_resnet_bottleneck_base_down1.json`
  - `model_profile_spectral_resnet_bottleneck_base_transpose.json`
- After that, generate `compare_10ep_against_existing.{json,csv}` against the frozen `reference_runs_10ep.json`, apply the plan ordering keys (`relative_l2`, then `err_nRMSE`, then `fRMSE_high`), and decide whether either row earns a `40`-epoch promotion.

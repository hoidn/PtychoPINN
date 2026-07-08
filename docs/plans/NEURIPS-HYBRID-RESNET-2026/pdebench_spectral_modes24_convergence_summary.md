# PDEBench CNS Spectral Modes-24 Convergence Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-28-pdebench-cns-spectral-modes24-convergence-compare`
- Date: `2026-04-29`
- Status: implementation complete; capped `80`-epoch CNS convergence compare complete
- Scope: rerun the shared spectral `12/12` and `24/24` rows fresh on the fixed capped CNS contract, prove the shared batch size, emit a convergence audit, and publish a durable capped-evidence interpretation
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-28-pdebench-cns-spectral-modes24-convergence-compare/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare`

This summary records capped decision-support evidence only. It does not create a benchmark-complete CNS claim and does not justify promoting the `24/24` row into a default profile bundle.

## Fixed Fairness Contract

- task: `2d_cfd_cns`
- dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split: `1024 / 128 / 128` trajectories
- `history_len=2`
- `max_windows_per_trajectory=8`
- training loss: `mse`
- epochs: `80`
- resolved batch size: `16`
- metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`

The shell stayed fixed at:

- `base_model="spectral_resnet_bottleneck_net"`
- `hidden_channels=32`
- `fno_blocks=4`
- `hybrid_downsample_steps=2`
- `hybrid_resnet_blocks=6`
- `hybrid_skip_connections=True`
- `hybrid_skip_style="add"`
- `hybrid_upsampler="pixelshuffle"`
- `spectral_bottleneck_blocks=6`
- `spectral_bottleneck_share_weights=True`
- `spectral_bottleneck_gate_init=0.1`
- `spectral_bottleneck_gate_mode="shared"`

The only intended config change was:

- `fno_modes: 12 -> 24`
- `spectral_bottleneck_modes: 12 -> 24`

No code patching was needed for this item. The landed `spectral_resnet_bottleneck_modes24` profile, inspect gate, and convergence-audit helper already matched the approved contract.

## Contract And Batch-Size Readiness

Exact-contract inspect root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/inspect-1024cap`

Shared batch-size decision record:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/resolved_batch_size.json`

Observed readiness outcome:

- target batch size `16` held for both rows
- no fallback was needed
- inspect and probe artifacts both preserve the fixed `1024 / 128 / 128`, `history_len=2`, `max_windows_per_trajectory=8`, `mse` contract

## Fresh Paired 80-Epoch Run

Fresh paired run root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/cns-spectral-modes24-vs-base-1024cap-80ep`

Launcher completion evidence:

- tracked launcher exit code: `0`
- launcher log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/.launch/cns-spectral-modes24-vs-base-1024cap-80ep/stdout_stderr.log`
- required run artifacts present:
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_spectral_resnet_bottleneck_base.json`
  - `metrics_spectral_resnet_bottleneck_modes24.json`
  - `model_profile_spectral_resnet_bottleneck_base.json`
  - `model_profile_spectral_resnet_bottleneck_modes24.json`

Observed final metrics at `80` epochs:

- `spectral_resnet_bottleneck_base`:
  - `err_nRMSE=0.0382686`
  - `err_RMSE=0.912674`
  - `relative_l2=0.0382686`
  - `fRMSE_low=2.11528`
  - `fRMSE_mid=0.220081`
  - `fRMSE_high=0.282737`
- `spectral_resnet_bottleneck_modes24`:
  - `err_nRMSE=0.0395326`
  - `err_RMSE=0.942819`
  - `relative_l2=0.0395326`
  - `fRMSE_low=2.17721`
  - `fRMSE_mid=0.241145`
  - `fRMSE_high=0.306903`

At the stop point, the shared `12/12` base row finished ahead on every reported eval metric.

## Convergence Audit

Convergence audit outputs:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/convergence_audit.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/convergence_audit.csv`

Fixed convergence rule:

- `late_window_mean_prev = mean(losses[60:70])`
- `late_window_mean_final = mean(losses[70:80])`
- `late_window_ratio = late_window_mean_final / late_window_mean_prev`
- `last5_delta = losses[79] - losses[74]`
- a row remains materially improving if `late_window_ratio < 0.95` or `last5_delta <= -0.001`

Observed audit values:

- `spectral_resnet_bottleneck_base`:
  - `late_window_mean_prev=0.002571`
  - `late_window_mean_final=0.002344`
  - `late_window_ratio=0.911709`
  - `last5_delta=0.000030`
  - `still_materially_improving=true`
- `spectral_resnet_bottleneck_modes24`:
  - `late_window_mean_prev=0.002561`
  - `late_window_mean_final=0.002298`
  - `late_window_ratio=0.897442`
  - `last5_delta=-0.000083`
  - `still_materially_improving=true`

Interpretation under the plan's fixed rule:

- both rows still satisfy the material-improvement rule at `80` epochs because both late-window ratios are below `0.95`
- the result is therefore inconclusive at this capped stop point
- because both rows are still improving, the base row's better final `80`-epoch metrics are not enough to claim a clean spectral-mode win over the still-improving `24/24` row

## Capacity And Runtime Note

- `spectral_resnet_bottleneck_base`: `8,391,814` parameters, `runtime_sec=3218.01`
- `spectral_resnet_bottleneck_modes24`: `25,152,646` parameters, `runtime_sec=3296.38`

The `24/24` row remained materially larger and slightly slower while finishing behind the shared `12/12` row on every final eval metric at the capped `80`-epoch stop. That makes the current evidence weaker than a clean promotion case even before applying the audit's inconclusive label.

## Verification

Commands run from `/home/ollie/Documents/PtychoPINN`:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
python - <<'PY'
from pathlib import Path

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare")
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes24_convergence_summary.md"),
    artifact_root / "resolved_batch_size.json",
    artifact_root / "convergence_audit.json",
    artifact_root / "convergence_audit.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing required outputs: {missing}")
print("modes24 convergence summary and audit outputs present")
PY
```

Observed results:

- `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `77 passed`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`
- required summary and convergence-audit outputs present

## Claim Boundary

- this is capped CNS decision-support evidence only
- it does not satisfy the PDEBench full-training benchmark gate
- it does not justify promoting `spectral_resnet_bottleneck_modes24` into a default bundle
- because both rows were still materially improving at `80` epochs, the correct read is inconclusive rather than a clean `12/12` or `24/24` convergence verdict

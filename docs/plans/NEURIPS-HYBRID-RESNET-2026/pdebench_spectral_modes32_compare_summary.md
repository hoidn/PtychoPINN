# PDEBench CNS Spectral Modes-32 Compare Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-21-pdebench-cns-spectral-modes32-compare`
- Date: `2026-04-28`
- Status: implementation complete; capped CNS modes-32 compare complete
- Scope: raise both spectral mode knobs from `12` to `32` for the shared spectral CNS row, keep the capped local CNS contract fixed, reuse the recovered fresh `10`-epoch row, finish the fresh `40`-epoch row, and publish anchored `10`/`40`-epoch compare sidecars
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-pdebench-cns-spectral-modes32-compare/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare`

This summary records capped decision-support evidence only. It does not create a benchmark-complete CNS claim or justify promoting the modes-32 row into a default profile bundle.

## Fixed Fairness Contract

- task: `2d_cfd_cns`
- dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split: `512 / 64 / 64` trajectories
- `max_windows_per_trajectory=8`
- `history_len=2`
- training loss: `mse`
- batch size: `4`
- budgets: `10` and `40` epochs
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

- `fno_modes: 12 -> 32`
- `spectral_bottleneck_modes: 12 -> 32`

No code patching was needed after the audit; the landed profile/tests already matched this contract.

## Reused 10-Epoch Row

Recovered fresh run root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/cns-spectral-modes32-10ep-20260428T010825Z`

Anchored sidecars:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_10ep_against_existing.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_10ep_against_existing.csv`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_10ep_sample0.png`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_10ep_sample0_error.png`

Observed `10`-epoch outcome:

- `spectral_resnet_bottleneck_modes32`: `err_nRMSE=0.0840402`, `err_RMSE=2.03099`, `fRMSE_low=4.7240`, `fRMSE_mid=0.3788`, `fRMSE_high=0.6861`
- shared `spectral_resnet_bottleneck_base`: `err_nRMSE=0.0869939`, `err_RMSE=2.10237`, `fRMSE_low=4.8862`, `fRMSE_mid=0.4221`, `fRMSE_high=0.6955`
- `fno_base`: `err_nRMSE=0.1063433`, `fRMSE_high=0.9280`
- `unet_strong`: `err_nRMSE=0.6222500`, `fRMSE_high=3.6294`

Interpretation at `10` epochs:

- modes-32 improved every tracked eval metric over the shared `12/12` spectral row
- it also stayed ahead of the reused `fno_base` and `unet_strong` anchors

## Authoritative 40-Epoch Row

Authoritative fresh run root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/cns-spectral-modes32-40ep-20260428T014353Z`

The paired `.launch` tracker recorded exit code `0`, and the root contains the required completion artifacts:

- `metrics_spectral_resnet_bottleneck_modes32.json`
- `comparison_summary.json`
- `comparison_summary.csv`
- `comparison_spectral_resnet_bottleneck_modes32_sample0.npz`
- `comparison_spectral_resnet_bottleneck_modes32_sample0.png`

Anchored sidecars:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_40ep_against_existing.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_40ep_against_existing.csv`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_40ep_sample0.png`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_40ep_sample0_error.png`

Observed `40`-epoch outcome:

- `spectral_resnet_bottleneck_modes32`: `err_nRMSE=0.0645505`, `err_RMSE=1.55999`, `fRMSE_low=3.6761`, `fRMSE_mid=0.2425`, `fRMSE_high=0.3472`
- shared `spectral_resnet_bottleneck_base`: `err_nRMSE=0.0615620`, `err_RMSE=1.48776`, `fRMSE_low=3.4756`, `fRMSE_mid=0.2800`, `fRMSE_high=0.4349`
- `fno_base`: `err_nRMSE=0.0740992`, `fRMSE_high=0.6718`
- `unet_strong`: `err_nRMSE=0.6757976`, `fRMSE_high=1.3326`

Interpretation at `40` epochs:

- modes-32 did not beat the shared `12/12` spectral row on aggregate denormalized error:
  - `err_nRMSE`: `0.0645505` vs `0.0615620`
  - `err_RMSE`: `1.55999` vs `1.48776`
  - `fRMSE_low`: `3.6761` vs `3.4756`
- modes-32 did improve the higher-frequency side of the contract versus shared `12/12`:
  - `fRMSE_mid`: `0.2425` vs `0.2800`
  - `fRMSE_high`: `0.3472` vs `0.4349`
- modes-32 still stayed ahead of the reused `fno_base` and `unet_strong` anchors on aggregate error

## Capacity And Runtime Note

The modes-32 row remained manual-only and materially larger than the shared `12/12` spectral row:

- `spectral_resnet_bottleneck_modes32`: `42,388,614` parameters, `runtime_sec=1382.12`
- `spectral_resnet_bottleneck_base` (`12/12`): `8,186,726` parameters, `runtime_sec=1861.63`

That parameter jump, together with the `40`-epoch aggregate regression, is why this item does not support a default-profile promotion even though the higher-mode row improved `fRMSE_mid/high`.

## Verification

Commands run from `/home/ollie/Documents/PtychoPINN`:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes32_compare_summary.md"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_10ep_against_existing.json"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_40ep_against_existing.json"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing required outputs: {missing}")
print("modes32 summary and compare outputs present")
PY
```

Observed results:

- `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `65 passed`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`
- required summary/compare outputs present

Comparison standard:

- the anchored `10`/`40`-epoch sidecars enforce exact equality on the fixed capped contract and permit only the coupled mode change `fno_modes=32`, `spectral_bottleneck_modes=32`
- gallery rendering succeeded for both budgets, so no tolerance-based alignment fallback was needed for this item

## Claim Boundary

- this is capped CNS decision-support evidence only
- it does not satisfy the PDEBench full-training benchmark gate
- it does not justify promoting `spectral_resnet_bottleneck_modes32` into any default bundle
- the shared `12/12` spectral row remains the better aggregate local capped `40`-epoch spectral reference, while the modes-32 row remains useful as a manual high-frequency follow-up

# PDEBench CNS Hybrid-Spectral To FFNO Parameter-Space Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-28-pdebench-cns-hybrid-spectral-ffno-parameter-space`
- Date: `2026-04-29`
- Status: implementation complete; capped CNS shell-bridge follow-up closed without a `40`-epoch promotion
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-28-pdebench-cns-hybrid-spectral-ffno-parameter-space/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space`

This summary records capped decision-support evidence only. It does not create a
benchmark-complete CNS claim, a paper-facing FFNO conclusion, or a default-profile
promotion.

## Fixed Fairness Contract

- task: `2d_cfd_cns`
- dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- history contract: `concat u[t-2:t] -> u[t]`
- trajectories: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- training loss: `mse`
- batch size: `4`
- metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- promotion ordering keys: `relative_l2`, then `err_nRMSE`, then `fRMSE_high`

Fresh rows had to stay anchored to `spectral_resnet_bottleneck_base`, changing
only one shell axis at a time:

- `spectral_resnet_bottleneck_base_down1`: only `hybrid_downsample_steps: 2 -> 1`
- `spectral_resnet_bottleneck_base_transpose`: only
  `hybrid_upsampler: pixelshuffle -> cyclegan_transpose`

## Exact Profile Matrix

Frozen authorities written for this item:

- study matrix:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/study_matrix.json`
- `10`-epoch reference manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/reference_runs_10ep.json`
- `40`-epoch reference manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/reference_runs_40ep.json`

Required reused compare rows:

- `author_ffno_cns_base`
- `ffno_bottleneck_base`
- `spectral_resnet_bottleneck_base`

Optional reused context:

- `hybrid_resnet_cns`
- `fno_base`
- `unet_strong`
- `spectral_resnet_bottleneck_shared_blocks10` remained available only as
  optional deeper-spectral `40`-epoch context if a promotion had been needed
- `ffno_bottleneck_localconv_base` remained part of the reused bottleneck-family
  interpretation from the prerequisite FFNO local-conv summary rather than a
  fresh row in this item

Fresh rows executed in this item:

- `spectral_resnet_bottleneck_base_down1`
- `spectral_resnet_bottleneck_base_transpose`

Promotion outcome:

- neither fresh row qualified for a `40`-epoch rerun
- no `compare_40ep_against_existing.{json,csv}` was created for this item
- durable promotion decision:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/promotion_decision.json`

## Inspection And Compare Artifacts

- inspect proof:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/inspect-20260429T115506Z`
- authoritative fresh `10`-epoch run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/cns-shell-probes-10ep-20260429T115757Z`
- launcher log with tracked `EXIT:0`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/launcher-20260429T115757Z.log`
- anchored `10`-epoch compare:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/compare_10ep_against_existing.json`
- anchored `10`-epoch compare CSV:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/compare_10ep_against_existing.csv`
- anchored prediction gallery:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/compare_10ep_sample0.png`
- anchored error gallery:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/compare_10ep_sample0_error.png`

## Per-Axis Results

Reference anchor at `10` epochs:

- `spectral_resnet_bottleneck_base`:
  `relative_l2=0.0869938582`, `err_nRMSE=0.0869938582`,
  `fRMSE_high=0.6955373287`, `runtime_sec=426.66`,
  `parameter_count=8,186,726`

### Encoder / Downsampling

- `spectral_resnet_bottleneck_base_down1`:
  `relative_l2=0.1049272269`, `err_nRMSE=0.1049272269`,
  `fRMSE_high=0.9592211246`, `runtime_sec=279.35`,
  `parameter_count=2,674,886`

Directional read against the shared spectral anchor:

- aggregate ordering keys got materially worse:
  - `relative_l2`: `0.0869938582 -> 0.1049272269`
  - `err_nRMSE`: `0.0869938582 -> 0.1049272269`
  - `fRMSE_high`: `0.6955373287 -> 0.9592211246`
- `fRMSE_mid` improved (`0.4221225977 -> 0.3117409647`), but not enough to
  offset the losses on the declared promotion keys
- the row did reduce cost sharply:
  - parameters: `8,186,726 -> 2,674,886`
  - runtime: `426.66s -> 279.35s`

Decision:

- do not promote `spectral_resnet_bottleneck_base_down1` to `40` epochs from
  this item

### Decoder

- `spectral_resnet_bottleneck_base_transpose`:
  `relative_l2=0.1664283574`, `err_nRMSE=0.1664283574`,
  `fRMSE_high=0.7184003592`, `runtime_sec=289.51`,
  `parameter_count=8,186,726`

Directional read against the shared spectral anchor:

- every declared promotion key worsened:
  - `relative_l2`: `0.0869938582 -> 0.1664283574`
  - `err_nRMSE`: `0.0869938582 -> 0.1664283574`
  - `fRMSE_high`: `0.6955373287 -> 0.7184003592`
- low-band error regressed badly:
  `fRMSE_low 4.8862357140 -> 9.5397205353`
- `fRMSE_mid` improved slightly (`0.4221225977 -> 0.4035171568`), but the
  capped decision rule does not justify carrying that trade into a longer run
- runtime was lower (`426.66s -> 289.51s`) but parameter count stayed the same

Decision:

- do not promote `spectral_resnet_bottleneck_base_transpose` to `40` epochs

### Bottleneck Continuum

This item reused the completed bottleneck-family authorities rather than
inventing a new bottleneck row.

What still stands after the new shell probes:

- the shared spectral anchor remains stronger than the plain repo-local
  `ffno_bottleneck_base` on the capped `10`-epoch contract:
  `relative_l2 0.0869938582` vs `0.1110704020`
- the prerequisite FFNO local-conv lane remains the stronger repo-local
  FFNO-family alternative:
  - `10` epochs: `relative_l2=0.0846254751`, `fRMSE_high=0.6369161010`
  - `40` epochs: `relative_l2=0.0557734184`, `fRMSE_high=0.3090891540`
- the official authored FFNO row remains the strongest FFNO-family same-contract
  reference at `40` epochs:
  `relative_l2=0.0281477310`, `fRMSE_high=0.1210141182`

Interpretation:

- neither shell-bridge probe beat the spectral anchor on the ordering keys
- this item therefore narrows the conclusion to a negative shell result:
  removing one downsampling stage or swapping the canonical pixelshuffle decoder
  for `cyclegan_transpose` does not create a better bridge from the current
  Hybrid-spectral shell toward the local FFNO-family lane on this capped CNS
  slice

## Carry-Forward Recommendation

- keep `spectral_resnet_bottleneck_base` as the aggregate local shell anchor for
  capped CNS comparisons
- keep `ffno_bottleneck_localconv_base` as the stronger repo-local FFNO-family
  follow-up profile when later local studies need an FFNO-side comparator
- do not promote `spectral_resnet_bottleneck_base_down1` or
  `spectral_resnet_bottleneck_base_transpose` into longer capped follow-up,
  full-training CNS, or CDI follow-up from this item alone

## Verification

Commands run from `/home/ollie/Documents/PtychoPINN`:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
pytest -v -m integration
python - <<'PY'
import json
from pathlib import Path
required = [
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/reference_runs_10ep.json"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/reference_runs_40ep.json"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/compare_10ep_against_existing.json"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/promotion_decision.json"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_ffno_parameter_space_summary.md"),
]
for path in required:
    if not path.exists():
        raise SystemExit(f"missing required output: {path}")
for path in required[:4]:
    json.loads(path.read_text())
print("required CNS parameter-space outputs parse successfully")
PY
```

Observed results:

- `tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`:
  `86 passed in 57.86s`
- `compileall`: exit `0`
- `pytest -v -m integration`:
  `5 passed, 4 skipped, 1651 deselected in 298.81s`
- inspect/manifests validation, compare generation, launcher exit capture, and
  final output validation logs archived under:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/verification/`

## Claim Boundary

- this is capped decision-support evidence only
- it does not satisfy the PDEBench full-training benchmark gate
- it does not justify a default-profile promotion
- it does not create a new `40`-epoch shell row because the `10`-epoch
  promotion gate closed cleanly

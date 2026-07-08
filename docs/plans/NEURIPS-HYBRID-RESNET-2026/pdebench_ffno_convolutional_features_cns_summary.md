# PDEBench FFNO Convolutional Features CNS Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-27-pdebench-ffno-convolutional-features-cns`
- Date: `2026-04-28`
- Status: implementation complete; capped CNS FFNO local-conv compare complete
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-pdebench-ffno-convolutional-features-cns/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns`

This summary records capped decision-support evidence only. It does not create a
benchmark-complete CNS claim, a paper-facing FFNO result, or a default-profile
promotion.

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

The shell stayed fixed relative to `ffno_bottleneck_base`:

- same encoder / decoder shell
- same latent width and block count
- same downsampling depth
- same skip-add routing
- same output head

The only intended model change was the bottleneck interior:

- `ffno_bottleneck_base`: FFNO-close spectral-plus-feedforward blocks
- `ffno_bottleneck_localconv_base`: the same FFNO-close bottleneck plus an
  explicit local `3x3` residual branch per block

## Implementation Audit

The Task 1 execution-plan inspection artifact is now recorded at:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/inspect-20260428T082501Z/inspection_manifest.json`

That audit confirmed that the bounded local-conv profile was already landed
before this pass:

- generator support: `ptycho_torch/generators/ffno_bottleneck.py`
- model wiring: `scripts/studies/pdebench_image128/models.py`
- profile registration: `scripts/studies/pdebench_image128/run_config.py`
- focused tests: `tests/torch/test_ffno_bottleneck.py`,
  `tests/studies/test_pdebench_image128_models.py`,
  `tests/studies/test_pdebench_image128_runner.py`

No additional model or runner patch was required in this pass. The remaining
work was execution, compare collation, and durable reporting.

## Reference Roots And Compare Artifacts

Frozen manifests:

- `reference_runs_10ep.json`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/reference_runs_10ep.json`
- `reference_runs_40ep.json`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/reference_runs_40ep.json`

Fresh or newly generated roots in this item:

- `10`-epoch local-conv row:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-10ep-20260428T082501Z`
- `40`-epoch FFNO-close fairness backfill:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-close-backfill-40ep-20260428T084852Z`
- authoritative `40`-epoch local-conv row:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-40ep-20260428T090626Z`

The earlier root
`.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-40ep-20260428T090543Z`
is non-authoritative. It retains invocation and fixed-contract metadata only,
while the later `20260428T090626Z` root is the one with the required pilot-mode
comparison and metrics artifacts.

Reused historical anchors:

- `10`-epoch `author_ffno_cns_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-10ep-20260422T232119Z`
- `10`-epoch `ffno_bottleneck_base` and `spectral_resnet_bottleneck_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- `40`-epoch `author_ffno_cns_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`
- `40`-epoch `spectral_resnet_bottleneck_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`

Anchored sidecars:

- `compare_10ep_against_existing.json`
- `compare_10ep_against_existing.csv`
- `compare_10ep_sample0.png`
- `compare_10ep_sample0_error.png`
- `compare_40ep_against_existing.json`
- `compare_40ep_against_existing.csv`
- `compare_40ep_sample0.png`
- `compare_40ep_sample0_error.png`

The `40`-epoch FFNO-close backfill was required. No authoritative same-contract
`pilot` root existed before this item, so the fairness-critical anchor had to be
generated here before interpreting the local-conv `40`-epoch result.

## Results

### `10`-Epoch Local-Conv Row

Observed metrics:

- `ffno_bottleneck_localconv_base`: `err_nRMSE=0.0846255`,
  `relative_l2=0.0846255`, `fRMSE_low=4.7740`, `fRMSE_mid=0.3639`,
  `fRMSE_high=0.6369`
- `author_ffno_cns_base`: `relative_l2=0.0878335`,
  `fRMSE_high=0.2597`
- `spectral_resnet_bottleneck_base`: `relative_l2=0.0869939`,
  `fRMSE_high=0.6955`
- `ffno_bottleneck_base`: `relative_l2=0.1110704`,
  `fRMSE_high=0.7277`

Directional read:

- local-conv clearly improved the repo-local FFNO-close row on both aggregate
  and high-frequency error:
  - `relative_l2`: `0.11107 -> 0.08463`
  - `fRMSE_high`: `0.7277 -> 0.6369`
- local-conv also edged the capped shared-spectral row on aggregate error and
  on `fRMSE_high`
- the official authored FFNO row still kept a much lower `fRMSE_high`, even
  though the local-conv row slightly won aggregate `relative_l2` on this
  10-epoch slice

### `40`-Epoch Local-Conv Row

Observed metrics:

- `ffno_bottleneck_localconv_base`: `err_nRMSE=0.0557734`,
  `relative_l2=0.0557734`, `fRMSE_low=3.1655`, `fRMSE_mid=0.2523`,
  `fRMSE_high=0.3091`
- `author_ffno_cns_base`: `relative_l2=0.0281477`,
  `fRMSE_high=0.1210`
- `spectral_resnet_bottleneck_base`: `relative_l2=0.0615620`,
  `fRMSE_high=0.4349`
- `ffno_bottleneck_base`: `relative_l2=0.0762242`,
  `fRMSE_high=0.3934`

Directional read:

- local-conv again materially improved the repo-local FFNO-close row:
  - `relative_l2`: `0.07622 -> 0.05577`
  - `fRMSE_high`: `0.3934 -> 0.3091`
- local-conv also beat the capped shared-spectral row on both aggregate error
  and high-frequency error:
  - `relative_l2`: `0.06156 -> 0.05577`
  - `fRMSE_high`: `0.4349 -> 0.3091`
- the official authored FFNO row still remained clearly stronger than the
  local-conv row on the same capped contract, both on aggregate error and on
  `fRMSE_high`

## Cost And Carry-Forward Read

Parameter and runtime context:

- `ffno_bottleneck_localconv_base`: `7,695,205` params,
  `275.40s` at `10` epochs, `1031.22s` at `40` epochs
- `ffno_bottleneck_base`: `6,809,701` params,
  `353.56s` at `10` epochs, `980.52s` at `40` epochs
- `spectral_resnet_bottleneck_base`: `8,186,726` params,
  `426.66s` at `10` epochs, `1861.63s` at `40` epochs
- `author_ffno_cns_base`: `1,073,672` params,
  `1328.71s` at `10` epochs, `4725.51s` at `40` epochs

Interpretation:

- the local-conv branch is worth keeping as the stronger repo-local FFNO-family
  proxy for future bounded local follow-ups
- it is not a default-profile promotion for the broader suite because this lane
  remains capped evidence only and the official authored FFNO row still sets the
  stronger capped reference at `40` epochs
- the local-conv row does, however, replace the plain FFNO-close bottleneck as
  the more credible local answer to "what if FFNO gets an explicit local
  branch?"

## Verification

Commands run from `/home/ollie/Documents/PtychoPINN`:

```bash
pytest -q tests/torch/test_ffno_bottleneck.py
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py ptycho_torch/generators/ffno_bottleneck.py
python - <<'PY'
import json
from pathlib import Path
required = [
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/reference_runs_40ep.json"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/compare_40ep_against_existing.json"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md"),
]
for path in required:
    if not path.exists():
        raise SystemExit(f"missing required output: {path}")
json.loads(Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/reference_runs_40ep.json").read_text())
json.loads(Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/compare_40ep_against_existing.json").read_text())
print("required FFNO local-conv outputs parse successfully")
PY
```

Observed results:

- `tests/torch/test_ffno_bottleneck.py`: `5 passed in 4.63s`
- `tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`: `69 passed in 46.42s`
- `compileall`: exit `0`
- archived logs:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/verification/pytest_ffno_bottleneck_20260428T1600Z.log`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/verification/pytest_pdebench_image128_20260428T1600Z.log`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/verification/compileall_20260428T1600Z.log`

## Claim Boundary

- this is capped decision-support evidence only
- it does not satisfy the PDEBench full-training benchmark gate
- it does not promote the local-conv row into the primary benchmark bundle
- it does justify carrying `ffno_bottleneck_localconv_base` forward as the
  stronger local FFNO-family follow-up profile when future local CNS ablations
  need a repo-native FFNO variant with explicit local features

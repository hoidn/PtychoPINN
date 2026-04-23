# NeurIPS Hybrid ResNet PDEBench Author FFNO Equal-Footing Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-21-pdebench-author-ffno-equal-footing-cns`
- Date: `2026-04-22`
- Status: implementation complete; fresh author runs and merged cross-run compares complete
- Governing design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-pdebench-author-ffno-equal-footing-cns/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/`

This summary records the official-author FFNO integration path for the PDEBench
`2d_cfd_cns` local equal-footing lane and the fresh author-only capped runs
required by the backlog plan. The evidence from this item remains capped
decision-support evidence only. It does not create a benchmark-complete CNS
claim or a manuscript-facing FFNO result.

## Authoritative Source And Host Environment

The official authored FFNO source is pinned at:

- repo: `https://github.com/alasdairtran/fourierflow`
- commit: `82d1689e02c95f7656a7eaa1d88d0ab407b2483d`
- local clone: `.artifacts/external/fourierflow`
- provenance artifact:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author_ffno_source.json`

The local compare host remains:

- env: `ptycho311`
- python: `3.11.13`
- torch: `2.9.1+cu128`
- einops: `0.8.1`

The adapter uses the authored `FNOFactorized2DBlock` from
`fourierflow.modules.factorized_fno.grid_2d` with the authored config lineage
anchored by `experiments/torus_li/markov/24_layers/config.yaml`. The local
wrapper keeps the authored factorized-FNO body and adapts only the PDEBench CNS
I/O contract:

- input adaptation: append unit-interval `y/x` coordinate channels and reshape
  `B,C,H,W -> B,H,W,C`
- output adaptation: replace the scalar authored head with the same authored
  `WNLinear` family widened to the local four-channel CNS target
- contract boundary: the local runner keeps the existing PDEBench `2d_cfd_cns`
  split, normalization, training-loss, and reporting surfaces fixed

## Implemented Surface

New code:

- `scripts/studies/pdebench_image128/author_ffno_adapter.py`

Modified code:

- `scripts/studies/pdebench_image128/models.py`
- `scripts/studies/pdebench_image128/run_config.py`
- `scripts/studies/pdebench_image128/reporting.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`

Implemented contract:

- `author_ffno_cns_base` is a manual opt-in profile and stays out of the
  primary and readiness-required bundle lists
- the runner uses the fixed local equal-footing CNS recipe for this external
  baseline lane:
  - task: `2d_cfd_cns`
  - mode: `readiness`
  - `history_len=2`
  - trajectories: `512 / 64 / 64`
  - `max_windows_per_trajectory=8`
  - training loss: `mse`
  - optimizer: `Adam`
  - learning rate: `2e-4`
  - scheduler: `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, threshold=0.0)`
  - batch size: `4`
- the local cross-run compare now has a reusable collator that:
  - freezes the reference manifest with row-local contract fields
  - validates required artifact presence and contract parity row-by-row
  - writes merged `compare_10ep_against_existing.*` /
    `compare_40ep_against_existing.*` artifacts
  - renders the cross-run sample gallery only when saved sample targets align

## Verification Before Long Runs

Focused verification completed during the adapter/collator implementation:

- `pytest tests/studies/test_pdebench_image128_models.py -k 'author_ffno' -v`
- `pytest tests/studies/test_pdebench_image128_runner.py -k 'author_ffno' -v`
- `pytest tests/studies/test_pdebench_image128_runner.py -k 'reference_run_manifest or cross_run_compare' -v`

Observed results:

- author-FFNO model/profile tests: passed
- reference-manifest and cross-run compare tests: passed

## Smoke Gate

The bounded smoke gate completed successfully at:

- smoke run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/smoke-author-ffno-20260422T224011Z`

Smoke contract:

- task: `2d_cfd_cns`
- mode: `readiness`
- `history_len=2`
- trajectories: `8 / 2 / 2`
- `max_windows_per_trajectory=2`
- epochs: `1`
- batch size: `4`
- training loss: `mse`

Smoke outcome:

- `err_nRMSE=0.6850664616`
- `relative_l2=0.6850664616`
- `fRMSE_high=0.0336652398`
- runtime: `2.8037s`
- peak CUDA memory: `3410884096` bytes

This smoke result proves contract fit only. It does not support benchmark
interpretation.

## Frozen Existing Reference Rows

Reference manifest:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/reference_runs.json`

Required `10`-epoch rows:

- `spectral_resnet_bottleneck_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- `fno_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
- `unet_strong`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`

Optional `10`-epoch continuity rows:

- `hybrid_resnet_cns`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- `hybrid_resnet_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`

Required `40`-epoch rows:

- `spectral_resnet_bottleneck_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
- `fno_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- `unet_strong`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`

Optional `40`-epoch continuity row:

- `hybrid_resnet_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`

## Fresh Author Runs

### `10`-Epoch Author Row

Run root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-10ep-20260422T232119Z`

Contract:

- task: `2d_cfd_cns`
- mode: `readiness`
- `history_len=2`
- trajectories: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- train / val / test windows: `4096 / 512 / 512`
- epochs: `10`
- batch size: `4`
- training loss: `mse`

Observed author metrics:

- `err_RMSE=2.1226658821`
- `err_nRMSE=0.0878334790`
- `relative_l2=0.0878334790`
- `fRMSE_low=5.0589919090`
- `fRMSE_mid=0.0841526389`
- `fRMSE_high=0.2596977651`
- params: `1,073,672`
- runtime: `1328.7122s`
- peak CUDA memory: `3410884096` bytes

Training-loss trace:

- `[0.26541, 0.10737, 0.07187, 0.06255, 0.04460, 0.04532, 0.03354, 0.04378, 0.03082, 0.03033]`

### `10`-Epoch Cross-Run Compare

Merged artifacts:

- `compare_10ep_against_existing.json`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_10ep_against_existing.json`
- `compare_10ep_against_existing.csv`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_10ep_against_existing.csv`

Cross-run gallery:

- prediction:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_10ep_sample0.png`
- error:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_10ep_sample0_error.png`
- gallery status: rendered; no target-alignment blocker was recorded

Included rows:

- `author_ffno_cns_base`
- `spectral_resnet_bottleneck_base`
- `fno_base`
- `unet_strong`
- optional continuity:
  `hybrid_resnet_cns`, `hybrid_resnet_base`

Observed `10`-epoch compare metrics:

- `author_ffno_cns_base`
  - `err_nRMSE=0.0878334790`
  - `relative_l2=0.0878334790`
  - `fRMSE_high=0.2596977651`
- `spectral_resnet_bottleneck_base`
  - `err_nRMSE=0.0869938582`
  - `relative_l2=0.0869938582`
  - `fRMSE_high=0.6955373287`
- `fno_base`
  - `err_nRMSE=0.1063433066`
  - `relative_l2=0.1063433066`
  - `fRMSE_high=0.9280493259`
- `unet_strong`
  - `err_nRMSE=0.6222500205`
  - `relative_l2=0.6222500205`
  - `fRMSE_high=3.6293647289`
- `hybrid_resnet_cns`
  - `err_nRMSE=0.0944002941`
  - `relative_l2=0.0944002941`
  - `fRMSE_high=0.8000375628`
- `hybrid_resnet_base`
  - `err_nRMSE=0.1945149451`
  - `relative_l2=0.1945149451`
  - `fRMSE_high=1.0892750025`

Practical read:

- on aggregate denormalized error, the authored FFNO row is effectively tied
  with the capped shared-spectral row and better than the earlier local FNO,
  Hybrid, and U-Net rows
- on the saved sample target, the authored FFNO row aligned cleanly enough with
  the reused local rows for a cross-run gallery to render
- the authored FFNO row uses far fewer parameters than the local spectral row,
  but this is still capped decision-support evidence only

### `40`-Epoch Author Row

Run root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`

Contract:

- task: `2d_cfd_cns`
- mode: `readiness`
- `history_len=2`
- trajectories: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- train / val / test windows: `4096 / 512 / 512`
- epochs: `40`
- batch size: `4`
- training loss: `mse`

Observed author metrics:

- `err_RMSE=0.6802443266`
- `err_nRMSE=0.0281477310`
- `relative_l2=0.0281477310`
- `fRMSE_low=1.6124732494`
- `fRMSE_mid=0.0759288296`
- `fRMSE_high=0.1210141182`
- params: `1,073,672`
- runtime: `4725.5117s`
- peak CUDA memory: `3410884096` bytes

Training-loss trace:

- `[0.26541, 0.10737, 0.07187, 0.06255, 0.04460, 0.04532, 0.03354, 0.04378, 0.03082, 0.03033, 0.02403, 0.03023, 0.02126, 0.02102, 0.02660, 0.01719, 0.01865, 0.01633, 0.01631, 0.01752, 0.01934, 0.01350, 0.01766, 0.01283, 0.01369, 0.01600, 0.01313, 0.00741, 0.00642, 0.00640, 0.00597, 0.00623, 0.00650, 0.00652, 0.00397, 0.00360, 0.00341, 0.00356, 0.00368, 0.00337]`

### `40`-Epoch Cross-Run Compare

Merged artifacts:

- `compare_40ep_against_existing.json`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_40ep_against_existing.json`
- `compare_40ep_against_existing.csv`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_40ep_against_existing.csv`

Cross-run gallery:

- prediction:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_40ep_sample0.png`
- error:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_40ep_sample0_error.png`
- gallery status: rendered; no target-alignment blocker was recorded

Included rows:

- `author_ffno_cns_base`
- `spectral_resnet_bottleneck_base`
- `fno_base`
- `unet_strong`
- optional continuity:
  `hybrid_resnet_base`

Observed `40`-epoch compare metrics:

- `author_ffno_cns_base`
  - `err_nRMSE=0.0281477310`
  - `relative_l2=0.0281477310`
  - `fRMSE_high=0.1210141182`
- `spectral_resnet_bottleneck_base`
  - `err_nRMSE=0.0615620054`
  - `relative_l2=0.0615620054`
  - `fRMSE_high=0.4349334538`
- `fno_base`
  - `err_nRMSE=0.0740992129`
  - `relative_l2=0.0740992129`
  - `fRMSE_high=0.6717720628`
- `unet_strong`
  - `err_nRMSE=0.6757976413`
  - `relative_l2=0.6757976413`
  - `fRMSE_high=1.3326253891`
- `hybrid_resnet_base`
  - `err_nRMSE=0.1324555874`
  - `relative_l2=0.1324555874`
  - `fRMSE_high=0.7680469155`

Practical read:

- on aggregate denormalized error, the authored FFNO row clearly beats the
  capped shared-spectral, local FNO, Hybrid, and U-Net rows on this `40`-epoch
  slice
- the authored FFNO row also posts the lowest saved-sample `fRMSE_high` among
  the included rows in the merged `40`-epoch compare
- the authored FFNO row again aligned cleanly enough with the reused local rows
  for the cross-run gallery to render

## Fairness Caveats

- This item compares the authored FFNO model under the repo's fixed local CNS
  equal-footing contract. It does not claim paper-default FFNO performance.
- `author_ffno_cns_base` and `ffno_bottleneck_base` are different rows:
  - `author_ffno_cns_base`: external authored model wrapped to the local CNS
    contract
  - `ffno_bottleneck_base`: local FFNO-close bottleneck variant inside the
    repo's canonical CNS shell
- All results from this item remain capped decision-support evidence only.

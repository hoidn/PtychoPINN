# PDEBench CNS Markov History-1 Compare Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-21-pdebench-cns-markov-history1-compare`
- Date: `2026-04-27`
- Status: implementation complete; frozen history-2 audit, missing hybrid anchor backfill, fresh history-1 pilot compares, and cross-history sidecars are all recorded
- Governing design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-pdebench-cns-markov-history1-compare/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/`

This summary records capped decision-support evidence only. It does not create a
benchmark-complete CNS ranking, a rollout/autoregressive result, or a
paper-facing artifact under `/home/ollie/Documents/neurips/`.

## Fixed Compare Contract

The fresh history-1 rows and frozen history-2 anchors kept the local capped CNS
surface fixed everywhere except temporal context:

- task: `2d_cfd_cns`
- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- trajectories: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- batch size: `4`
- training loss: `mse`
- metric family:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`,
  `fRMSE_high`
- allowed contract delta only:
  - frozen reference: `history_len=2`, `concat u[t-2:t] -> u[t]`,
    `input_channels=8`, `target_channels=4`
  - fresh run: `history_len=1`, `concat u[t-1:t] -> u[t]`,
    `input_channels=4`, `target_channels=4`

The frozen manifest is:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2_reference_runs.json`

The compare payloads enforce the rule:

- comparison standard: `Only history_len and its derived sample/input-channel contract may differ.`

Cross-run gallery rendering stayed non-fatal and required exact target
alignment under `np.allclose(..., atol=1e-6, rtol=1e-6)`.

## Recorded Artifacts

Frozen history-2 anchors:

- `10ep` manifest bucket:
  - spectral / hybrid:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
  - FNO / U-Net:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
- `40ep` manifest bucket:
  - spectral:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
  - backfilled `hybrid_resnet_cns`:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z`
  - FNO / U-Net:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`

Fresh history-1 pilot runs:

- `10ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-10ep-20260423T224907Z`
- `40ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-40ep-20260423T230352Z`

Cross-history compare sidecars:

- `10ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/compare_10ep_against_history2.json`
  and `.csv`
- `40ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/compare_40ep_against_history2.json`
  and `.csv`

Both compare payloads recorded `cross_run_gallery_blocked.reason =
"target_mismatch"`, so no merged prediction/error gallery was emitted. That is
an artifact-alignment limitation only, not a run failure.

## Results

### `10` Epochs

Frozen history-2 ranking by `err_nRMSE`:

1. `spectral_resnet_bottleneck_base` - `0.0869938582`
2. `hybrid_resnet_cns` - `0.0944002941`
3. `fno_base` - `0.1063433066`
4. `unet_strong` - `0.6222500205`

Fresh history-1 ranking by `err_nRMSE`:

1. `spectral_resnet_bottleneck_base` - `0.1139530987`
2. `hybrid_resnet_cns` - `0.1446415037`
3. `fno_base` - `0.1608743966`
4. `unet_strong` - `0.6098560691`

Directional answer:

- `spectral_resnet_bottleneck_base` did not improve under `history_len=1`;
  `err_nRMSE` worsened by `+0.0269592404`
  (`0.0869938582 -> 0.1139530987`)
- the four-row ranking did not change
- `hybrid_resnet_cns` worsened by `+0.0502412096`
- `fno_base` worsened by `+0.0545310900`
- `unet_strong` improved slightly by `-0.0123939514`

### `40` Epochs

Frozen history-2 ranking by `err_nRMSE`:

1. `spectral_resnet_bottleneck_base` - `0.0615620054`
2. `hybrid_resnet_cns` - `0.0644183308`
3. `fno_base` - `0.0740992129`
4. `unet_strong` - `0.6757976413`

Fresh history-1 ranking by `err_nRMSE`:

1. `spectral_resnet_bottleneck_base` - `0.0998256728`
2. `hybrid_resnet_cns` - `0.1063092798`
3. `fno_base` - `0.1106384769`
4. `unet_strong` - `0.6264002323`

Directional answer:

- `spectral_resnet_bottleneck_base` again did not improve under
  `history_len=1`; `err_nRMSE` worsened by `+0.0382636674`
  (`0.0615620054 -> 0.0998256728`)
- the four-row ranking again did not change
- `hybrid_resnet_cns` worsened by `+0.0418909490`
- `fno_base` worsened by `+0.0365392640`
- `unet_strong` improved by `-0.0493974090`

## Interpretation

This backlog item answers the intended scientific question narrowly:

- lowering temporal context from `history_len=2` to `history_len=1` did not
  help the spectral bottleneck row at either budget
- the lower-context Markov contract did not reorder the local four-row capped
  ranking at either budget
- on this fixed capped contract, the lower-context variant mostly trades away
  aggregate denormalized accuracy for the stronger rows and only helps the
  weakest local U-Net row

This should remain summary-local rather than a reusable repo-wide finding for
now because the result is still limited to one capped PDEBench CNS slice, two
epoch budgets, and one fixed local recipe family.

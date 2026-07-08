# CDI Hybrid-Spectral to FFNO Parameter-Space Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-27-hybrid-spectral-ffno-parameter-space-cdi`
- Date: `2026-05-01`
- Status: implementation complete with refreshed closeout verification and
  repaired collated model-manifest bundle; CDI-only decision-support evidence
  with no paper promotion
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/execution_plan.md`
- Artifact root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi`
- Preflight note: `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_hybrid_spectral_ffno_parameter_space_preflight.md`

This item stays inside the opened Phase 2/Phase 3 parallel gate but remains
Phase 3 CDI work. It does not satisfy any remaining PDEBench requirement and
does not change the current paper-grade `lines128` CDI authority.

## Fixed Contract

- shared contract: synthetic grid-lines, `N=128`, `gridsize=1`, `set_phi=True`,
  custom Run1084 probe, `pad_extrapolate`, `nimgs_train=2`, `nimgs_test=2`,
  `seed=3`, `torch_epochs=40`, `torch_learning_rate=2e-4`,
  `ReduceLROnPlateau`, `torch_loss_mode=mae`, `torch_output_mode=real_imag`,
  `fno_modes=12`, `fno_width=32`, `fno_blocks=4`, `fno_cnn_blocks=2`
- reused anchors from the authoritative complete-table root:
  - `pinn_hybrid_resnet`
  - `pinn_spectral_resnet_bottleneck_net`
  - `pinn_ffno`
- fresh bridge rows:
  - `pinn_spectral_resnet_bottleneck_ds1`
  - `pinn_spectral_resnet_bottleneck_linear_decoder`
  - `pinn_hybrid_resnet_ffno_bottleneck`
- frozen machine-readable authorities:
  - `preflight/study_matrix.json`
  - `preflight/reference_runs.json`
  - `model_manifest.json`
  - `analysis/bundle_validation.json`
    - refreshed to carry the stricter closeout fields
      `fresh_row_completion_failures`, `missing_merged_outputs`,
      `merged_output_failures`, canonical `probe_npz` digest validation, and
      required collated model-manifest presence

## Per-Axis Results

### Encoder / Downsampling Bridge

- nearest-anchor compare:
  `pinn_spectral_resnet_bottleneck_net -> pinn_spectral_resnet_bottleneck_ds1`
- amplitude regressed while phase improved:
  - amp MAE: `0.0249436181 -> 0.0354603752`
  - phase MAE: `0.0928813860 -> 0.0643633276`
  - amp SSIM: `0.9898549878 -> 0.9799357611`
  - phase SSIM: `0.9722190667 -> 0.9939341632`
- reused `pinn_ffno` endpoint compare:
  - amp MAE: `0.0627724752 -> 0.0354603752`
  - phase MAE: `0.0828386688 -> 0.0643633276`
  - amp SSIM: `0.9348303400 -> 0.9799357611`
  - phase SSIM: `0.9815915191 -> 0.9939341632`

Interpretation:

- reducing the downsampling depth created a real phase-leaning trade rather
  than a clean improvement
- the row did not dominate either existing shell anchor: the spectral anchor
  kept the best amplitude fidelity, while the Hybrid anchor kept the stronger
  balanced local-shell reference point
- relative to the reused `pinn_ffno` endpoint, the DS1 row is clearly stronger
  on this fixed CDI contract, but that does not change the study's no-promotion
  boundary
- this row remains decision-support context only

### Decoder Bridge

- nearest-anchor compare:
  `pinn_spectral_resnet_bottleneck_net -> pinn_spectral_resnet_bottleneck_linear_decoder`
- the lightweight bilinear-plus-`1x1` decoder regressed sharply:
  - amp MAE: `0.0249436181 -> 0.1347432733`
  - phase MAE: `0.0928813860 -> 0.1169255985`
  - amp SSIM: `0.9898549878 -> 0.7234640576`
  - phase SSIM: `0.9722190667 -> 0.9042721953`
- reused `pinn_ffno` endpoint compare:
  - amp MAE: `0.0627724752 -> 0.1347432733`
  - phase MAE: `0.0828386688 -> 0.1169255985`
  - amp SSIM: `0.9348303400 -> 0.7234640576`
  - phase SSIM: `0.9815915191 -> 0.9042721953`

Interpretation:

- on the fixed CDI contract, this decoder swap is not viable as an
  FFNO-adjacent bridge
- the existing transpose-decoder spectral shell remains decisively better
- the row also underperforms the reused `pinn_ffno` endpoint, so this decoder
  family should not be the next CDI bridge target

### Bottleneck Bridge

- nearest-anchor compare:
  `pinn_hybrid_resnet -> pinn_hybrid_resnet_ffno_bottleneck`
- replacing only the Hybrid bottleneck with the FFNO bottleneck did not help:
  - amp MAE: `0.0269394740 -> 0.0311762355`
  - phase MAE: `0.0720634771 -> 0.0875998761`
  - amp SSIM: `0.9881142974 -> 0.9835814654`
  - phase SSIM: `0.9947399866 -> 0.9909663579`
- reused `pinn_ffno` endpoint compare:
  - amp MAE: `0.0627724752 -> 0.0311762355`
  - phase MAE: `0.0828386688 -> 0.0875998761`
  - amp SSIM: `0.9348303400 -> 0.9835814654`
  - phase SSIM: `0.9815915191 -> 0.9909663579`

Interpretation:

- the FFNO bottleneck alone is not a better same-shell replacement for the
  current Hybrid anchor on this CDI slice
- the result weakens the case that the local FFNO endpoint should be approached
  first by swapping only the bottleneck family
- relative to the reused `pinn_ffno` endpoint, this row improves amplitude
  fidelity and both SSIMs but gives back phase MAE, so it is still a mixed
  bridge rather than a clean FFNO-path improvement

## Carry-Forward Result

- no fresh bridge row earned promotion or displaced the existing paper-grade
  `lines128` CDI rows
- `pinn_spectral_resnet_bottleneck_net` remains the best amplitude-fidelity
  local bridge anchor in this study
- `pinn_hybrid_resnet` remains the stronger balanced local shell reference
- `pinn_ffno` remains an endpoint comparator, not a promoted bridge target
- the strongest fresh-row signal is the DS1 phase trade, but it is not strong
  enough to justify paper-facing CDI promotion or a new default bridge profile

## Provenance And Recovery Note

- the first tmux-backed launcher exposed a runner bug: row-local
  `model_id_override` was not being used for the top-level recon output path
- that bug caused the fresh DS1 row to save under the reused spectral-anchor
  recon path when reused rows were materialized by symlink
- the runner was fixed so recon artifacts and paper payloads now respect the
  override model ID, the overwritten reused spectral recon was restored from a
  clean sibling authoritative root, the DS1 recon was preserved under its own
  row ID, and the remaining fresh rows were resumed successfully
- the repaired harness now uses copy-on-write materialization for reused
  anchors, validates those copied reused rows against the frozen authoritative
  digests before any fresh launch, scrubs failed or incomplete fresh-row
  leftovers before relaunch, records an explicit fresh-row contract projection
  in the frozen preflight matrix, and aborts if final bundle validation reports
  reused-root drift, missing required row artifacts, fresh-row
  invocation/config contract drift, failed fresh-row completion proof, or
  missing/malformed merged comparison outputs
- the closeout repair also persisted the previously missing top-level
  `model_manifest.json` from the collated row payloads and upgraded the bundle
  validator so the archived study root now fails closed if that manifest is
  absent or drifts from the frozen six-row roster

## Verification

Commands run from `/home/ollie/Documents/PtychoPINN`:

```bash
pytest -q tests/studies/test_cdi_hybrid_spectral_ffno_parameter_space.py
pytest -q tests/test_grid_lines_compare_wrapper.py
pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
python -m compileall -q ptycho_torch scripts/studies
```

Observed results:

- final review-fix harness selector: `21 passed in 3.64s`
- focused compare-wrapper selector:
  `63 passed, 23 warnings in 17.75s`
- targeted closeout selector:
  `192 passed, 49 warnings in 303.88s (0:05:03)`
- `compileall`: exit `0`
- final study launcher proof:
  `logs/launcher_resume.log` ends with `__EXIT__:0`
- review-fix deterministic bundle validation:
  `analysis/bundle_validation.json` and
  `verification/artifact_validation_review_fix8.log` report `"ok": true`,
  preserve
  `claim_boundary: "no_paper_promotion_without_later_authority"`, and confirm
  the repaired top-level `model_manifest.json`

Archived logs for this pass live under:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/logs/`
- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/verification/`
  - including `verification/pytest_study_harness_review_fix8.log`,
    `verification/pytest_grid_compare_wrapper_review_fix8.log`,
    `verification/pytest_backlog_checks_review_fix8.log`,
    `verification/compileall_review_fix8.log`, and
    `verification/artifact_validation_review_fix8.log` for the archived bundle
    repair

## Boundary

- this summary is CDI-only decision-support evidence
- it does not alter the paper-grade `lines128` complete-table authority
- it does not promote any fresh bridge row into manuscript claim territory

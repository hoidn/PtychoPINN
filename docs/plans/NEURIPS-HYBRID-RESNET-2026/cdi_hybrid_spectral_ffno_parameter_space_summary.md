# CDI Hybrid-Spectral to FFNO Parameter-Space Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-27-hybrid-spectral-ffno-parameter-space-cdi`
- Date: `2026-04-30`
- Status: implementation complete; CDI-only decision-support evidence with no paper promotion
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
  - `analysis/bundle_validation.json`

## Per-Axis Results

### Encoder / Downsampling Bridge

- nearest-anchor compare:
  `pinn_spectral_resnet_bottleneck_net -> pinn_spectral_resnet_bottleneck_ds1`
- amplitude regressed while phase improved:
  - amp MAE: `0.0249436181 -> 0.0354603752`
  - phase MAE: `0.0928813860 -> 0.0643633276`
  - amp SSIM: `0.9898549878 -> 0.9799357611`
  - phase SSIM: `0.9722190667 -> 0.9939341632`

Interpretation:

- reducing the downsampling depth created a real phase-leaning trade rather
  than a clean improvement
- the row did not dominate either existing shell anchor: the spectral anchor
  kept the best amplitude fidelity, while the Hybrid anchor kept the stronger
  balanced local-shell reference point
- this row remains decision-support context only

### Decoder Bridge

- nearest-anchor compare:
  `pinn_spectral_resnet_bottleneck_net -> pinn_spectral_resnet_bottleneck_linear_decoder`
- the lightweight bilinear-plus-`1x1` decoder regressed sharply:
  - amp MAE: `0.0249436181 -> 0.1347432733`
  - phase MAE: `0.0928813860 -> 0.1169255985`
  - amp SSIM: `0.9898549878 -> 0.7234640576`
  - phase SSIM: `0.9722190667 -> 0.9042721953`

Interpretation:

- on the fixed CDI contract, this decoder swap is not viable as an
  FFNO-adjacent bridge
- the existing transpose-decoder spectral shell remains decisively better

### Bottleneck Bridge

- nearest-anchor compare:
  `pinn_hybrid_resnet -> pinn_hybrid_resnet_ffno_bottleneck`
- replacing only the Hybrid bottleneck with the FFNO bottleneck did not help:
  - amp MAE: `0.0269394740 -> 0.0311762355`
  - phase MAE: `0.0720634771 -> 0.0875998761`
  - amp SSIM: `0.9881142974 -> 0.9835814654`
  - phase SSIM: `0.9947399866 -> 0.9909663579`

Interpretation:

- the FFNO bottleneck alone is not a better same-shell replacement for the
  current Hybrid anchor on this CDI slice
- the result weakens the case that the local FFNO endpoint should be approached
  first by swapping only the bottleneck family

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

## Verification

Commands run from `/home/ollie/Documents/PtychoPINN`:

```bash
pytest -q tests/torch/test_grid_lines_torch_runner.py::TestRunGridLinesTorchScaffold::test_runner_writes_recon_artifact_under_model_id_override
pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
python -m compileall -q ptycho_torch scripts/studies
pytest -v -m integration
python scripts/studies/runbooks/run_cdi_hybrid_spectral_ffno_parameter_space.py \
  --authoritative-root .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux \
  --output-root .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi \
  --preflight-root .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/preflight \
  --note-path docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_hybrid_spectral_ffno_parameter_space_preflight.md \
  --preflight-only
```

Observed results:

- override regression selector: `1 passed in 5.69s`
- targeted closeout selector:
  `191 passed, 49 warnings in 305.67s (0:05:05)`
- `compileall`: exit `0`
- `pytest -v -m integration`:
  `5 passed, 4 skipped, 1800 deselected, 2 warnings in 302.76s (0:05:02)`
- final study launcher proof:
  `logs/launcher_resume.log` ends with `__EXIT__:0`
- deterministic bundle validation:
  `analysis/bundle_validation.json` reports `"ok": true`

Archived logs for this pass live under:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/logs/`
- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/verification/`

## Boundary

- this summary is CDI-only decision-support evidence
- it does not alter the paper-grade `lines128` complete-table authority
- it does not promote any fresh bridge row into manuscript claim territory

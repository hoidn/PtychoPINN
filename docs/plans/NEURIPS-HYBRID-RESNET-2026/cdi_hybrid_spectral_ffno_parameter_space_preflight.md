# CDI Hybrid-Spectral to FFNO Parameter-Space Preflight

- Scope: CDI-only decision-support evidence under the opened Phase 2/Phase 3 parallel gate.
- Phase accounting: this remains Phase 3 CDI work and does not satisfy remaining Phase 2 PDEBench requirements.
- Claim boundary: no reused or fresh row is paper-facing without a later checked-in promotion authority.
- Authoritative reused-anchor root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Frozen study matrix: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/preflight/study_matrix.json`
- Frozen reference-run manifest: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/preflight/reference_runs.json`

## Fixed Contract

- `N=128`, `gridsize=1`, `set_phi=True`, `seed=3`
- probe: `custom`, `pad_extrapolate`, `sigma=0.5`
- split: `nimgs_train=2`, `nimgs_test=2`, `nphotons=1000000000.0`
- training: `epochs=40`, `lr=0.0002`, `ReduceLROnPlateau`, `plateau_factor=0.5`, `plateau_patience=2`, `plateau_min_lr=0.0001`, `plateau_threshold=0.0`, `loss=mae`, `output=real_imag`
- spectral shell: `fno_modes=12`, `fno_width=32`, `fno_blocks=4`, `fno_cnn_blocks=2`

## Output-root layout

- Live study root layout:
  - `runs/<model_id>/` under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi`
  - `recons/<model_id>/` under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi`
- Display labels are frozen per row and must be reused in summaries and collated outputs.

## Reused Anchors

### `pinn_hybrid_resnet`

- Display label: `Hybrid ResNet + PINN`
- Architecture: `hybrid_resnet`
- Expression path: authoritative row copy from the fixed complete-table bundle
- Run dir: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/runs/pinn_hybrid_resnet`
- Recon path: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/recons/pinn_hybrid_resnet/recon.npz`
- Reuse acceptability: accepted from a completed row-local launcher record with git/runtime provenance

### `pinn_spectral_resnet_bottleneck_net`

- Display label: `Spectral ResNet Bottleneck + PINN`
- Architecture: `spectral_resnet_bottleneck_net`
- Expression path: authoritative row copy from the fixed complete-table bundle
- Run dir: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/runs/pinn_spectral_resnet_bottleneck_net`
- Recon path: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/recons/pinn_spectral_resnet_bottleneck_net/recon.npz`
- Reuse acceptability: accepted from a completed row-local launcher record with git/runtime provenance

### `pinn_ffno`

- Display label: `FFNO-local proxy + PINN` (`fno_cnn_blocks=2`; corrected
  no-refiner rerun pending for pure-FFNO claims)
- Architecture: `ffno`
- Expression path: authoritative row copy from the fixed complete-table bundle
- Run dir: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/runs/pinn_ffno`
- Recon path: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/recons/pinn_ffno/recon.npz`
- Reuse acceptability: accepted from a completed promotion/backfill record with parent invocation and recovered exit-code proof

## Fresh Bridge Rows

### `pinn_spectral_resnet_bottleneck_ds1`

- Display label: `Spectral ResNet Bottleneck DS1 + PINN`
- Nearest anchor: `pinn_spectral_resnet_bottleneck_net`
- Architecture: `spectral_resnet_bottleneck_net`
- Expression path: runner override on `spectral_resnet_bottleneck_net`
- Output run dir: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/runs/pinn_spectral_resnet_bottleneck_ds1`
- Output recon dir: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/recons/pinn_spectral_resnet_bottleneck_ds1`
- Overrides: `{"hybrid_downsample_steps": 1}`

### `pinn_spectral_resnet_bottleneck_linear_decoder`

- Display label: `Spectral ResNet Linear Decoder + PINN`
- Nearest anchor: `pinn_spectral_resnet_bottleneck_net`
- Architecture: `spectral_resnet_bottleneck_linear_decoder`
- Expression path: generator registry entry `spectral_resnet_bottleneck_linear_decoder`
- Output run dir: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/runs/pinn_spectral_resnet_bottleneck_linear_decoder`
- Output recon dir: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/recons/pinn_spectral_resnet_bottleneck_linear_decoder`
- Overrides: `{}`

### `pinn_hybrid_resnet_ffno_bottleneck`

- Display label: `Hybrid ResNet FFNO Bottleneck + PINN`
- Nearest anchor: `pinn_hybrid_resnet`
- Architecture: `hybrid_resnet_ffno_bottleneck`
- Expression path: generator registry entry `hybrid_resnet_ffno_bottleneck`
- Output run dir: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/runs/pinn_hybrid_resnet_ffno_bottleneck`
- Output recon dir: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/recons/pinn_hybrid_resnet_ffno_bottleneck`
- Overrides: `{}`

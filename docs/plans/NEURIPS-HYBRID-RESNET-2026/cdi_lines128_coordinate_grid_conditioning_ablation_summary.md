# CDI Lines128 Coordinate-Grid Conditioning Ablation Summary

- Date: `2026-05-09`
- Backlog item: `2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation/execution_plan.md`
- State: `decision_support_append_only`
- Fixed contract id: `cdi_lines128_seed3`
- Claim boundary: `cdi_lines128_coordinate_grid_conditioning_ablation_only`
- Governing design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Item root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation/`

## Objective

- Add two append-only CDI `lines128` rows that keep the authoritative
  unconditioned dataset/optimizer/probe/epoch contract fixed while appending
  deterministic unit-interval `[y, x]` coordinate channels to the learned-model
  input tensor only.
- Execute only:
  - `pinn_hybrid_resnet_grid_channels`
  - `pinn_ffno_grid_channels`
- Compare those fresh rows only against their same-contract lineage anchors:
  - `pinn_hybrid_resnet` from
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - corrected pure `pinn_ffno` from
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md`

## Claim Boundary

- This item is append-only input-conditioning evidence only.
- It does not replace the immutable six-row CDI authority or the current
  paper-local FFNO refresh.
- The only intended row-level deltas were:
  - `input_conditioning_mode: diffraction_only -> coordinate_grid`
  - `learned_input_channels: 1 -> 3`
  - for the FFNO conditioned row, `fno_cnn_blocks` remained aligned with the
    corrected pure-FFNO authority at `0`
- Coordinate-grid conditioning was not combined with probe-channel conditioning
  in the same row.

## Fixed Contract And Conditioning Proof

- Contract note:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation/verification/contract_note.json`
- Comparison standard: exact equality on the fixed-contract fields recorded in
  `contract_note.json`; only the allowed row-level deltas above are permitted.
- Coordinate convention held constant across both rows:
  - `coordinate_channels = ["y", "x"]`
  - `coordinate_value_range = [0.0, 1.0]`
  - `coordinate_meshgrid_indexing = "ij"`
  - `coordinate_spatial_shape = [128, 128]`
  - `y` built from `np.linspace(0.0, 1.0, 128)`
  - `x` built from `np.linspace(0.0, 1.0, 128)`
  - channels appended onto the diffraction tensor in `[y, x]` order
- Fresh row configs record (from `runs/<row>/config.json` `input_conditioning`):
  - `mode = coordinate_grid`
  - `enabled = true`
  - `base_input_channels = 1`
  - `learned_input_channels = 3`
  - `coordinate_channel_count = 2`
- Row-local completion proofs exist for both rows:
  - `runs/pinn_hybrid_resnet_grid_channels/launcher_completion.json`
  - `runs/pinn_hybrid_resnet_grid_channels/exit_code_proof.json`
  - `runs/pinn_ffno_grid_channels/launcher_completion.json`
  - `runs/pinn_ffno_grid_channels/exit_code_proof.json`

## Row Outcomes

### `pinn_hybrid_resnet_grid_channels` vs `pinn_hybrid_resnet`

- Fresh row root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation/runs/pinn_hybrid_resnet_grid_channels`
- Baseline root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/runs/pinn_hybrid_resnet`

| Metric | Baseline | Coordinate-grid-conditioned | Delta |
|---|---:|---:|---:|
| amp MAE | 0.026939 | 0.025132 | -0.001807 |
| phase MAE | 0.072063 | 3.014876 | +2.942813 |
| amp SSIM | 0.988114 | 0.989999 | +0.001885 |
| phase SSIM | 0.994740 | 0.186289 | -0.808451 |
| amp FRC50 | 135.464222 | 138.408316 | +2.944094 |
| phase FRC50 | 106.800609 | 49.615425 | -57.185184 |

- Fresh train loss `0.024465`; fresh validation loss `0.029676`.
- Read: the conditioned Hybrid row marginally improves amplitude metrics over
  the authoritative Hybrid baseline but catastrophically regresses phase-side
  metrics (`phase_ssim` drops from `0.9947` to `0.1863`, `phase_frc50` from
  `106.8` to `49.6`).

### `pinn_ffno_grid_channels` vs corrected pure `pinn_ffno`

- Fresh row root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation/runs/pinn_ffno_grid_channels`
- Baseline root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/runs/ffno_no_refiner_20260506T223454Z/runs/pinn_ffno`

| Metric | Baseline | Coordinate-grid-conditioned | Delta |
|---|---:|---:|---:|
| amp MAE | 0.082043 | 0.069789 | -0.012254 |
| phase MAE | 0.137965 | 1.682742 | +1.544778 |
| amp SSIM | 0.890305 | 0.914300 | +0.023995 |
| phase SSIM | 0.959644 | 0.384336 | -0.575308 |
| amp FRC50 | 61.465010 | 60.647149 | -0.817861 |
| phase FRC50 | 58.728607 | 27.273829 | -31.454777 |

- Fresh train loss `0.063318`; fresh validation loss `0.069754`.
- Read: the conditioned pure-FFNO row improves amplitude-side metrics versus
  the corrected pure-FFNO anchor, but phase quality collapses
  (`phase_ssim 0.9596 -> 0.3843`, `phase_frc50 58.7 -> 27.3`), so the row is
  not promotable.

## Interpretation

- On this locked single-seed CDI contract, appending deterministic
  unit-interval `[y, x]` coordinate channels to the learned input did not
  produce a promotable improvement for either architecture.
- The Hybrid coordinate-grid row very slightly improves amplitude metrics
  versus the authoritative Hybrid baseline but collapses phase reconstruction
  badly enough that it cannot replace the canonical Hybrid row.
- The FFNO coordinate-grid row improves amplitude-side metrics versus the
  corrected pure-FFNO anchor but again loses phase quality at a magnitude that
  rules out paper-local promotion.
- Coordinate-grid conditioning did not help Hybrid ResNet/SRU-Net, did not
  help FFNO, and the effect is not strong enough to justify a later
  table-refresh item.

## Verification

Verification logs are archived under
`.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation/verification/`.

- Prerequisite gate:
  - `prerequisite_check.log`
- Deterministic code gates:
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno or coordinate or grid"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno or coordinate or grid"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Saved outputs:
  - `preflight_payload.json`
  - `contract_note.json`
  - `live_stdout.log`
  - `live_stderr.log`
- Long-run completion proof:
  - tracked launcher PID `3658772` exited `0` at `2026-05-09T08:19:22Z`
  - row-local exit-code proofs and `launcher_completion.json` files listed
    under "Fixed Contract And Conditioning Proof"

## Residual Risks

- This remains a single-seed, `2 / 2` train/test CDI ablation. The result is
  useful for same-contract directionality, not for broad statistical claims.
- Both conditioned rows show the same qualitative pattern (modest amplitude
  gain, severe phase collapse) as the previously completed probe-channel
  conditioning ablation. The directional signal is consistent but does not
  imply broad claims beyond this locked CDI contract.
- FFNO evaluation emitted `Negative SSIM` clamp warnings during MS-SSIM
  computation, but the tracked launcher PID still exited `0` and the required
  row-local artifacts were written successfully.

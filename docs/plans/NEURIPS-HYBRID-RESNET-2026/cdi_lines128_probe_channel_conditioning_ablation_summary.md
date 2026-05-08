# CDI Lines128 Probe-Channel Conditioning Ablation Summary

- Date: `2026-05-08`
- Backlog item: `2026-05-07-cdi-lines128-probe-channel-conditioning-ablation`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/execution_plan.md`
- State: `decision_support_append_only`
- Fixed contract id: `cdi_lines128_seed3`
- Claim boundary: `cdi_lines128_probe_channel_conditioning_ablation_only`
- Governing design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Item root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/`

## Objective

- Add two append-only CDI `lines128` rows that keep the authoritative
  unconditioned dataset/optimizer/probe/epoch contract fixed while appending
  the fixed Run1084 probe real and imaginary channels to the learned-model
  input tensor only.
- Execute only:
  - `pinn_hybrid_resnet_probe_channels`
  - `pinn_ffno_probe_channels`
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
  - `input_conditioning_mode: diffraction_only -> probe_real_imag`
  - `learned_input_channels: 1 -> 3`
  - for the FFNO conditioned row, `fno_cnn_blocks` remained aligned with the
    corrected pure-FFNO authority at `0`

## Fixed Contract And Conditioning Proof

- Contract note:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/verification/contract_note.json`
- Comparison standard: exact equality on the fixed-contract fields recorded in
  `contract_note.json`; only the allowed row-level deltas above are permitted.
- Fixed probe lineage held constant across both rows:
  - `probe_source = custom`
  - `probe_npz = datasets/Run1084_recon3_postPC_shrunk_3.npz`
  - `probe_scale_mode = pad_extrapolate`
  - `probe_smoothing_sigma = 0.5`
  - `probe_transform_pipeline = pad_extrapolate:128|smooth:0.5`
- Fresh row configs record:
  - `mode = probe_real_imag`
  - `enabled = true`
  - `base_input_channels = 1`
  - `learned_input_channels = 3`
  - `probe_channel_count = 2`
  - `probe_channels = ["probe_real", "probe_imag"]`
- Row-local completion proofs exist for both rows:
  - `runs/pinn_hybrid_resnet_probe_channels/launcher_completion.json`
  - `runs/pinn_hybrid_resnet_probe_channels/exit_code_proof.json`
  - `runs/pinn_ffno_probe_channels/launcher_completion.json`
  - `runs/pinn_ffno_probe_channels/exit_code_proof.json`

## Row Outcomes

### `pinn_hybrid_resnet_probe_channels` vs `pinn_hybrid_resnet`

- Fresh row root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/runs/pinn_hybrid_resnet_probe_channels`
- Baseline root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/runs/pinn_hybrid_resnet`

| Metric | Baseline | Probe-conditioned | Delta |
|---|---:|---:|---:|
| amp MAE | 0.026939 | 0.028848 | +0.001909 |
| phase MAE | 0.072063 | 0.131669 | +0.059606 |
| amp SSIM | 0.988114 | 0.987296 | -0.000818 |
| phase SSIM | 0.994740 | 0.961703 | -0.033037 |
| amp FRC50 | 135.464222 | 132.825383 | -2.638839 |
| phase FRC50 | 106.800609 | 0.929580 | -105.871029 |

- Fresh train loss `0.025675`; fresh validation loss `0.032059`.
- Read: the conditioned Hybrid row does not improve over the authoritative
  Hybrid baseline and materially regresses the phase-side metrics, especially
  phase FRC50.

### `pinn_ffno_probe_channels` vs corrected pure `pinn_ffno`

- Fresh row root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/runs/pinn_ffno_probe_channels`
- Baseline root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/runs/ffno_no_refiner_20260506T223454Z/runs/pinn_ffno`

| Metric | Baseline | Probe-conditioned | Delta |
|---|---:|---:|---:|
| amp MAE | 0.082043 | 0.073340 | -0.008703 |
| phase MAE | 0.137965 | 3.154161 | +3.016196 |
| amp SSIM | 0.890305 | 0.907495 | +0.017190 |
| phase SSIM | 0.959644 | 0.012315 | -0.947329 |
| amp FRC50 | 61.465010 | 57.675396 | -3.789614 |
| phase FRC50 | 58.728607 | 48.611318 | -10.117289 |

- Fresh train loss `0.064114`; fresh validation loss `0.063928`.
- Read: the conditioned pure-FFNO row improves a few amplitude-side metrics,
  but phase quality collapses badly enough that the row is not promotable.

## Interpretation

- On this locked single-seed CDI contract, appending fixed-probe real/imag
  channels to the learned input did not produce a promotable improvement for
  either architecture.
- The Hybrid conditioning row regresses relative to the authoritative Hybrid
  baseline on the balanced reconstruction metrics that already justified the
  baseline's status.
- The FFNO conditioning row shows mixed amplitude movement but catastrophic
  phase degradation, so it cannot replace the corrected pure-FFNO row for any
  paper-local or default claim.

## Verification

Verification logs are archived under
`.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/verification/`.

- Prerequisite gate:
  - `prerequisite_check.log`
- Deterministic code gates:
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno or probe"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno or probe"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Saved outputs:
  - `pytest_torch_runner.log`
  - `pytest_compare_wrapper.log`
  - `compileall.log`
  - `preflight_payload.json`
- Long-run completion proof:
  - tmux capture:
    `verification/final_tmux_capture.log`
  - root invocation:
    `invocation.json`
  - row-local completion proofs listed above

## Residual Risks

- This remains a single-seed, `2 / 2` train/test CDI ablation. The result is
  useful for same-contract directionality, not for broad statistical claims.
- The published comparison is intentionally metric- and contract-focused.
  Historical parameter-count fields do not line up cleanly enough across the
  fresh conditioned rows and older authorities to support an efficiency claim
  in this item.
- FFNO evaluation emitted negative-SSIM clamp warnings during MS-SSIM
  computation, but the tracked launcher PID still exited `0` and the required
  row-local artifacts were written successfully.

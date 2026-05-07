# CDI Lines128 Supervised FFNO Depth-24 No-Refiner Summary

- Date: `2026-05-07`
- Backlog item: `2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/execution_plan.md`
- State: `decision_support_append_only`
- Fixed contract id: `cdi_lines128_seed3`
- Claim boundary: `cdi_supervised_ffno_depth_companion_only`
- Corrected four-block supervised baseline source root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/runs/supervised_ffno_no_refiner_20260506T232535Z`
- Fresh supervised depth-24 root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/runs/supervised_ffno_depth24_20260507T192840Z`
- Completed PINN depth-24 companion context:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z`

## Claim Boundary

- This item is append-only supervised pure-FFNO depth-companion evidence only.
- It does not replace the immutable six-row CDI authority
  (`docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`)
  or the current paper-local FFNO packaging authority
  (`docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_no_refiner_ffno_table_refresh_summary.md`).
- Final paper-facing promotion remains deferred to
  `2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh`.

## Fixed Contract

- dataset contract id: `cdi_lines128_seed3`
- `N = 128`, `gridsize = 1`, `seed = 3`
- `40` epochs, `batch_size = 16`, `learning_rate = 2e-4`
- `ReduceLROnPlateau` (factor `0.5`, patience `2`, min_lr `1e-4`, threshold `0.0`)
- `torch_loss_mode = mae`, `torch_output_mode = real_imag`
- `fno_modes = 12`, `fno_width = 32`, `fno_cnn_blocks = 0`
- `nimgs_train = 2`, `nimgs_test = 2`, `nphotons = 1e9`
- `set_phi = True`, probe `Run1084_recon3_postPC_shrunk_3.npz`,
  `probe_source = custom`, `probe_scale_mode = pad_extrapolate`,
  `probe_smoothing_sigma = 0.5`
- fixed sample ids: `0`, `1`
- intended architecture delta only:
  - baseline `supervised_ffno`: `fno_blocks = 4`
  - fresh `supervised_ffno_depth24`: `fno_blocks = 24`

## Row Roster

| row_id | label | training | evidence_source | fno_blocks | fno_cnn_blocks |
|---|---|---|---|---:|---:|
| `supervised_ffno` | `FFNO + supervised` | supervised | reused_by_lineage | 4 | 0 |
| `supervised_ffno_depth24` | `FFNO-24 + supervised` | supervised | fresh_run | 24 | 0 |

## Contract Audit

- Contract audit:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/verification/contract_audit_supervised_depth24_vs_depth4.json`
- Audit result: PASS.
- Comparison standard:
  - config and invocation fields: exact equality on the listed locked-contract fields
  - dataset payloads: exact equality for every NPZ key after dropping only
    `_metadata.creation_info.timestamp`
- No unexpected config drift was found.
- Fresh train/test datasets are numerically identical to the corrected
  four-block supervised baseline after timestamp normalization; only run-local
  provenance and allowed row-id/label differences vary.

## Side-By-Side Metrics

| Row | amp_mae | phase_mae | amp_mse | phase_mse | amp_psnr | phase_psnr | amp_ssim | phase_ssim | amp_ms_ssim | phase_ms_ssim | amp_frc50 | phase_frc50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `supervised_ffno` | 0.351512 | 0.066118 | 0.195675 | 0.006884 | 55.2155 | 69.7523 | 0.265006 | 0.901529 | 0.692219 | 0.813509 | 24.0737 | 24.3153 |
| `supervised_ffno_depth24` | 0.348855 | 0.061699 | 0.193818 | 0.006077 | 55.2569 | 70.2940 | 0.258666 | 0.910437 | 0.693530 | 0.825113 | 23.9197 | 24.8749 |

Δ (`depth24 - depth4`):

- amp MAE `-0.002657`, phase MAE `-0.004419`
- amp MSE `-0.001857`, phase MSE `-0.000807`
- amp PSNR `+0.0414`, phase PSNR `+0.5417`
- amp SSIM `-0.006340`, phase SSIM `+0.008907`
- amp MS-SSIM `+0.001311`, phase MS-SSIM `+0.011604`
- amp FRC50 `-0.1541`, phase FRC50 `+0.5596`

## Parameter And Runtime Context

| Row | Parameters | Final train loss | Validation loss | Train wall time (s) | Inference time (s) |
|---|---:|---:|---:|---:|---:|
| `supervised_ffno` | 124,966 | 151648.031250 | 152468.109375 | 874.939 | 1.229 |
| `supervised_ffno_depth24` | 136,355 | 153125.265625 | 158522.859375 | 4609.254 | 6.373 |

Δ (`depth24 - depth4`):

- parameter count `+11,389` (`1.09x`)
- final train loss `+1,477.234375`
- validation loss `+6,054.750000`
- train wall time `+3,734.315 s` (`5.27x`)
- inference time `+5.144 s` (`5.19x`)

## Interpretation

- On the locked corrected pure-FFNO supervised CDI contract, increasing depth
  from `4 -> 24` produces a mixed but directionally interesting result.
- The depth-24 row improves phase-side metrics (`phase_mae`, `phase_psnr`,
  `phase_ssim`, `phase_ms_ssim`, `phase_frc50`) and makes only small changes
  on amplitude-side scalars.
- Those gains are narrow and not free. `amp_ssim` and `amp_frc50` regress
  slightly, validation loss worsens, and runtime expands by more than `5x`.
- This item therefore supports the narrower claim that deeper supervised pure
  FFNO can shift the objective-control tradeoff toward slightly better phase
  behavior on the locked `lines128` CDI contract, but it does not justify
  automatic promotion over the corrected four-block supervised row.
- The completed PINN depth-24 companion remains stronger, broader append-only
  evidence for the depth axis overall. This supervised row exists to complete
  the paired depth-24 FFNO family before the later final-refresh decision.

## Derived Artifacts

- Comparison JSON:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/verification/comparison_supervised_depth24_vs_depth4.json`
- Comparison CSV:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/verification/comparison_supervised_depth24_vs_depth4.csv`
- Contract audit markdown:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/verification/contract_audit_supervised_depth24_vs_depth4.md`

## Verification

Verification logs are archived under
`.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/verification/`.

- Required deterministic gates:
  - presence gate:
    `input_presence_gate.log`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "supervised_ffno or ffno"`:
    `pytest_grid_lines_torch_runner_ffno.log`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "supervised_ffno or ffno"`:
    `pytest_grid_lines_compare_wrapper_ffno.log`
  - `python -m compileall -q ptycho_torch scripts/studies`:
    `compileall.log`
  - focused supported-row preflight:
    `preflight_supervised_depth24_manifest.json`
- Long-run completion proof:
  - tmux-owned shell exit marker `__EXIT_CODE__=0`
  - pane capture:
    `tmux_completion_capture.log`
  - wrapper invocation:
    `runs/supervised_ffno_depth24_20260507T192840Z/invocation.json`
  - row-local completion proofs:
    `runs/supervised_ffno_depth24/launcher_completion.json`
    and `runs/supervised_ffno_depth24/exit_code_proof.json`
- Post-run audit:
  - `verification/contract_audit_supervised_depth24_vs_depth4.{json,md}`
  - `verification/comparison_supervised_depth24_vs_depth4.{json,csv}`

## Residual Risks

- This is a single-seed, two-image-per-split CDI follow-up. It is useful for
  same-contract directionality, not for broad statistical claims.
- The wrapper run itself records the generic claim boundary
  `decision_support_append_only`; this summary intentionally narrows the
  interpretation to `cdi_supervised_ffno_depth_companion_only`.
- The current paper-local FFNO packaging still points at the corrected
  four-block no-refiner rows by design. Any manuscript-facing promotion of this
  supervised depth-24 row must be handled explicitly by the later
  final-refresh item.

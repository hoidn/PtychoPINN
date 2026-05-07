# CDI Lines128 FFNO Depth-24 Ablation Summary

- Date: `2026-05-06`
- Backlog item: `2026-05-06-cdi-lines128-ffno-depth24-ablation`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/execution_plan.md`
- State: `decision_support_append_only`
- Fixed contract id: `cdi_lines128_seed3`
- Claim boundary: `cdi_ffno_depth_ablation_only`
- Corrected four-block baseline source root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/runs/ffno_no_refiner_20260506T223454Z`
- Fresh depth-24 root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z`

## Claim Boundary

- This item is append-only CDI FFNO depth-ablation evidence only.
- It does not replace the immutable six-row CDI authority
  (`docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`)
  or the current paper-local FFNO packaging authority
  (`docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_no_refiner_ffno_table_refresh_summary.md`).
- Manuscript-facing promotion remains deferred to
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
  - baseline `pinn_ffno`: `fno_blocks = 4`
  - fresh `pinn_ffno_depth24`: `fno_blocks = 24`

## Row Roster

| row_id | label | training | evidence_source | fno_blocks | fno_cnn_blocks |
|---|---|---|---|---:|---:|
| `pinn_ffno` | `FFNO + PINN` | pinn | reused_by_lineage | 4 | 0 |
| `pinn_ffno_depth24` | `FFNO-24 + PINN` | pinn | fresh_run | 24 | 0 |

## Contract Audit

- Contract audit:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/contract_audit.json`
- Audit result: PASS.
- Comparison standard:
  - config and invocation fields: exact equality on the listed locked-contract fields
  - dataset payloads: exact equality for every NPZ key after dropping only
    `_metadata.creation_info.timestamp`
- No unexpected config or invocation drift was found.
- Fresh train/test datasets are numerically identical to the corrected
  four-block baseline after timestamp normalization; only run-local provenance
  timestamps differ.

## Side-By-Side Metrics

| Row | amp_mae | phase_mae | amp_mse | phase_mse | amp_psnr | phase_psnr | amp_ssim | phase_ssim | amp_ms_ssim | phase_ms_ssim | amp_frc50 | phase_frc50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pinn_ffno` | 0.082043 | 0.137965 | 0.010590 | 0.030478 | 67.8819 | 63.2910 | 0.890305 | 0.959644 | 0.979843 | 0.720259 | 61.4650 | 58.7286 |
| `pinn_ffno_depth24` | 0.056506 | 0.121740 | 0.004984 | 0.023518 | 71.1549 | 64.4167 | 0.944487 | 0.974460 | 0.992207 | 0.811215 | 68.6174 | 65.6225 |

Δ (`depth24 - depth4`):

- amp MAE `-0.025537`, phase MAE `-0.016225`
- amp MSE `-0.005606`, phase MSE `-0.006960`
- amp PSNR `+3.2730`, phase PSNR `+1.1258`
- amp SSIM `+0.054182`, phase SSIM `+0.014817`
- amp MS-SSIM `+0.012364`, phase MS-SSIM `+0.090956`
- amp FRC50 `+7.1524`, phase FRC50 `+6.8939`

## Parameter And Runtime Context

| Row | Parameters | Final train loss | Validation loss | Train wall time (s) | Inference time (s) |
|---|---:|---:|---:|---:|---:|
| `pinn_ffno` | 124,968 | 0.064261 | 0.066389 | 873.742 | 1.231 |
| `pinn_ffno_depth24` | 701,628 | 0.051149 | 0.052045 | 4,754.923 | 8.505 |

Δ (`depth24 - depth4`):

- parameter count `+576,660` (`5.61x`)
- final train loss `-0.013112`
- validation loss `-0.014345`
- train wall time `+3,881.181 s` (`5.44x`)
- inference time `+7.275 s` (`6.91x`)

## Interpretation

- On the locked corrected pure-FFNO CDI contract, increasing depth from
  `4 -> 24` improves every tracked scalar reconstruction metric in this
  single-seed comparison.
- The gain is not free. The depth-24 row is far more expensive than the
  corrected four-block baseline in parameters, training time, and inference
  time.
- This item therefore supports the narrow claim that deeper pure FFNO can help
  on the locked `lines128` CDI contract, but only as append-only ablation
  evidence. It does not itself decide whether the depth-24 row should replace
  the current paper-facing four-block FFNO rows.

## Derived Artifacts

- Comparison JSON:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z/comparison_depth24_vs_depth4.json`
- Comparison CSV:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z/comparison_depth24_vs_depth4.csv`

## Verification

Verification logs are archived under
`.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/`.

- Required deterministic gates:
  - presence gate:
    `input_presence_gate.log`
  - `pytest -q tests/torch/test_generator_registry.py -k "ffno"`:
    `pytest_generator_registry_ffno.log`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "ffno"`:
    `pytest_grid_lines_torch_runner_ffno.log`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno"`:
    `pytest_grid_lines_compare_wrapper_ffno.log`
  - `python -m compileall -q ptycho_torch scripts/studies`:
    `compileall.log`
- Long-run completion proof:
  - tmux-owned shell printed `__EXIT_CODE__=0`
  - wrapper invocation:
    `runs/ffno_depth24_20260507T052301Z/invocation.json`
  - row-local completion proofs:
    `runs/pinn_ffno_depth24/launcher_completion.json`
    and `runs/pinn_ffno_depth24/exit_code_proof.json`
- Post-run audit:
  - `verification/contract_audit.json`
  - `comparison_depth24_vs_depth4.{json,csv}`

## Residual Risks

- This is a single-seed, two-image-per-split CDI ablation. The result is useful
  for same-contract directionality, not for broad statistical claims.
- The current paper-local FFNO packaging still points at the corrected
  four-block no-refiner rows by design. Any manuscript-facing promotion of the
  depth-24 row must be handled explicitly by the later final-refresh item.

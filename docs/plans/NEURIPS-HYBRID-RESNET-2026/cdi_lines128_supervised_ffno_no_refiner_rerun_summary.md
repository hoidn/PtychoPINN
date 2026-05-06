# CDI Lines128 Supervised FFNO No-Refiner Rerun Summary

- Date: `2026-05-06`
- Backlog item: `2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun`
- State: `completed`
- Claim boundary: `lines128_ffno_objective_control_corrected_pair`
- Fresh corrected objective-control root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/runs/supervised_ffno_no_refiner_20260506T232535Z`
- Corrected comparator root reused by lineage:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/runs/ffno_no_refiner_20260506T223454Z`

## Objective

Produce the corrected `lines128` CDI `supervised_ffno` row under the locked
paper contract with `fno_cnn_blocks=0`, then refresh the narrow FFNO
objective-control evidence so both compared training procedures use the same
pure no-refiner FFNO architecture.

Preserved historical proxy root:

- supervised extension:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z`

## Completed In This Pass

- Reused the existing compare-wrapper/runner path without production code
  edits because it already threaded `--fno-cnn-blocks 0` truthfully into the
  `supervised_ffno` launch path.
- Ran the required deterministic preflight before the expensive launch:
  - `python` FFNO instantiation proof with `cnn_blocks=0`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "supervised_ffno or ffno"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "supervised_ffno or ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Launched exactly one tmux-owned compare-wrapper rerun for
  `supervised_ffno` under the frozen `lines128` contract with
  `--fno-cnn-blocks 0`, tracked the exact launched PID, and waited for shell
  exit marker `__EXIT_CODE__=0`.
- Wrote row-level audits:
  - contract diff:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/verification/contract_diff.json`
  - no-refiner inspection:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/verification/no_refiner_inspection.json`
  - corrected objective-control audit:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/verification/objective_control_audit.json`
- Repaired the missing row-local `launcher_completion.json` from the finished
  wrapper stderr completion markers so the required completion proof is
  present alongside the saved row artifacts.

## Contract, Purity, And Fairness Audit

- The contract diff shows no non-allowed drift versus the historical
  supervised extension. The only allowed change is `fno_cnn_blocks: 2 -> 0`.
- The fresh train/test dataset payloads are numerically identical to the
  historical supervised-extension payloads after normalizing away only the
  `_metadata.creation_info.timestamp` field inside the `.npz` envelopes.
- The corrected supervised row proves no-refiner purity in executed artifacts:
  - instantiated fresh config:
    `len(model.refiners) == 0`
  - executed saved state dict:
    `refiner_key_count == 0`
- The historical supervised proxy still carries `16` refiner-related keys, so
  the correction is architectural and not a manifest relabel.
- The corrected comparator row reused by lineage is the completed no-refiner
  `pinn_ffno` rerun from
  `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun`; the objective-control
  audit confirms both compared FFNO rows use `fno_cnn_blocks=0`.

## Corrected Objective-Control Outcome

Corrected pure-FFNO supervised row (`fno_cnn_blocks=0`):

- parameter count: `124,966`
- final train loss: `151648.03125`
- validation loss: `152468.109375`
- runtime: `874.939 s` train, `1.229 s` inference
- headline metrics:
  - amp MAE `0.351512`
  - phase MAE `0.066118`
  - amp SSIM `0.265006`
  - phase SSIM `0.901529`
  - amp FRC50 `24.0737`
  - phase FRC50 `24.3153`

Corrected pure-FFNO comparator (`pinn_ffno`, `fno_cnn_blocks=0`):

- parameter count: `124,968`
- final train loss: `0.064261`
- validation loss: `0.066389`
- runtime: `873.742 s` train, `1.231 s` inference
- headline metrics:
  - amp MAE `0.082043`
  - phase MAE `0.137965`
  - amp SSIM `0.890305`
  - phase SSIM `0.959644`
  - amp FRC50 `61.4650`
  - phase FRC50 `58.7286`

Historical supervised FFNO-local-refiner proxy context (`fno_cnn_blocks=2`):

- parameter count: `323,916`
- validation loss: `144882.046875`
- amp MAE `0.386413`
- phase MAE `0.046563`
- amp SSIM `0.248427`
- phase SSIM `0.937179`
- amp FRC50 `24.2498`
- phase FRC50 `36.6597`

Key deltas:

- corrected supervised minus corrected PINN:
  - parameter count `-2`
  - amp MAE `+0.269469`
  - phase MAE `-0.071847`
  - amp SSIM `-0.625299`
  - phase SSIM `-0.058115`
  - amp FRC50 `-37.3913`
  - phase FRC50 `-34.4133`
- corrected supervised minus historical supervised proxy:
  - parameter count `-198,950`
  - validation loss `+7586.0625`
  - amp MAE `-0.034901`
  - phase MAE `+0.019555`
  - amp SSIM `+0.016579`
  - phase SSIM `-0.035650`
  - amp FRC50 `-0.1761`
  - phase FRC50 `-12.3444`

Interpretation:

- The corrected pair closes the pure-FFNO objective-control gap under a shared
  no-refiner architecture contract.
- Relative to corrected `pinn_ffno`, the supervised row keeps lower phase MAE
  but is materially worse on amplitude quality and on phase SSIM/FRC.
- Relative to the historical supervised proxy, removing local refiners makes
  the executed model much smaller and slightly improves amplitude metrics, but
  phase-side quality worsens and validation loss increases.
- This item updates the active FFNO objective-control evidence, but it does
  not rewrite the immutable six-row CDI authority. Broader CDI table promotion
  remains deferred to the separate no-refiner table-refresh scope.

## Verification

- Preflight logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/verification/model_instantiation_no_refiner.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/verification/pytest_grid_lines_torch_runner_ffno.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/verification/pytest_grid_lines_compare_wrapper_ffno.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/verification/compileall.log`
- Wrapper completion proof:
  - tmux-owned shell exit marker `__EXIT_CODE__=0`
  - wrapper invocation:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/runs/supervised_ffno_no_refiner_20260506T232535Z/invocation.json`
- Required row outputs present:
  - `runs/supervised_ffno/{invocation.json,config.json,history.json,metrics.json,model.pt,exit_code_proof.json,launcher_completion.json,stdout.log,stderr.log}`
  - `recons/supervised_ffno/recon.npz`
  - `visuals/amp_phase_supervised_ffno.png`
  - `visuals/amp_phase_error_supervised_ffno.png`
- Objective-control comparison standard:
  - same locked `lines128` contract, exact payload equality for train/test
    arrays after timestamp normalization, and no numerical tolerance beyond
    `np.array_equal` on the normalized payload fields.

## Residual Risks

- The shared completion helper still does not emit `launcher_completion.json`
  for ordinary fresh compare-wrapper runs; this item repaired the required
  completion proof manually from the finished wrapper logs rather than changing
  shared study code late in the cycle.
- The corrected pair is active objective-control evidence only. The canonical
  CDI headline table and any broader manuscript wording that previously relied
  on the historical FFNO-local-refiner objective-control rows still require a
  later explicit table-refresh sweep to consume this lineage cleanly.

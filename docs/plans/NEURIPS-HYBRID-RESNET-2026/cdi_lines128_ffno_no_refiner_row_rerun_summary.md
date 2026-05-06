# CDI Lines128 FFNO No-Refiner Row Rerun Summary

- Date: `2026-05-06`
- Backlog item: `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun`
- State: `completed`
- Claim boundary: `lines128_ffno_vs_hybrid_prerequisite_pair`
- Fresh corrected row root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/runs/ffno_no_refiner_20260506T223454Z`

## Objective

Produce a corrected `lines128` CDI `pinn_ffno` row under the locked paper
contract with `fno_cnn_blocks=0`, while preserving the historical
`fno_cnn_blocks=2` artifact lineage as FFNO-local-refiner proxy evidence only.

Preserved historical proxy roots:

- prerequisite pair:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`
- authoritative six-row paper bundle:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`

## Completed In This Pass

- Ran the required deterministic preflight without production code edits:
  - `python` FFNO instantiation proof with `cnn_blocks=0`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "ffno"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Launched exactly one tmux-owned compare-wrapper rerun for `pinn_ffno` under
  the frozen `lines128` contract with `--fno-cnn-blocks 0`.
- Waited on the tracked tmux-launched shell process until the wrapper exited
  with `__EXIT_CODE__=0`.
- Published row-local audit artifacts:
  - contract diff:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/verification/contract_diff.json`
  - no-refiner inspection:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/verification/no_refiner_inspection.json`
- Repaired the missing row-local `launcher_completion.json` from the fresh
  wrapper stderr markers because the shared completion helper currently gates
  ordinary fresh compare-wrapper runs and otherwise leaves this required file
  absent.

## Contract And No-Refiner Audit

- The contract diff shows no non-allowed field drift versus the authoritative
  `lines128` contract. The only allowed contract change is
  `fno_cnn_blocks: 2 -> 0`.
- Fresh train/test datasets are numerically identical to the authoritative
  bundle after normalizing away `_metadata.creation_info.timestamp`; all array
  payloads, coordinates, probe tensors, and physics fields match exactly.
- The corrected row proves pure FFNO execution in two ways:
  - instantiated fresh config:
    `len(model.refiners) == 0`
  - executed saved state dict:
    `refiner_key_count == 0`
- The historical proxy state dict still carries `16` refiner-related keys, so
  the no-refiner correction is real rather than a manifest-only relabel.

## Corrected Row Outcome

Corrected pure-FFNO row (`fno_cnn_blocks=0`):

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

Historical FFNO-local-refiner proxy (`fno_cnn_blocks=2`) comparison:

- parameter count: `323,918 -> 124,968` (`-198,950`)
- validation loss: `0.052043 -> 0.066389` (`+0.014347`)
- amp MAE: `0.062772 -> 0.082043` (`+0.019271`)
- phase MAE: `0.082839 -> 0.137965` (`+0.055126`)
- amp SSIM: `0.934830 -> 0.890305` (`-0.044525`)
- phase SSIM: `0.981592 -> 0.959644` (`-0.021948`)
- amp FRC50: `68.9009 -> 61.4650` (`-7.4359`)
- phase FRC50: `67.8409 -> 58.7286` (`-9.1123`)

Interpretation:

- Removing the local residual refiners makes the executed model materially
  smaller and strictly purer as FFNO evidence.
- On the locked `lines128` contract, the corrected pure-FFNO row is weaker than
  the historical local-refiner proxy on the main amplitude and phase metrics.
- This item therefore closes the architecture-correctness question for CDI
  pure-FFNO claims, but it does not promote FFNO into the canonical six-row
  table. That promotion remains deferred to
  `2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh`.

## Verification

- Preflight logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/verification/model_instantiation_no_refiner.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/verification/pytest_grid_lines_torch_runner_ffno.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/verification/pytest_grid_lines_compare_wrapper_ffno.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/verification/compileall.log`
- Wrapper completion proof:
  - tmux-owned shell exit marker `__EXIT_CODE__=0`
  - wrapper invocation:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/runs/ffno_no_refiner_20260506T223454Z/invocation.json`
- Required row outputs present:
  - `runs/pinn_ffno/{invocation.json,config.json,history.json,metrics.json,model.pt,exit_code_proof.json,launcher_completion.json,stdout.log,stderr.log}`
  - `recons/pinn_ffno/recon.npz`
  - `visuals/amp_phase_pinn_ffno.png`
  - `visuals/amp_phase_error_pinn_ffno.png`

## Residual Risks

- The shared completion helper still does not emit `launcher_completion.json`
  for ordinary fresh compare-wrapper runs; this item repaired the required
  completion proof manually from the finished wrapper logs rather than changing
  shared study code late in the cycle.
- The corrected row is prerequisite evidence only. The canonical CDI table,
  manuscript tables, and any downstream FFNO-vs-Hybrid narrative still require
  the separate table-refresh item to consume this rerun by lineage.

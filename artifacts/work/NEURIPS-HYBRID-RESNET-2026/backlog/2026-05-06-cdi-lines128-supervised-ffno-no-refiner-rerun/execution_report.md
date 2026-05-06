# Execution Report

## Completed In This Pass

- Confirmed the existing `supervised_ffno` compare-wrapper path already
  supports truthful `fno_cnn_blocks=0` launches; no production code edit was
  required.
- Ran the deterministic preflight gate:
  - `python` FFNO no-refiner instantiation proof
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "supervised_ffno or ffno"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "supervised_ffno or ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Launched and completed one tmux-owned `supervised_ffno` no-refiner rerun
  under the locked `lines128` contract:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/runs/supervised_ffno_no_refiner_20260506T232535Z`
- Wrote the required row-level audits:
  - `verification/contract_diff.json`
  - `verification/no_refiner_inspection.json`
  - `verification/objective_control_audit.json`
- Wrote the durable summary and refreshed the required evidence and table
  surfaces to point at the corrected no-refiner FFNO objective-control pair.

## Completed Plan Tasks

- Task 1: Freeze authorities and audit the existing launch path
- Task 2: Apply only the minimal runner/wrapper/test fixes needed
  - outcome: no code/test change required
- Task 3: Launch the fresh supervised FFNO no-refiner row
- Task 4: Audit no-refiner purity and same-contract fairness
- Task 5: Refresh objective-control outputs without rewriting the base table
- Task 6: Write the durable summary and update discoverability

## Remaining Required Plan Tasks

- None

## Verification

- Preflight commands passed:
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "supervised_ffno or ffno"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "supervised_ffno or ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Expensive launch completion proof:
  - tracked tmux-owned shell for the supervised rerun exited with
    `__EXIT_CODE__=0`
  - required row-local artifacts and reconstruction/visual outputs are present
- Fairness and purity checks passed:
  - contract diff shows no non-allowed drift
  - no-refiner inspection proves `refiner_key_count == 0`
  - objective-control audit uses only the corrected no-refiner
    `pinn_ffno`/`supervised_ffno` pair
- Comparison standard:
  - train/test payload comparison used exact equality after normalizing only
    `_metadata.creation_info.timestamp`; no `atol`/`rtol` relaxation was used

## Residual Risks

- Fresh compare-wrapper runs still require manual reconstruction of
  `launcher_completion.json` from wrapper stderr completion markers.
- The refreshed FFNO objective-control tables now point at the corrected
  no-refiner pair, but the immutable six-row CDI authority remains unchanged;
  broader canonical table promotion still belongs to the separate no-refiner
  table-refresh scope.

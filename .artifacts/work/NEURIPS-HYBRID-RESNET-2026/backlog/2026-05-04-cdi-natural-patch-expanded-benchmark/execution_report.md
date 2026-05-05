# Execution Report

## Completed In This Pass

- Implemented the natural-patch benchmark harness in [scripts/studies/cdi_natural_patch_benchmark.py](/home/ollie/Documents/PtychoPINN/scripts/studies/cdi_natural_patch_benchmark.py) and the CLI entrypoint in [scripts/studies/run_cdi_natural_patch_benchmark.py](/home/ollie/Documents/PtychoPINN/scripts/studies/run_cdi_natural_patch_benchmark.py).
- Added focused coverage in [tests/studies/test_cdi_natural_patch_benchmark.py](/home/ollie/Documents/PtychoPINN/tests/studies/test_cdi_natural_patch_benchmark.py) for:
  - immutable prepared-input derivation
  - prepared-input reuse when existing grouped artifacts are valid
  - dry-run contract emission
  - explicit row-status recording
  - default benchmark-mode dispatch through the live executor
  - direct-script CLI invocation
- Fixed two live-path issues discovered during verification:
  - the CLI failed in direct script mode because the repo root was not added to `sys.path`
  - benchmark mode defaulted to the stub executor instead of `_execute_rows`
- Added validated prepared-input reuse so benchmark launches consume the dry-run artifacts instead of rewriting the item-local grouped NPZs on every invocation.
- Refreshed the item root with a successful canonical dry-run after an interrupted diagnostic benchmark attempt left the grouped train file partially rewritten.

## Completed Plan Tasks

- Task 1 scope completed for the new harness surface: red/green tests now cover the frozen dataset intake, row roster, row-status handling, and CLI contract.
- Task 2 implementation completed for prepared-input derivation and identity-audit emission under the item-local `prepared_inputs/` root.
- Task 3 implementation completed for the dry-run contract path:
  - canonical dry-run command succeeds
  - contract artifacts are emitted under `contract/`
  - benchmark mode reaches the live row executor
  - repeated dry-runs reuse valid prepared inputs instead of rewriting them

## Remaining Required Plan Tasks

- Run the full authoritative natural-patch benchmark under tmux with the locked six-row roster and capture the required bundle artifacts under `runs/<run_id>/`.
- Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_expanded_benchmark_summary.md`.
- Update the discoverability/index surfaces:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`
  - `docs/index.md`
- Run the final broad verification required by the plan after the benchmark/docs pass, including `pytest -q -m integration`.

## Verification

- `pytest -q tests/studies/test_cdi_natural_patch_benchmark.py -k cli_entrypoint_runs_direct_script_mode`
  - passed
- `pytest -q tests/studies/test_cdi_natural_patch_benchmark.py -k default_executor_in_benchmark_mode`
  - passed
- `pytest -q tests/studies/test_cdi_natural_patch_benchmark.py -k reuses_existing_prepared_inputs_when_valid`
  - passed
- `pytest -q tests/studies/test_cdi_natural_patch_dataset.py tests/studies/test_cdi_natural_patch_benchmark.py`
  - passed (`18 passed`)
- `python -m compileall -q scripts/studies ptycho_torch`
  - passed
- Canonical dry-run command from the execution plan
  - passed after the CLI bootstrap fix
  - passed again after the prepared-input reuse fix; the second rerun completed via the reuse path instead of regenerating grouped inputs
- Diagnostic benchmark smoke run:
  - `--mode benchmark --rows baseline --seed 3` reached live GPU training and entered `Epoch 1/40`
  - the run was then intentionally terminated after proving the repaired benchmark path reached real execution

## Residual Risks

- The authoritative six-row benchmark bundle has not been produced in this pass, so row-level correctness for `pinn`, `pinn_hybrid_resnet`, `pinn_fno_vanilla`, `pinn_ffno`, and `pinn_neuralop_uno` is still unverified on the locked dataset.
- The interrupted diagnostic smoke run did not produce a durable benchmark bundle and should not be treated as claim evidence.
- Final documentation/index synchronization and the required `pytest -q -m integration` gate remain outstanding until the benchmark results pass is completed.

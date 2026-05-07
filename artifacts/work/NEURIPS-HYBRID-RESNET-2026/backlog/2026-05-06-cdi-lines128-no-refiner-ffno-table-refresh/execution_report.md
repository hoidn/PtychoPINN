# Execution Report

## Completed In This Pass

- Patched the CDI paper-refresh generators so the active Synthetic CDI FFNO
  paper rows, figure inputs, model-config packaging, and efficiency packaging
  consume the corrected four-block no-refiner reruns.
- Added focused regression coverage for corrected CDI FFNO labels, corrected
  phase-zoom source routing, corrected model-config root selection, and
  corrected efficiency counts/labels.
- Regenerated the paper-local CDI tables and phase-zoom figures.
- Updated the durable discovery surfaces and wrote the paper-refresh summary
  authority plus the item-local stale-consumer audit.

## Completed Plan Tasks

- Task 1: Freeze the source authorities and enumerate stale consumers.
- Task 2: Repair only the minimal paper-refresh generators needed.
- Task 3: Regenerate the paper-local CDI tables and figure inputs.
- Task 4: Regenerate model-config and efficiency packaging.
- Task 5: Refresh discovery surfaces and write the durable summary.

## Remaining Required Plan Tasks

- None.

## Verification

- Refresh command:
  - `python scripts/studies/paper_results_refresh.py --write-cdi-extended-assets --write-cdi-phase-zoom-figure --write-cdi-phase-zoom-per-panel-figure --write-model-config-table --write-efficiency-table`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh/verification/paper_results_refresh.log`
- Generator tests:
  - `pytest -q tests/studies/test_paper_results_refresh.py -k 'cdi or objective or phase_zoom'`
  - `pytest -q tests/studies/test_paper_model_config_table.py`
  - `pytest -q tests/studies/test_paper_efficiency_table.py`
- Compile gate:
  - `python -m compileall -q scripts/studies`
- Item-local audit:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh/verification/stale_consumer_audit.md`
- Comparison standard:
  - active-row lineage checks were exact source-path/label/claim-boundary checks;
    no `atol`/`rtol` tolerance gate was required for this refresh item.

## Residual Risks

- The manuscript draft prose and embedded figure-metadata comments still carry
  historical `FFNO-local refiner` wording and old figure-source comments. That
  prose layer was not edited in this plan item.
- The Synthetic CDI model-config and efficiency tables intentionally use the
  deduped effective state-dict count (`124,966`) for corrected `pinn_ffno`
  while preserving the raw recorded manifest count (`124,968`) in the JSON
  payload for auditability.

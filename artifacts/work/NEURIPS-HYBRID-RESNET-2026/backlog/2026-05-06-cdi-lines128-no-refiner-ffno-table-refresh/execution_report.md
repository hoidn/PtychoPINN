# Execution Report

## Completed In This Pass

- Addressed the implementation review (`REVISE`) findings:
  - **High:** restricted `render_cdi_objective_comparison_table()` to the active
    FFNO pair only via the new `CDI_OBJECTIVE_CONTROL_ACTIVE_MODELS = ("FFNO",)`
    constant. The renderer now raises `ValueError` if the FFNO pair is missing
    from the input rows. Replaced the regression test
    `test_render_cdi_objective_comparison_table_keeps_only_paired_models` with
    `test_render_cdi_objective_comparison_table_only_emits_active_ffno_pair`
    (which asserts CNN, U-NO, and SRU-Net are absent and the FFNO pair is the
    sole emitted section) plus a missing-pair guard test.
  - **Medium:** rewrote the `paper_evidence_index.md` Manuscript Incorporation
    Map so the CDI objective-control table and the CDI extended-metrics assets
    point at the no-refiner refresh authority
    (`cdi_lines128_no_refiner_ffno_table_refresh_summary.md`, claim boundary
    `complete_lines128_cdi_benchmark_plus_uno_extension_with_corrected_ffno_objective_control_pair`).
    The non-FFNO row authority remains the immutable six-row base bundle plus
    U-NO extension; the historical supervised-FFNO extension root is recorded
    only as caveated proxy provenance.
- Regenerated `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_objective_comparison.tex`;
  the file now contains a single FFNO objective block (PINN vs supervised) with
  bolded best values per column.
- Refreshed the durable summary
  (`cdi_lines128_no_refiner_ffno_table_refresh_summary.md`) so it documents the
  FFNO-only objective-control filter.

## Completed Current-Scope Work

- All plan tasks (Task 1 through Task 5) remain complete; this pass tightened
  Task 2 (renderer + regression test) and Task 5 (incorporation map) so the
  delivered artifacts now match the approved scope.

## Completed Plan Tasks

- Task 1: Freeze the source authorities and enumerate stale consumers.
- Task 2: Repair only the minimal paper-refresh generators needed (FFNO-only
  objective-control filter added in this pass).
- Task 3: Regenerate the paper-local CDI tables and figure inputs (FFNO-only
  `cdi_lines128_objective_comparison.tex` regenerated in this pass).
- Task 4: Regenerate model-config and efficiency packaging.
- Task 5: Refresh discovery surfaces and write the durable summary
  (`paper_evidence_index.md` incorporation map updated in this pass).

## Remaining Required Plan Tasks

- None.

## Verification

- Refresh command (current pass):
  - `python scripts/studies/paper_results_refresh.py --write-cdi-extended-assets`
- Generator tests (re-run in current pass):
  - `pytest -q tests/studies/test_paper_results_refresh.py -k 'cdi or objective or phase_zoom'`
    (9 passed)
    -> `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh/verification/pytest_paper_results_refresh.log`
  - `pytest -q tests/studies/test_paper_model_config_table.py` (5 passed)
    -> `.../verification/pytest_paper_model_config.log`
  - `pytest -q tests/studies/test_paper_efficiency_table.py` (8 passed)
    -> `.../verification/pytest_paper_efficiency.log`
- Compile gate:
  - `python -m compileall -q scripts/studies`
    -> `.../verification/compileall.log`
- Verification gate:
  - JSON validation for `model_variant_index.json`, `ablation_index.json`,
    `cdi_lines128_metrics_extended.json`, `model_config_by_benchmark.json`,
    and `paper_efficiency_table.json` all passed.
  - CSV smoke check passed for the three regenerated CSV assets.
  - Active-row lineage audit `rg` check returned no stale FFNO proxy
    designations as active CDI paper rows.
  - Depth-24 guard `rg` check returned no `fno_blocks=24` / `depth24` content.
- Item-local audit:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh/verification/stale_consumer_audit.md`
- Comparison standard:
  - active-row lineage checks were exact source-path/label/claim-boundary checks;
    no `atol`/`rtol` tolerance gate was required for this refresh item.

## Follow-Up Work

- Manuscript draft prose and embedded figure-metadata comments in
  `hybrid_resnet_neurips_first_draft.tex` still carry historical
  `FFNO-local refiner` wording and old figure-source comments. A separate
  manuscript-text pass remains required before paper prose can claim the
  corrected FFNO rows directly. This is recorded as follow-up work and is not
  in scope for this packaging item.

## Residual Risks

- The Synthetic CDI model-config and efficiency tables intentionally use the
  deduped effective state-dict count (`124,966`) for corrected `pinn_ffno`
  while preserving the raw recorded manifest count (`124,968`) in the JSON
  payload for auditability.
- The objective-control renderer is now strict: any future refresh that fails
  to provide the corrected FFNO pair will fail loudly rather than silently
  emit other paired models. This is the intended behavior for this paper-asset
  refresh, but downstream callers wanting a different active-model set must
  pass `active_models=` explicitly to `render_cdi_objective_comparison_table()`.

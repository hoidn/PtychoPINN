# Execution Report

## Completed In This Pass

- froze the accepted `lines128` CDI contract and the preserved six-row
  `paper_complete` root as the prerequisite authority for any optional
  classical extension
- audited both currently exposed classical solver branches in
  `scripts/reconstruction/hio_cdi_benchmark.py` against the accepted
  `lines128` paper-bundle contract
- recorded the durable incompatibility closeout in the checked-in authority
  note, the machine-readable execution manifest, and the protocol audit
- updated the study and top-level docs indexes so the classical closeout is
  discoverable without replacing the six-row CDI authority

## Completed Plan Tasks

- Tranche 1: completed
  - wrote `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_classical_cdi_execution_authority.md`
  - wrote `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-classical-baseline-feasibility/execution/classical_baseline_execution_manifest.json`
  - wrote `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-classical-baseline-feasibility/execution/protocol_compatibility_audit.md`
  - recorded structured audit decisions for `pynx_cdi_hio_er` and
    `known_probe_object_hio_er`
- Tranche 2: completed as truthful closeout
  - no narrow compatibility fix was taken because the audit showed broad
    contract and schema drift, not one recoverable implementation gap
- Tranche 3: completed
  - ran the mandatory compile and required-context gates and archived logs
  - ran focused pytest selectors for the classical script and the `lines128`
    paper-bundle contract surfaces and archived the log
- Tranche 4: completed as `not_protocol_compatible`
  - no classical run launched because the protocol audit resolved the item
    before any expensive execution step
- Tranche 5: completed
  - wrote `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_classical_cdi_baseline_summary.md`
  - updated `docs/studies/index.md` and `docs/index.md`

## Remaining Required Plan Tasks

- none
- the current-scope item closes truthfully as `not_protocol_compatible`

## Verification

- required context gate:
  `python - <<'PY' ...`
  -> `classical CDI context present`
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-classical-baseline-feasibility/verification/required_context_check.log`
- compile gate:
  `python -m compileall -q scripts/reconstruction scripts/studies`
  -> exit `0`
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-classical-baseline-feasibility/verification/compileall_classical_lines128.log`
- focused contract-surface tests:
  `pytest -q tests/scripts/test_hio_cdi_benchmark.py tests/studies/test_metrics_tables.py tests/studies/test_lines128_paper_benchmark.py tests/studies/test_paper_provenance.py tests/test_grid_lines_compare_wrapper.py`
  -> `154 passed, 23 warnings in 27.21s`
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-classical-baseline-feasibility/verification/pytest_classical_contract_surfaces.log`
- authority/report/state consistency check:
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-classical-baseline-feasibility/verification/authority_consistency_check.log`

## Residual Risks

- this closeout only covers the currently exposed classical branches in
  `scripts/reconstruction/hio_cdi_benchmark.py`
- a future classical `lines128` row is still possible, but it needs a broader
  reviewed plan that starts from the frozen `N=128` bundle contract rather than
  extending the current Table-2 `N=64` study path
- no new classical metrics were generated in this pass, so the outcome is a
  contract-scope decision rather than a performance claim

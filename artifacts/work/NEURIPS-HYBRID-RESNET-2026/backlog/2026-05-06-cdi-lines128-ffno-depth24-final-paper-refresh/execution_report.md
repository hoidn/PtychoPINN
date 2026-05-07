# Execution Report

## Completed In This Pass

- Addressed the High implementation-review finding by populating the
  machine-readable checks ledger
  (`artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh-checks.json`)
  with every archived verification command, exit code, and log path required by
  the plan: the three preflight pytest selectors, the new postfix efficiency
  selector, the collect-only proof for the changed test module, both compileall
  smokes, the deterministic refresh invocation log, and all six canonical JSON
  validations. The checks JSON now lists 14 commands (up from 3).
- Addressed the Medium implementation-review finding in
  `scripts/studies/paper_efficiency_table.py` (`_collect_cdi_rows`). The active
  FFNO rows (`pinn_ffno`, `supervised_ffno`) now derive their `claim_boundary`
  directly from the resolved `final_ffno_pair.claim_boundary`, instead of
  silently inheriting the table-level boundary recorded in
  `tables/cdi_lines128_metrics_extended.json`. Standalone calls such as
  `write_paper_efficiency_table(..., cdi_final_ffno_pair_key="depth24_no_refiner")`
  now emit the matching depth-24 claim boundary even when the canonical CDI
  table on disk still reflects the four-block refresh. Non-FFNO CDI rows
  continue to inherit their lineage claim boundary unchanged.
- Added a focused regression test
  (`tests/studies/test_paper_efficiency_table.py::test_collect_cdi_rows_uses_final_pair_claim_boundary_for_active_ffno_rows`)
  that pins the corrected precedence: under
  `cdi_final_ffno_pair_key="depth24_no_refiner"`, the active FFNO rows must
  carry the depth-24 claim boundary while non-FFNO rows must not.
- Re-ran postfix verification after the code change and recorded the fresh
  evidence under the item's verification root: `pytest_efficiency_postfix.log`,
  `compileall_postfix.log`, `pytest_collect.log`, plus refreshed `json_*.log`
  proofs for the six canonical JSON outputs.

## Completed Current-Scope Work

- Task 1 — Audit prerequisites and freeze the promotion rule.
- Task 2 — Parameterize the paper-refresh generators around an explicit final
  FFNO pair, with the standalone-writer claim-boundary precedence now also
  matching the resolved pair.
- Task 3 — Regenerate final CDI FFNO paper-local assets from the chosen
  four-block no-refiner pair (canonical + versioned outputs).
- Task 4 — Refresh durable evidence and study discovery surfaces so the
  promotion decision is consistent across the manifest, indexes, evidence
  matrix, and study catalog.
- Task 5 — Durable final summary in
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_final_paper_refresh_summary.md`.

## Follow-Up Work

- Implementation-review medium follow-up: extend the model-config and
  efficiency writers to emit versioned copies before overwriting canonical
  filenames, matching the plan's preferred packaging for those writer
  surfaces. The one-shot refresh flow already produces matching versioned
  outputs via `paper_results_refresh.py`; tracking the writer-level versioning
  as a separate non-blocking item.
- Pyright currently reports pre-existing argument-type findings in
  `paper_efficiency_table.py` (lines 135, 144, 147) and an unused `repo_root`
  finding (line 533). These are not introduced by this pass and are not in
  scope for the current backlog item. Tracking as code-quality cleanup.

## Residual Risks

- The depth-24 family remains scientifically useful append-only evidence, but
  it is intentionally not the current paper-local CDI FFNO pair. Future
  readers must not treat the versioned depth-24 studies as manuscript-facing
  replacement authority without a new explicit refresh summary.
- This pass updates only repo-local paper assets, the verification ledger, and
  the standalone-writer claim-boundary precedence. It does not update
  `/home/ollie/Documents/neurips/` or manuscript prose, per the approved
  plan's non-goals.
- Non-FFNO CDI rows remain reused strictly by lineage from the immutable
  six-row authority and the append-only U-NO extension. Any future rerun or
  promotion outside the FFNO pair would require a separate approved plan.

## Verification

- `pytest -q tests/studies/test_paper_results_refresh.py -k 'cdi or objective or phase_zoom'`
  passed: `11 passed, 44 deselected`
  (`verification/pytest_preflight.log`).
- `pytest -q tests/studies/test_paper_model_config_table.py`
  passed: `7 passed`
  (`verification/pytest_model_config_preflight.log`).
- `pytest -q tests/studies/test_paper_efficiency_table.py`
  preflight: `9 passed`
  (`verification/pytest_efficiency_preflight.log`).
- `pytest -q tests/studies/test_paper_efficiency_table.py`
  postfix after the precedence fix and new regression test: `10 passed`
  (`verification/pytest_efficiency_postfix.log`).
- `pytest --collect-only tests/studies/test_paper_results_refresh.py tests/studies/test_paper_model_config_table.py tests/studies/test_paper_efficiency_table.py -q`
  collected `72 tests`
  (`verification/pytest_collect.log`).
- `python -m compileall -q scripts/studies` exited `0`
  (`verification/compileall_preflight.log`,
  `verification/compileall_postfix.log`).
- `python scripts/studies/paper_results_refresh.py --cdi-final-ffno-pair four_block_no_refiner --cdi-final-output-stem ffno_final_depth4pair --write-cdi-extended-assets --write-cdi-phase-zoom-figure --write-cdi-phase-zoom-per-panel-figure --write-model-config-table --write-efficiency-table`
  exited `0`
  (`verification/paper_results_refresh_ffno_final_depth4pair.log`).
- `python -m json.tool` validation passed for:
  - `paper_evidence_manifest.json`
  - `model_variant_index.json`
  - `ablation_index.json`
  - `tables/cdi_lines128_metrics_extended.json`
  - `tables/model_config_by_benchmark.json`
  - `tables/paper_efficiency_table.json`
  Archived under `verification/json_*.log`.
- The refreshed canonical CDI metrics JSON records
  `final_ffno_pair.pair_key=four_block_no_refiner`,
  `final_output_stem=ffno_final_depth4pair`, and claim boundary
  `complete_lines128_cdi_benchmark_plus_uno_extension_with_final_four_block_no_refiner_ffno_pair`.
- All 14 archived commands above are now individually recorded in
  `artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh-checks.json`
  with their exit codes and log paths.

# Execution Report

## Completed In This Pass

- Audited the corrected four-block no-refiner FFNO pair and both completed
  depth-24 companions against the approved summaries and artifact roots, then
  froze the final promotion rule before touching the generators: the active
  manuscript-facing CDI FFNO pair must remain same-depth, and because the
  supervised depth-24 companion is mixed while cost expands sharply, the
  corrected four-block no-refiner pair remains the canonical repo-local final
  paper pair.
- Added explicit final-pair source selection shared by the paper-refresh
  tooling via `scripts/studies/cdi_final_ffno_pair.py`, then threaded that
  selection through:
  - `scripts/studies/paper_results_refresh.py`
  - `scripts/studies/paper_model_config_table.py`
  - `scripts/studies/paper_efficiency_table.py`
  The refresh stack now accepts
  `--cdi-final-ffno-pair {four_block_no_refiner,depth24_no_refiner}` plus
  `--cdi-final-output-stem ...`, emits versioned outputs, and records
  provenance for the chosen final FFNO pair instead of hardcoding the prior
  interim roots.
- Extended the focused paper-refresh regression coverage so the final-pair
  contract is enforced in tests:
  - `tests/studies/test_paper_results_refresh.py`
  - `tests/studies/test_paper_model_config_table.py`
  - `tests/studies/test_paper_efficiency_table.py`
  The new cases cover final-pair resolution, deterministic versioned output
  paths, CDI metrics provenance, model-config pair switching, and efficiency
  row switching.
- Regenerated the CDI FFNO paper-local derived assets from the retained
  four-block pair using
  `--cdi-final-ffno-pair four_block_no_refiner` and
  `--cdi-final-output-stem ffno_final_depth4pair`, producing both canonical
  manuscript-consumption outputs and provenance-safe versioned copies for the
  metrics, objective-control, phase-zoom, model-config, and efficiency asset
  families.
- Refreshed the durable evidence and discoverability surfaces so the final
  paper-local authority is explicit and consistent:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_final_paper_refresh_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/studies/index.md`
  Historical and append-only summaries were also updated so they no longer
  imply that depth-24 promotion is still unresolved.

## Completed Plan Tasks

- Task 1: Audit prerequisites and freeze the promotion rule.
- Task 2: Parameterize the paper-refresh generators around an explicit final
  FFNO pair.
- Task 3: Regenerate final CDI FFNO paper-local assets.
- Task 4: Refresh durable evidence and study discovery surfaces.
- Task 5: Write the durable final summary and close the refresh.

## Remaining Required Plan Tasks

- None. The approved plan scope is complete in the current checkout.

## Verification

- `pytest -q tests/studies/test_paper_results_refresh.py -k 'cdi or objective or phase_zoom'`
  passed: `11 passed, 44 deselected`
  (`verification/pytest_preflight.log`).
- `pytest -q tests/studies/test_paper_model_config_table.py`
  passed: `7 passed`
  (`verification/pytest_model_config_preflight.log`).
- `pytest -q tests/studies/test_paper_efficiency_table.py`
  passed: `9 passed`
  (`verification/pytest_efficiency_preflight.log`).
- `python -m compileall -q scripts/studies tests/studies`
  exited `0` (`verification/compileall_preflight.log`).
- `python scripts/studies/paper_results_refresh.py --cdi-final-ffno-pair four_block_no_refiner --cdi-final-output-stem ffno_final_depth4pair --write-cdi-extended-assets --write-cdi-phase-zoom-figure --write-cdi-phase-zoom-per-panel-figure --write-model-config-table --write-efficiency-table`
  exited `0` and refreshed the canonical plus versioned CDI FFNO paper-local
  assets (`verification/paper_results_refresh_ffno_final_depth4pair.log`).
- `python -m json.tool` validation passed for:
  - `paper_evidence_manifest.json`
  - `model_variant_index.json`
  - `ablation_index.json`
  - `tables/cdi_lines128_metrics_extended.json`
  - `tables/model_config_by_benchmark.json`
  - `tables/paper_efficiency_table.json`
  Archived under `verification/json_*.log`.
- The refreshed canonical CDI metrics JSON now records
  `final_ffno_pair.pair_key=four_block_no_refiner`,
  `final_output_stem=ffno_final_depth4pair`, and claim boundary
  `complete_lines128_cdi_benchmark_plus_uno_extension_with_final_four_block_no_refiner_ffno_pair`.
- No new numerical parity or regression threshold was introduced in this plan.
  The promotion decision reused the completed same-contract summary deltas
  rather than an `atol`/`rtol` comparison gate.

## Residual Risks

- The depth-24 family remains scientifically useful append-only evidence, but
  it is intentionally not the current paper-local CDI FFNO pair. Future
  readers must not treat the versioned depth-24 studies as manuscript-facing
  replacement authority without a new explicit refresh summary.
- This pass updates only repo-local paper assets and discovery surfaces. It
  does not update `/home/ollie/Documents/neurips/` or manuscript prose, per
  the approved plan's non-goals.
- Non-FFNO CDI rows remain reused strictly by lineage from the immutable
  six-row authority and the append-only U-NO extension. Any future rerun or
  promotion outside the FFNO pair would require a separate approved plan.

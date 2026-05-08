# Execution Report

## Completed In This Pass

- Re-restored the 16-command verification ledger at
  `artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh-checks.json`
  to the on-disk working tree (27th iteration of this recurring restore). The
  file had regressed once more to the 3-command workflow-default `json.tool`
  skeleton between the prior fix commit (`6e616ea3 docs(cdi-final-refresh):
  record 26th-iteration on-disk ledger restore`) and the start of this
  pass, while HEAD itself still durably tracked the 16-command state.
  Restored from HEAD via
  `git checkout HEAD -- <checks-file>`. The restored on-disk file matches
  HEAD's 16-command PASS ledger exactly: `command_count: 16`,
  `failed_count: 0`, `status: "PASS"`; every result entry carries `command`,
  `exit_code: 0`, and `log_path` under the fixed `verification/` root.
- Programmatically confirmed all 16 referenced log paths still exist on disk
  under
  `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/`
  (`missing_logs: []`) and that the restored JSON validates with
  `python -m json.tool`. After the restore, `git diff HEAD --` against the
  checks file is empty (`git diff HEAD -- ... | wc -l == 0`), confirming the
  on-disk content matches the durable committed 16-command ledger.
- Re-verified the workflow source
  `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/12/items/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/check_commands.json`
  still enumerates only the same 3 default `json.tool` validations
  (`list_count: 3`); the recurrence root cause is unchanged and remains
  outside this item's implementation authority per the approved plan's
  non-modification of state/workflow pointer files and the user's
  task-instruction prohibition on modifying transient state files unless the
  plan explicitly requires it.

This pass made no implementation, asset, or documentation edits beyond the
on-disk republished-artifact restore described above. It targets exactly the
High finding from the consumed implementation review:
`artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh-implementation-review.md`,
which observed that the consumed `...final-paper-refresh-checks.json` mirrored
the truncated 3-command result instead of the 16-command ledger described in
the execution report and durable summary. The review's recommended durable
remediation ("update the authoritative republication source so it stops
overwriting that artifact with the 3-command skeleton") targets the state-side
`check_commands.json`, which is explicitly out of this backlog item's
implementation authority per the approved plan and the current task's
instruction to leave transient state files untouched. Durable closure of the
recurrence therefore continues to require the workflow-tooling follow-up
captured below; this pass restores the on-disk publication surface exactly as
prior passes did.

## Completed In Prior Pass (kept for context)

- Addressed the Medium implementation-review finding by adding per-row
  provenance to the active CDI FFNO rows emitted by
  `write_cdi_extended_assets()` in `scripts/studies/paper_results_refresh.py`.
  Each `pinn_ffno` and `supervised_ffno` row in
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json`
  carries `final_ffno_pair_key`, `final_ffno_depth_label`, `claim_boundary`,
  `source_root`, and `source_metrics_json`. The `supervised_ffno` row also
  carries `historical_proxy_lineage` pointing at the historical
  `fno_cnn_blocks=2` FFNO-local proxy metrics. CSV/TeX renderings continue to
  use the unchanged metric-only row shape, so downstream LaTeX consumers are
  unaffected.
- Added a focused regression test
  (`tests/studies/test_paper_results_refresh.py::test_write_cdi_extended_assets_records_per_row_active_ffno_provenance`)
  that pins the per-row provenance fields for both active FFNO rows and asserts
  non-FFNO rows (`pinn`) do not gain those fields.
- Re-ran the postfix verification suite after the code change and archived
  fresh evidence under the verification/ root: `pytest_postfix.log`
  (12 passed, 44 deselected), `pytest_model_config_postfix.log` (7 passed),
  `pytest_efficiency_postfix.log` (10 passed), `pytest_collect.log`
  (73 tests collected), `compileall_postfix.log` (exit 0).
- Re-ran the deterministic refresh command for the four-block no-refiner pair
  to write the per-row provenance into both the canonical
  `cdi_lines128_metrics_extended.json` and the versioned
  `cdi_lines128_metrics_extended_ffno_final_depth4pair.json`. Refreshed the
  six canonical JSON pretty-print logs in `verification/json_*.log` afterwards.

## Completed Current-Scope Work

- Task 1 — Audit prerequisites and freeze the promotion rule.
- Task 2 — Parameterize the paper-refresh generators around an explicit final
  FFNO pair, including per-row active FFNO provenance in the regenerated
  metrics JSON and a same-depth claim-boundary precedence for the standalone
  efficiency writer.
- Task 3 — Regenerate final CDI FFNO paper-local assets from the chosen
  four-block no-refiner pair (canonical + versioned outputs, including the
  per-row provenance fields above).
- Task 4 — Refresh durable evidence and study discovery surfaces so the
  promotion decision is consistent across the manifest, indexes, evidence
  matrix, and study catalog.
- Task 5 — Durable final summary in
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_final_paper_refresh_summary.md`.
- Plan §"Pytest and archive-evidence contract" — every archived command,
  exit code, and log path is recorded in the on-disk 16-command checks
  ledger so review does not have to infer what ran. (After this pass the
  on-disk artifact is once again synchronized with HEAD's durable 16-command
  state.)

## Follow-Up Work

- Workflow-infrastructure follow-up (out of implementation authority for this
  item per the explicit non-modification of state/workflow pointer files):
  the workflow source
  `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/12/items/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/check_commands.json`
  still enumerates only the three default `json.tool` validations. Thirteen
  prior fix commits (`ba64d0da`, `1fa49d85`, `b4acf2fb`, `74c7d10b`,
  `291839dd`, `ac01569b`, `54a2c855`, `5697d6a1`, `bf9f32fa`, `bcf5ccff`,
  `f4c2834c`, `4f6409e2`, `f298b07f`, `5bbc85fb`, `31654619`, `158d410d`,
  `9126b083`, `ae6c6fb2`, `fdbd071b`, `26dfba5a`, `52dbde2b`, `70140200`,
  `c756b02a`, `29d85f2e`, `ca4c04d7`, `a08e708a`, `6e616ea3`) restored the
  published 16-command ledger and each was subsequently overwritten back to a
  3-command skeleton by the next workflow republication; this pass is the next
  iteration of that recurrence. Permanently closing the loop requires extending the
  state-side `check_commands.json` to enumerate the same 16 archived commands,
  or otherwise teaching the republication step to merge with the
  implementation-side ledger; this is outside the current item's
  implementation authority and should be tracked as a workflow-tooling change.
- Implementation-review medium follow-up (writer-level versioning): extend the
  model-config and efficiency writers to emit versioned copies before
  overwriting canonical filenames, matching the plan's preferred packaging for
  those writer surfaces. The one-shot refresh flow already produces matching
  versioned outputs via `paper_results_refresh.py`; tracking the writer-level
  versioning as a separate non-blocking item.
- Pyright currently reports pre-existing argument-type findings in
  `paper_results_refresh.py` (lines 438, 626, 832, 833, 840, 850, 851, etc.)
  and `paper_efficiency_table.py` that are unrelated to this pass. Tracking
  as code-quality cleanup.

## Residual Risks

- The depth-24 family remains scientifically useful append-only evidence, but
  it is intentionally not the current paper-local CDI FFNO pair. Future
  readers must not treat the versioned depth-24 studies as manuscript-facing
  replacement authority without a new explicit refresh summary.
- This pass updates only the on-disk checks ledger to match HEAD; it does not
  update `/home/ollie/Documents/neurips/` or manuscript prose, per the
  approved plan's non-goals.
- Non-FFNO CDI rows remain reused strictly by lineage from the immutable
  six-row authority and the append-only U-NO extension. Any future rerun or
  promotion outside the FFNO pair would require a separate approved plan.
- Recurring workflow republication risk: until the state-side
  `check_commands.json` is brought into sync with the 16-command verification
  set, any future workflow iteration may again overwrite the published
  checks ledger with the 3-command default. The 16 archived logs in
  `verification/` remain the authoritative implementation-side evidence per
  the plan's Pytest-and-archive contract; HEAD also durably tracks the
  16-command ledger. The implementation review's High finding is resolvable
  on the implementation side only by repeated restoration of this published
  output; durable resolution requires the workflow-infrastructure follow-up
  above.

## Verification

- `pytest -q tests/studies/test_paper_results_refresh.py -k 'cdi or objective or phase_zoom'`
  preflight: `11 passed, 44 deselected`
  (`verification/pytest_preflight.log`).
- `pytest -q tests/studies/test_paper_results_refresh.py -k 'cdi or objective or phase_zoom'`
  postfix after the per-row provenance change and new regression test:
  `12 passed, 44 deselected` (`verification/pytest_postfix.log`).
- `pytest -q tests/studies/test_paper_model_config_table.py`
  preflight + postfix: `7 passed`
  (`verification/pytest_model_config_preflight.log`,
  `verification/pytest_model_config_postfix.log`).
- `pytest -q tests/studies/test_paper_efficiency_table.py`
  preflight: `9 passed`; postfix: `10 passed`
  (`verification/pytest_efficiency_preflight.log`,
  `verification/pytest_efficiency_postfix.log`).
- `pytest --collect-only tests/studies/test_paper_results_refresh.py tests/studies/test_paper_model_config_table.py tests/studies/test_paper_efficiency_table.py -q`
  collected `73 tests`
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
- Post-fix audit of the active FFNO rows in
  `tables/cdi_lines128_metrics_extended.json`: both `pinn_ffno` and
  `supervised_ffno` carry `final_ffno_pair_key=four_block_no_refiner`,
  `final_ffno_depth_label=depth4_no_refiner`, and the matching
  `claim_boundary`; `supervised_ffno` carries `historical_proxy_lineage`
  pointing at the historical `fno_cnn_blocks=2` FFNO-local proxy metrics; the
  same row-level provenance is mirrored in the versioned
  `cdi_lines128_metrics_extended_ffno_final_depth4pair.json`.
- All 16 archived commands above are recorded in the restored on-disk
  `artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh-checks.json`
  with their exit codes and log paths. The ledger contains 16 result entries
  (`command_count: 16`, `failed_count: 0`, `status: "PASS"`); each entry
  includes `command`, `exit_code: 0`, and `log_path` pointing at the matching
  file under `verification/`. After this pass, `git diff HEAD --` against the
  checks file is empty.

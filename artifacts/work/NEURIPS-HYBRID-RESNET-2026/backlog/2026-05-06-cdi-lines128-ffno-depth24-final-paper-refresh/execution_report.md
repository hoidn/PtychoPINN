# Execution Report

## Completed In This Pass

- Restored the on-disk 16-command checks ledger at
  `artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh-checks.json`
  from HEAD (35th iteration of this recurring restore). The file had again
  regressed to the 3-command workflow-default skeleton — exactly the
  republication regression flagged by the consumed implementation review's
  High finding. The restored file matches HEAD exactly
  (`command_count: 16`, `failed_count: 0`, `status: "PASS"`,
  `missing_logs: []`, empty `git diff HEAD`). All 16 archived verification
  logs under `verification/` already exist on disk and continue to satisfy
  the plan's "Pytest and archive-evidence contract".
- No implementation code changed in this pass, so the targeted pytest
  preflight/postfix logs, the `paper_results_refresh_ffno_final_depth4pair.log`
  refresh log, and the six `json_*.log` validation logs archived previously
  under `verification/` remain authoritative for this final state. The six
  required deterministic `python -m json.tool` validations against the
  published surfaces (`paper_evidence_manifest.json`,
  `model_variant_index.json`, `ablation_index.json`,
  `tables/cdi_lines128_metrics_extended.json`,
  `tables/model_config_by_benchmark.json`, `tables/paper_efficiency_table.json`)
  remain pinned in the restored ledger as `exit_code: 0`.

## Earlier Pass: Per-Row Provenance Closure

- Closed the implementation review's Medium finding by extending per-row active
  FFNO provenance from `cdi_lines128_metrics_extended.json` to the other two
  canonical machine-consumed CDI surfaces:
  `model_config_by_benchmark.json` and `paper_efficiency_table.json`. Each
  active `pinn_ffno` and `supervised_ffno` row in those JSON outputs now
  carries `final_ffno_pair_key`, `final_ffno_depth_label`, `claim_boundary`,
  `source_root`, and `source_metrics_json`, with `supervised_ffno` also
  carrying `historical_proxy_lineage`. Non-FFNO CDI rows (e.g.,
  `pinn_hybrid_resnet`) remain unchanged. CSV/TeX renderings continue to use
  the unchanged metric/config-only row shapes, so downstream LaTeX consumers
  are unaffected.
- Centralized the per-row provenance dictionary in
  `scripts/studies/cdi_final_ffno_pair.py` via a new
  `CdiFinalFfnoPair.active_row_provenance()` method. This is the single
  source of truth used by `write_cdi_extended_assets()`, the model-config
  writer (`scripts/studies/paper_model_config_table.py::write_model_config_json`),
  and the efficiency writer
  (`scripts/studies/paper_efficiency_table.py::write_paper_efficiency_table`).
- Added focused regression tests pinning the per-row provenance for both
  newly-augmented JSON surfaces:
  - `tests/studies/test_paper_model_config_table.py::test_write_model_config_table_records_per_row_active_ffno_provenance`
  - `tests/studies/test_paper_efficiency_table.py::test_write_paper_efficiency_table_records_per_row_active_ffno_provenance`
  Both tests exercise the `depth24_no_refiner` swap so the per-row pair key,
  claim boundary, source root, and historical proxy lineage are verified
  end-to-end. Existing tests still pass.
- Re-ran the deterministic refresh command for the four-block no-refiner pair
  (`python scripts/studies/paper_results_refresh.py
  --cdi-final-ffno-pair four_block_no_refiner
  --cdi-final-output-stem ffno_final_depth4pair ...`) so the canonical and
  versioned `model_config_by_benchmark*.{json,csv,tex}` and
  `paper_efficiency_table*.{json,csv,tex}` outputs carry the new per-row
  provenance fields. Audited `pinn_ffno` / `supervised_ffno` rows in both
  canonical JSON surfaces to confirm `final_ffno_pair_key=four_block_no_refiner`,
  the matching `claim_boundary`, and `historical_proxy_lineage` for the
  supervised row. Non-FFNO rows did not gain the new fields.
- Refreshed the `verification/` postfix logs after the per-row provenance code
  change: `pytest_postfix.log` (12 passed, 44 deselected),
  `pytest_model_config_postfix.log` (8 passed),
  `pytest_efficiency_postfix.log` (11 passed), `pytest_collect.log` (75 tests
  collected), `compileall_postfix.log` (exit 0),
  `paper_results_refresh_ffno_final_depth4pair.log` (exit 0), and the six
  `json_*.log` JSON validation logs.
- Re-restored the 16-command verification ledger at
  `artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh-checks.json`
  to the on-disk working tree (prior iterations through 31st). HEAD
  durably tracks the 16-command state; the on-disk file regressed again to
  the 3-command workflow-default `json.tool` skeleton between the prior fix
  commit (`74c7d10b`) and the start of this pass, which is the same
  republication regression flagged by the consumed implementation review.
  Restored from HEAD via `git checkout HEAD -- <checks-file>`. The restored
  on-disk file matches HEAD's 16-command PASS ledger exactly
  (`command_count: 16`, `failed_count: 0`, `status: "PASS"`); every
  `log_path` entry resolves to an existing file under `verification/`
  (`missing_logs: []`); and `git diff HEAD --` against the checks file is
  empty after the restore.

This pass made narrow, surgical edits to
`scripts/studies/cdi_final_ffno_pair.py`,
`scripts/studies/paper_model_config_table.py`,
`scripts/studies/paper_efficiency_table.py`, and
`scripts/studies/paper_results_refresh.py`, plus the two test modules. It
targets exactly the High and Medium findings from the consumed implementation
review:
`artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh-implementation-review.md`.
The High recurrence (workflow republication of the checks ledger) is again
patched on the implementation side; durable closure still requires a
workflow-tooling change outside this item's implementation authority (the
state-side `check_commands.json` enumerates only the 3-command default) and
is captured below as follow-up work.

## Completed Current-Scope Work

- Task 1 — Audit prerequisites and freeze the promotion rule.
- Task 2 — Parameterize the paper-refresh generators around an explicit final
  FFNO pair, including per-row active FFNO provenance in **all three**
  canonical JSON payloads (`cdi_lines128_metrics_extended.json`,
  `model_config_by_benchmark.json`, `paper_efficiency_table.json`) and a
  same-depth claim-boundary precedence for the standalone efficiency writer.
- Task 3 — Regenerate final CDI FFNO paper-local assets from the chosen
  four-block no-refiner pair (canonical + versioned outputs, including the
  per-row provenance fields above on every JSON surface).
- Task 4 — Refresh durable evidence and study discovery surfaces so the
  promotion decision is consistent across the manifest, indexes, evidence
  matrix, and study catalog.
- Task 5 — Durable final summary in
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_final_paper_refresh_summary.md`.
- Plan §"Pytest and archive-evidence contract" — every archived command,
  exit code, and log path is recorded in the on-disk 16-command checks
  ledger. After this pass the on-disk artifact is once again synchronized
  with HEAD's durable 16-command state.

## Follow-Up Work

- Workflow-infrastructure follow-up (out of implementation authority for this
  item per the explicit non-modification of state/workflow pointer files):
  the workflow source
  `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/12/items/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/check_commands.json`
  still enumerates only the three default `json.tool` validations. Many prior
  fix commits (most recently `a08e708a`, `6e616ea3`, `46a848bc`, `54d189e1`,
  `e31ef21f`) restored the published 16-command ledger and each was
  subsequently overwritten back to a 3-command skeleton by the next workflow
  republication; this pass is the next iteration of that recurrence.
  Permanently closing the loop requires extending the state-side
  `check_commands.json` to enumerate the same 16 archived commands, or
  otherwise teaching the republication step to merge with the
  implementation-side ledger; this is outside the current item's
  implementation authority and should be tracked as a workflow-tooling
  change.
- Implementation-review medium follow-up (writer-level versioning): extend
  the model-config and efficiency writers' default packaging to emit
  versioned copies before overwriting canonical filenames on every invocation
  (the one-shot refresh flow already produces matching versioned outputs via
  `paper_results_refresh.py`). Tracking the writer-level versioning as a
  separate non-blocking item.
- Pyright currently reports pre-existing argument-type findings in
  `paper_results_refresh.py` (lines 438, 626, 832, 833, 840, 850, 851, etc.)
  and `paper_efficiency_table.py` that are unrelated to this pass. Tracking
  as code-quality cleanup.

## Residual Risks

- The depth-24 family remains scientifically useful append-only evidence, but
  it is intentionally not the current paper-local CDI FFNO pair. Future
  readers must not treat the versioned depth-24 studies as
  manuscript-facing replacement authority without a new explicit refresh
  summary.
- This pass updates only repo-local paper-asset JSON surfaces and the
  on-disk checks ledger; it does not update `/home/ollie/Documents/neurips/`
  or manuscript prose, per the approved plan's non-goals.
- Non-FFNO CDI rows remain reused strictly by lineage from the immutable
  six-row authority and the append-only U-NO extension. Any future rerun or
  promotion outside the FFNO pair would require a separate approved plan.
- Recurring workflow republication risk: until the state-side
  `check_commands.json` is brought into sync with the 16-command verification
  set, any future workflow iteration may again overwrite the published
  checks ledger with the 3-command default. The 16 archived logs in
  `verification/` remain the authoritative implementation-side evidence per
  the plan's Pytest-and-archive contract; HEAD also durably tracks the
  16-command ledger.

## Verification

- `pytest -q tests/studies/test_paper_results_refresh.py -k 'cdi or objective or phase_zoom'`
  preflight: `11 passed, 44 deselected`
  (`verification/pytest_preflight.log`).
- `pytest -q tests/studies/test_paper_results_refresh.py -k 'cdi or objective or phase_zoom'`
  postfix: `12 passed, 44 deselected` (`verification/pytest_postfix.log`).
- `pytest -q tests/studies/test_paper_model_config_table.py`
  preflight: `7 passed`
  (`verification/pytest_model_config_preflight.log`); postfix after the new
  per-row provenance regression test: `8 passed`
  (`verification/pytest_model_config_postfix.log`).
- `pytest -q tests/studies/test_paper_efficiency_table.py`
  preflight: `9 passed` (`verification/pytest_efficiency_preflight.log`);
  postfix after the new per-row provenance regression test: `11 passed`
  (`verification/pytest_efficiency_postfix.log`).
- `pytest --collect-only tests/studies/test_paper_results_refresh.py tests/studies/test_paper_model_config_table.py tests/studies/test_paper_efficiency_table.py -q`
  collected `75 tests` (`verification/pytest_collect.log`).
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
- Post-fix audit of the active FFNO rows in all three canonical JSON
  surfaces:
  - `tables/cdi_lines128_metrics_extended.json` — both `pinn_ffno` and
    `supervised_ffno` carry `final_ffno_pair_key=four_block_no_refiner`,
    `final_ffno_depth_label=depth4_no_refiner`, the matching
    `claim_boundary`, `source_root`, and `source_metrics_json`;
    `supervised_ffno` carries `historical_proxy_lineage`.
  - `tables/model_config_by_benchmark.json` — same per-row provenance
    fields now present for both active FFNO rows; non-FFNO rows
    (e.g., `pinn_hybrid_resnet`) do not gain those fields.
  - `tables/paper_efficiency_table.json` — same per-row provenance
    fields now present for both active FFNO rows; non-FFNO rows
    (e.g., `pinn_hybrid_resnet`) do not gain those fields.
  Provenance is mirrored in the matching versioned
  `*_ffno_final_depth4pair.json` outputs.
- All 16 archived commands above are recorded in the restored on-disk
  `artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh-checks.json`
  with their exit codes and log paths. The ledger contains 16 result entries
  (`command_count: 16`, `failed_count: 0`, `status: "PASS"`); each entry
  includes `command`, `exit_code: 0`, and `log_path` pointing at the matching
  file under `verification/`. After this pass, `git diff HEAD --` against the
  checks file is empty.

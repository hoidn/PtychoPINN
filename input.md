Summary: Extend the D0 parity logger Markdown so every photon-dose dataset surfaces raw/normalized/grouped stats inline and record how to run the new CLI/test selector documented for maintainers.
Focus: seed — Inbox monitoring and response (checklist S4)
Branch: dose_experiments
Mapped tests: pytest tests/tools/test_d0_parity_logger.py -q
Artifacts: plans/active/seed/reports/2026-01-22T233418Z/

Do Now (hard validity contract):
- Implement: scripts/tools/d0_parity_logger.py::write_markdown (loop over every dataset and render raw/normalized/grouped tables with grouped scan counts); tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs + tests/tools/test_d0_parity_logger.py::test_cli_limit_datasets_filters_inputs (assert Markdown coverage + `--limit-datasets` filter); docs/TESTING_GUIDE.md & docs/development/TEST_SUITE_INDEX.md (document the new CLI/test selector + usage).
- Pytest: pytest tests/tools/test_d0_parity_logger.py -q | tee plans/active/seed/reports/2026-01-22T233418Z/pytest_d0_parity_logger.log
- Artifacts: plans/active/seed/reports/2026-01-22T233418Z/ (updated dose_parity_log.json/md/csv + pytest + collect-only logs + CLI stdout).

How-To Map:
1. Edit `scripts/tools/d0_parity_logger.py::write_markdown` so it emits a "Stage-Level Stats by Dataset" section with a sub-heading per dataset and per-stage tables (raw/normalized/grouped) using ~`{value:.6g}` formatting and including grouped `n_unique_scans`/`n_patterns`. Keep existing metadata/dataset tables intact.
2. Update `tests/tools/test_d0_parity_logger.py`: have `test_cli_emits_outputs` create two NPZ files before running `main()` and assert the Markdown lists both dataset sections + all three stage tables, then add `test_cli_limit_datasets_filters_inputs` that copies the fixture file, runs the CLI with `--limit-datasets data_p1e6.npz`, and asserts JSON/Markdown only mention the requested dataset.
3. Refresh `docs/TESTING_GUIDE.md` §Running Tests and `docs/development/TEST_SUITE_INDEX.md` so both explicitly list `pytest tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q` as the maintainer selector for the parity logger CLI, referencing the CLI scope per `inbox/README_prepare_d0_response.md`.
4. Validate the changes locally:
   - `pytest tests/tools/test_d0_parity_logger.py -q | tee plans/active/seed/reports/2026-01-22T233418Z/pytest_d0_parity_logger.log`
   - `pytest --collect-only tests/tools/test_d0_parity_logger.py::test_cli_limit_datasets_filters_inputs -q | tee plans/active/seed/reports/2026-01-22T233418Z/pytest_collect_limit.log`
   - `python scripts/tools/d0_parity_logger.py --dataset-root photon_grid_study_20250826_152459 --baseline-params photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/08-26-2025-16.38.17_baseline_gs1/params.dill --scenario-id PGRID-20250826-P1E5-T1024 --output plans/active/seed/reports/2026-01-22T233418Z | tee plans/active/seed/reports/2026-01-22T233418Z/d0_parity_cli_refresh.log`

Pitfalls To Avoid:
- Keep Markdown ASCII-only and preserve the existing metadata/dataset tables before adding new per-dataset sections.
- Iterate datasets in the sorted order already emitted by `main()` so photon doses stay comparable to the JSON output.
- Format floats consistently (e.g., `:.6g`) and guard against missing stats dictionaries so Markdown renders even when certain arrays are absent.
- Show grouped `n_unique_scans`/`n_patterns` counts explicitly; do not drop them when moving to tables.
- Do not touch dataset loading logic or change the glob/`--limit-datasets` semantics beyond test coverage.
- Use synthetic copies inside the test file—never point the tests at the large real datasets.
- Ensure new assertions work on Linux/macOS CI by avoiding hard-coded paths or locale-dependent formatting.
- When rerunning the CLI on the real dataset, leave earlier evidence directories untouched by writing into the new timestamped artifact path.
- Keep docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md synchronized so both list the selector verbatim.
- Capture pytest/d0 CLI logs into the artifacts directory; no stray files at repo root.

If Blocked:
- If `main()` fails to load the photon-grid datasets or params, capture the full stderr/stdout into `plans/active/seed/reports/2026-01-22T233418Z/blocker.log`, note the failure in `docs/fix_plan.md` Attempts History + `galph_memory.md`, and stop before touching Markdown/doc updates.

Findings Applied (Mandatory):
- No relevant findings in the knowledge base.

Pointers:
- docs/fix_plan.md:6 — Current focus + S4 scope (stage-level Markdown + doc sync).
- inbox/README_prepare_d0_response.md:39 — Maintainer requirement to expose raw/grouped/normalized stats + probe/intensity data.
- specs/data_contracts.md:1 — Defines the NPZ keys (diff3d, probeGuess, scan_index) the CLI parses.
- scripts/tools/d0_parity_logger.py:406 — Existing Markdown writer that currently samples only the first dataset.
- tests/tools/test_d0_parity_logger.py:224 — CLI acceptance test stub to expand for multi-dataset Markdown coverage.
- docs/TESTING_GUIDE.md:7 and docs/development/TEST_SUITE_INDEX.md:5 — Sections to update with the parity logger pytest selector.

Next Up (optional):
- If time remains, refresh `inbox/response_prepare_d0_response.md` with a brief addendum once the richer Markdown + doc updates are in place.

Doc Sync Plan (Conditional):
- After the implementation passes, run `pytest --collect-only tests/tools/test_d0_parity_logger.py::test_cli_limit_datasets_filters_inputs -q | tee plans/active/seed/reports/2026-01-22T233418Z/pytest_collect_limit.log`, attach the log under the artifacts path, and confirm both docs reference the selector verbatim.

Mapped Tests Guardrail:
- Before finalizing, confirm `pytest --collect-only tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q | tee plans/active/seed/reports/2026-01-22T233418Z/pytest_collect_cli.log` reports at least one test collected so the mapped selector stays valid.

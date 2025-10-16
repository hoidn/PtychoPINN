# Full Suite Sentinel Prompt

Purpose
- Run the entire regression suite (`pytest tests/`) to measure current project health.
- Record failing modules and open actionable follow-up items in `docs/fix_plan.md`.

Autonomy Contract
- Do not ask the user what to run; this loop always executes the full suite.
- Work from `docs/fix_plan.md`; append new TODOs for each failing module.
- Keep the loop to two messages: a short "next action" preamble and a final checklist/output summary.

Ground Rules
- No code changes in this loop; gather data and update the plan only.
- Environment: run tests from repo root with `KMP_DUPLICATE_LIB_OK=TRUE pytest -v tests/`.
- Treat every unique failing test module/file as a separate item to track.
- Do not modify tests, tolerances, or ignore failures.

Instructions
<step 0>
- Read `docs/index.md`, `docs/development/testing_strategy.md`, and `docs/fix_plan.md`.
- Ensure `docs/fix_plan.md` contains or append a `## Suite Failures` section to host TODOs from this run.
</step 0>

<step 1>
- Run `KMP_DUPLICATE_LIB_OK=TRUE pytest -v tests/` from the repository root.
- Capture stdout/stderr and collect the list of failing tests grouped by their Python module path (e.g., `tests/test_at_cli_004.py`).
- Note the total passed/failed/skipped counts.
</step 1>

<step 2>
- For each failing module:
  - Summarize the failing test node IDs.
  - Draft a reproduction command (e.g., `KMP_DUPLICATE_LIB_OK=TRUE pytest tests/test_at_cli_004.py -k "TestClass::test_name"`).
  - Add or update a TODO entry under `docs/fix_plan.md` › `## Suite Failures` using a compact block of at most four lines:
    - Line 1: `[Suite Failure][YYYY-MM-DD] <module> — Priority: High — Status: pending`
    - Line 2: `Repro:` followed by the single command
    - Line 3: `Failures:` with comma-separated node IDs or a short error signature
    - Line 4 (optional): `Log:` with the stored pytest log path (omit if unchanged)
- Do not exceed this structure; update the existing entry in-place if the module already has one.
</step 2>

<step 3>
- If a module already has an active suite-failure item, edit the existing four-line block to reflect the new failures (do not add extra lines or duplicate entries).
- Update the `**Last Updated:**` timestamp at the top of `docs/fix_plan.md` to today.
</step 3>

<step 4>
- Produce final output summarizing:
  - Pytest command executed and overall result counts
  - Modules with failures and the fix-plan entry anchors created/updated
  - Confirmation that no code changes were made
</step 4>

Process Hints
- Use `rg "FAILED" -n` on the pytest log to locate module names quickly.
- Store raw pytest output under `reports/YYYY-MM-DD/full-suite/pytest.log` and reference the path in plan entries.

Output Checklist
- Command run and counts (pass/fail/skip)
- Failure modules list with new/updated fix_plan anchors
- Path to stored pytest log
- Statement that no source files were modified

Summary: Capture Phase F3.4 regression evidence under the torch-required policy and confirm no new test failures.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 / Phase F3.4 Regression Verification
Branch: feature/torchapi
Mapped tests: pytest tests/torch/ -v --tb=short; pytest tests/ --ignore=tests/test_benchmark_throughput.py --ignore=tests/test_run_baseline.py -v --tb=line
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T201922Z/{pytest_torch_green.log,pytest_full_green.log,regression_summary.md}
Do Now:
- INTEGRATE-PYTORCH-001 Phase F3 — F3.4 torch suite @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:44-48 (tests: pytest tests/torch/ -v --tb=short): run the torch-only test suite, tee output to pytest_torch_green.log in the artifact directory, and call out results for TestBackendSelection::test_pytorch_unavailable_error and the tf_helper skips in the summary.
- INTEGRATE-PYTORCH-001 Phase F3 — F3.4 full regression @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:44-48 (tests: pytest tests/ --ignore=tests/test_benchmark_throughput.py --ignore=tests/test_run_baseline.py -v --tb=line): execute the full pytest run with the known legacy ignores, capture output to pytest_full_green.log, and note pass/skip/xfail counts versus the guard-removal baseline.
- INTEGRATE-PYTORCH-001 Phase F3 — F3.4 documentation @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:44-48 (tests: none): draft regression_summary.md with command lines, counts, and observations, then mark F3.4 complete in phase_f_torch_mandatory.md and implementation.md, and log Attempt #72 in docs/fix_plan.md with artifact links.
If Blocked: Stop after capturing failing logs, record the failing selectors and traceback snippets in regression_summary.md, and register the blocker plus artifact path in docs/fix_plan.md before leaving plan status untouched.
Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:37-48 — Phase F3.4 is the green gate before documentation work can begin.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T192500Z/migration_plan.md:388-421 — Defines the exact regression suites and success criteria for F3.4.
- docs/fix_plan.md:141-147 — Latest ledger entry directs the program toward F3.4 regression verification next.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T195624Z/skip_rewrite_summary.md:1-190 — Details expected skip behavior after torch-required transition; use it as comparison when reviewing results.
- docs/TESTING_GUIDE.md:5-90 — Authoritative commands and expectations for full-suite and integration tests.
How-To Map:
- mkdir -p plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T201922Z to stage logs and the regression summary before running any commands.
- Run `pytest tests/torch/ -v --tb=short | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T201922Z/pytest_torch_green.log`; after completion, record total pass/skip/xfail counts plus explicit notes on backend selection and tf_helper skips inside regression_summary.md.
- Run `pytest tests/ --ignore=tests/test_benchmark_throughput.py --ignore=tests/test_run_baseline.py -v --tb=line | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T201922Z/pytest_full_green.log`; verify no new failures appear and capture final statistics in the summary table.
- In regression_summary.md, include a comparison against the 203 passed / 13 skipped / 1 xfailed baseline from guard_removal_summary.md, mention any warning deltas, and call out how backend dispatcher tests behaved.
- Update plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md (set F3.4 to [x]) and plans/active/INTEGRATE-PYTORCH-001/implementation.md Phase F row accordingly, then add Attempt #72 to docs/fix_plan.md with links to the new logs and summary.
Pitfalls To Avoid:
- Do not drop the known ignores when running the full suite; those modules still lack dependencies and will cause collection errors unrelated to Phase F.
- Avoid truncating pytest output—ensure tee writes the entire logs so future reviewers can audit counts.
- Do not modify production code or tests during this loop; evidence collection only.
- If any test fails unexpectedly, stop after capturing logs and follow the If Blocked procedure rather than attempting fixes.
- Keep environment variables consistent (no PYTHONPATH stubs this time) so results reflect the torch-required baseline.
- Remember to update each planning document exactly once to prevent conflicting checklist states.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:37-48
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T192500Z/migration_plan.md:388-421
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T195624Z/skip_rewrite_summary.md:137-210
- docs/fix_plan.md:141-147
- docs/TESTING_GUIDE.md:5-90
Next Up: Phase F4 documentation and spec sync once F3.4 green gate is cleared.

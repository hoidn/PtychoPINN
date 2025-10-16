Summary: Kick off TEST-SUITE-TRIAGE Phase A by capturing a clean pytest baseline with full logs and environment metadata.
Mode: none
Focus: [TEST-SUITE-TRIAGE] Restore pytest signal and triage failures
Branch: feature/torchapi
Mapped tests: pytest tests/ -vv
Artifacts: plans/active/TEST-SUITE-TRIAGE/reports/2025-10-16T230539Z/{env.md,pytest.log,summary.md}
Do Now: [TEST-SUITE-TRIAGE] Phase A — Run baseline sweep (pytest tests/ -vv)
If Blocked: Capture `python -m pytest --version` and `which pytest` to `reports/<ts>/env.md`, note blocker in summary, then halt.
Priorities & Rationale:
- docs/TESTING_GUIDE.md:5 — Authoritative command for full-suite execution; align with documented workflow.
- docs/debugging/debugging.md:5 — Triaging must start with contract/config verification before blaming model code.
- docs/findings.md:10 — Known gridsize-related test failures may recur; capture them explicitly for comparison.
- CLAUDE.md:52 — TDD directive requires we log failing selectors before any fixes.
- specs/data_contracts.md:7 — Ensure data-driven failures are cross-checked against canonical contract before refactoring.
How-To Map:
- Activate project env (e.g., `conda activate ptycho`); record command plus `python --version` in env.md.
- From repo root run `pytest tests/ -vv 2>&1 | tee plans/active/TEST-SUITE-TRIAGE/reports/2025-10-16T230539Z/pytest.log`.
- After run, summarize failing selectors and key trace lines into `summary.md`; include counts of pass/xfail/skip/fail.
- Record package snapshot with `python -m pip freeze | sort > plans/active/TEST-SUITE-TRIAGE/reports/2025-10-16T230539Z/requirements.txt`.
- Note command exit code in `summary.md` and flag any immediate blockers (e.g., import errors, missing data files).
Pitfalls To Avoid:
- Do not prune or modify tests; this loop is evidence-only.
- Avoid partial runs (`-k`, `-m`, `--maxfail`)—full sweep required for baseline.
- Keep artifacts in the specified timestamped directory; no ad-hoc paths.
- Do not delete or rewrite existing datasets; follow data contract checks first.
- Resist editing production code or tests; only capture logs and metadata now.
- Watch for environment drift (unexpected GPU usage, MLflow writes); document any side effects.
Pointers:
- docs/TESTING_GUIDE.md:5
- docs/debugging/debugging.md:5
- docs/findings.md:10
- CLAUDE.md:52
- specs/data_contracts.md:7
Next Up: Phase B classification once failure ledger is written.

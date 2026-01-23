Summary: Add SLA breach detection plus regression tests to the inbox scan CLI so we can quantify how long Maintainer <2> has been silent and capture fresh evidence.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach, tests/test_generic_loader.py::test_generic_loader
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T020500Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::scan_inbox — add `--sla-hours` and `--fail-when-breached` CLI flags, pass the threshold into scan_inbox (with injectable `current_time` for tests), compute an `sla_watch` block (threshold, hours since last Maintainer <2> inbound, breached boolean, notes), surface it in the JSON + Markdown summaries, and have `main()` exit with code 2 when `--fail-when-breached` is set and the SLA is breached while `ack_detected` remains false.
- Implement: tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach — add a new pytest module that fabricates a temporary inbox, runs the CLI via `subprocess.run`, and asserts the JSON summary reports breached/not-breached states plus the optional failure exit code; cover both a breach case (>threshold) and a healthy case (<threshold).
- Update: docs/TESTING_GUIDE.md::§2 Test Selectors and docs/development/TEST_SUITE_INDEX.md::Tools — register `tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach` as the maintainer-acknowledgement guard with notes on when to run it, citing the artifact logs once the test passes.
- Run Pytest: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach -q && pytest tests/test_generic_loader.py::test_generic_loader -q (tee logs into $ARTIFACT_ROOT).
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T020500Z/

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T020500Z && mkdir -p "$ARTIFACT_ROOT" "$ARTIFACT_ROOT/inbox_sla_watch" "$ARTIFACT_ROOT/logs".
2. Edit plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py: extend argparse with `--sla-hours` (float) and `--fail-when-breached`, allow scan_inbox() to accept `sla_hours` + optional current_time for testing, compute an `sla_watch` dict after the waiting-clock metrics, add a Markdown "SLA Watch" section, and make `main()` set `exit_code = 2` when `fail_when_breached` is True and the SLA is breached while still lacking an acknowledgement.
3. Create tests/tools/test_check_inbox_for_ack_cli.py that builds a temp inbox + output dir, writes Maintainer <2> / Maintainer <1> markdown stubs, manipulates mtimes via `os.utime`, invokes the CLI with/without `--fail-when-breached`, and inspects the resulting JSON to assert `sla_watch['breached']` toggles as expected (plus exit status 2 when the fail flag is used).
4. pytest tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox.log".
5. pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_collect.log".
6. pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log".
7. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --sla-hours 2.0 --output "$ARTIFACT_ROOT/inbox_sla_watch" | tee "$ARTIFACT_ROOT/inbox_sla_watch/check_inbox.log" (repeat with --fail-when-breached if you want to assert the non-zero exit code; capture stderr/stdout either way).
8. Copy the refreshed inbox_scan_summary.{json,md} plus CLI logs into $ARTIFACT_ROOT/inbox_sla_watch and summarize hours-since + breach status inside docs/fix_plan.md and inbox/response_dose_experiments_ground_truth.md.
9. Update docs/TESTING_GUIDE.md §2 and docs/development/TEST_SUITE_INDEX.md to include the new CLI test selector + usage guidance, citing $ARTIFACT_ROOT/logs/pytest_check_inbox.log after the run succeeds.

Pitfalls To Avoid
- Keep the new CLI flags backward-compatible (defaults should preserve current exit behavior when no SLA threshold is supplied).
- Do not loosen the acknowledgement rule: ack still requires a Maintainer <2> message plus an ack keyword.
- Use UTC timestamps throughout; never localize to system tz when computing waiting-clock or SLA fields.
- Ensure tests do not read the real inbox; synthesize miniature inbox directories under tmp_path and clean up after each run.
- Avoid pulling large bundle assets into new tests; focus on lightweight markdown stubs.
- The SLA breach exit should only trigger when ack is still false; do not raise when ack_detected is already true even if the inbound timestamp is old.
- Keep JSON key ordering stable so prior diff tooling stays usable.
- Environment is frozen: no package installs or conda/pip changes.
- Capture every pytest/CLI log under $ARTIFACT_ROOT/logs or inbox_sla_watch for traceability.
- Treat Maintainer <2> ack arrival as authoritative—if an ack file appears mid-run, stop and fold it into docs instead of forcing SLA failure.

If Blocked
- If CLI or pytest fails, save stderr/stdout as $ARTIFACT_ROOT/logs/blocker.log, note the exact command + failure signature inside docs/fix_plan.md Attempts History and galph_memory, then pause for supervisor guidance before altering the acknowledgement workflow.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:343 — Details of DEBUG-SIM-LINES-DOSE-001.F1 attempts and outstanding acknowledgement requirement.
- docs/fix_plan.md:408 — TODO entry describing the continuing inbox-monitoring obligation.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:1 — Current CLI implementation to extend with SLA tracking.
- docs/TESTING_GUIDE.md:1-60 — Source of the authoritative pytest selector list and documentation update rules.
- docs/development/TEST_SUITE_INDEX.md:1-80 — Index that must include the new CLI selector once tests land.

Next Up (optional)
- If the SLA breach persists without Maintainer <2> reply, draft a Maintainer <1> escalation note referencing the new SLA metrics and attach the latest inbox scan summary.

Doc Sync Plan (Conditional)
- After the CLI tests pass, run `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach -q` (log to $ARTIFACT_ROOT/logs/pytest_check_inbox_collect.log) and update docs/TESTING_GUIDE.md §2 plus docs/development/TEST_SUITE_INDEX.md so the selector + usage guidance match reality.

Mapped Tests Guardrail
- `tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach` must collect via the explicit `pytest --collect-only` step above; treat failures as blocks until fixed.

Normative Math/Physics
- None — this loop only extends the maintainer acknowledgement tooling.

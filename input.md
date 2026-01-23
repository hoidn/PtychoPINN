Summary: Generate a history-dashboard feature for the inbox acknowledgement CLI so we can prove repeated SLA breaches, capture fresh evidence, and send a data-backed follow-up nudging Maintainer <2>.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_summarizes_runs, tests/test_generic_loader.py::test_generic_loader
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T023500Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::parse_args — add `--history-dashboard <path>` (requires `--history-jsonl`), call a new helper after `append_history_*`, and ensure failures bubble with clear errors when the history log is missing while still honoring existing SLA/status/escalation flags.
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::write_history_dashboard — ingest the JSONL history log, compute summary metrics (total scans, ack count, breach count, longest wait, last ack timestamp) plus a table of the latest 10 entries, and emit an idempotent Markdown dashboard to the requested path.
- Implement: tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_summarizes_runs — seed a fake JSONL history file, invoke `write_history_dashboard`, and assert the Markdown contains the derived metrics (Total Scans, Breach Count, Longest Wait) plus the recent timeline rows; keep the import helper local to the test file so we do not affect other suites.
- Update: docs/TESTING_GUIDE.md::Inbox Acknowledgement CLI + docs/development/TEST_SUITE_INDEX.md::Inbox acknowledgement entry — add the new selector/log references next to the SLA/history/snippet/escalation rows after capturing fresh pytest output.
- Record: docs/fix_plan.md::DEBUG-SIM-LINES-DOSE-001.F1 Attempts + inbox/response_dose_experiments_ground_truth.md::Maintainer Status + inbox/followup_dose_experiments_ground_truth_2026-01-23T023500Z.md — summarize the 2026-01-23T023500Z scan (hours since inbound/outbound, SLA breach verdict, dashboard path) and send the follow-up that pastes the escalation note highlights so Maintainer <2> can reply.
- Capture: python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords ack --keywords acknowledged --keywords confirm --keywords received --keywords thanks --sla-hours 2.0 --fail-when-breached --history-jsonl $ARTIFACT_ROOT/inbox_history/inbox_sla_watch.jsonl --history-markdown $ARTIFACT_ROOT/inbox_history/inbox_sla_watch.md --history-dashboard $ARTIFACT_ROOT/inbox_history/inbox_history_dashboard.md --status-snippet $ARTIFACT_ROOT/inbox_status/status_snippet.md --escalation-note $ARTIFACT_ROOT/inbox_status/escalation_note.md --escalation-recipient "Maintainer <2>" --output $ARTIFACT_ROOT/inbox_sla_watch | tee $ARTIFACT_ROOT/inbox_sla_watch/check_inbox.log (expect exit code 2 while the SLA window is still breached).
- Validate: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_summarizes_runs -q | tee $ARTIFACT_ROOT/logs/pytest_history_dashboard.log && pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee $ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log && pytest tests/test_generic_loader.py::test_generic_loader -q | tee $ARTIFACT_ROOT/logs/pytest_loader.log.

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T023500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch}.
2. Edit check_inbox_for_ack.py: extend argparse with --history-dashboard, add validation that it can only be used when --history-jsonl is present, create `write_history_dashboard(jsonl_path: Path, output_path: Path, max_entries: int = 10)` that reads the JSONL file (skip blank/invalid lines), computes total scans, ack count, breach count, longest wait (max hours_since_inbound), most recent ack timestamp, and renders tables (Summary Metrics, SLA Breach Stats, Recent Scans). Reuse sanitize_for_markdown for table cells, and ensure missing history produces a friendly "No history" message.
3. After `append_history_markdown`, invoke `write_history_dashboard` when args.history_dashboard is set so every run refreshes the dashboard; log the output path just like the other writers.
4. Update tests/tools/test_check_inbox_for_ack_cli.py: add a helper to load the CLI module (importlib), introduce `test_history_dashboard_summarizes_runs` that writes two JSONL entries with different hours/breach states, calls the helper, and asserts the Markdown includes "Total Scans | 2", "Breach Count", "Longest Wait", and both timestamps in the table; keep fixtures isolated under tmp_path.
5. Run pytest selectors: first targeted `tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_summarizes_runs`, then the full CLI module, then the loader guard as listed in Validate.
6. Run the CLI with the Capture command to refresh JSON/MD/history/status/escalation/dashboard outputs under $ARTIFACT_ROOT; treat exit code 2 as success because --fail-when-breached is set while waiting for Maintainer <2>.
7. Draft `inbox/followup_dose_experiments_ground_truth_2026-01-23T023500Z.md` pulling metrics from `$ARTIFACT_ROOT/inbox_status/escalation_note.md` + the new history dashboard, reiterate SLA breach length, link to status snippet/dashboard/note, and request acknowledgement.
8. Update docs: append a new "Status as of 2026-01-23T023500Z" section in inbox/response_dose_experiments_ground_truth.md plus an Attempts History entry in docs/fix_plan.md referencing the dashboard + CLI exit; refresh `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` with the new selector/log after capturing the pytest outputs.
9. Stage only the modified tooling/tests/docs/inbox/report artifacts plus the new $ARTIFACT_ROOT tree; leave unrelated dirty files untouched.

Pitfalls To Avoid
- Do not touch production modules; confine code to plans/active/DEBUG-SIM-LINES-DOSE-001/bin/ and tests/docs/inbox.
- Keep all Markdown/JSON outputs ASCII so Maintainer automation stays diff-friendly.
- When parsing JSONL, guard against blank/partial lines so a corrupted entry does not crash the CLI.
- Ensure --history-dashboard requires --history-jsonl; otherwise we risk dashboards built from stale data.
- Preserve existing CLI exit-code semantics (still exit 2 only when --fail-when-breached is set and SLA is breached without ack).
- Tests must never read the real inbox directory; fabricate messages under tmp_path only.
- When updating docs, cite the exact artifact paths/timestamps; do not remove older status sections or reports.
- Avoid committing the tarball or other large binary drops; only reference them.

If Blocked
- Capture the failing command output via `tee $ARTIFACT_ROOT/logs/blocker.log`, note the command + error inside docs/fix_plan.md Attempts and galph_memory, and pause for supervisor guidance before changing scope.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:583 — Active DEBUG-SIM-LINES-DOSE-001.F1 context + new history-dashboard scope.
- docs/TESTING_GUIDE.md:19 — Inbox acknowledgement CLI entries to expand with the dashboard selector.
- docs/development/TEST_SUITE_INDEX.md:13 — Catalog entry for inbox CLI tests that needs the new selector/log update.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:1 — CLI implementation surface to extend.
- tests/tools/test_check_inbox_for_ack_cli.py:1 — Existing SLA/history/snippet/escalation tests to mirror for the history dashboard coverage.
- inbox/response_dose_experiments_ground_truth.md:200 — Maintainer status chronology that must gain the 2026-01-23T023500Z update referencing the dashboard outputs.

Next Up (optional)
1. If Maintainer <2> still does not reply after the new follow-up, prep an escalation package for Maintainer <3> using the dashboard data.
2. Consider scheduling check_inbox_for_ack.py via cron/task runner so SLA breaches are captured automatically every few hours.

Doc Sync Plan (Conditional)
- After code/tests succeed, run `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_summarizes_runs -q | tee "$ARTIFACT_ROOT/logs/pytest_history_dashboard_collect.log"`, attach the log to $ARTIFACT_ROOT, and then update docs/TESTING_GUIDE.md + docs/development/TEST_SUITE_INDEX.md with the selector + log path.

Mapped Tests Guardrail
- The new selector `tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_summarizes_runs` must collect (>0) before finishing; treat collection failures as blockers and rerun after fixes.

Normative Math/Physics
- Not applicable — this loop only touches maintainer-monitoring tooling and documentation.

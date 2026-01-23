Summary: Add an escalation-note output to the inbox acknowledgement CLI so Maintainer <1> can paste a prefilled follow-up citing the SLA breach without reassembling data.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_note_emits_call_to_action, tests/test_generic_loader.py::test_generic_loader
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T021945Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::write_escalation_note — extend argparse with `--escalation-note` (and optional `--escalation-recipient`) plus a helper that writes a Markdown escalation draft covering Summary Metrics, SLA Watch, Action Items, a Proposed Message blockquote that names the recipient/request pattern/hours since inbound, and a timeline table; invoke the helper when the new flag is provided without disturbing existing JSON/Markdown/history writers.
- Implement: tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_note_emits_call_to_action — craft a pytest that stages inbound/outbound messages under tmp_path, runs the CLI with `--sla-hours 2.0 --escalation-note <tmp>/note.md --escalation-recipient "Maintainer <2>"`, and asserts the note contains the expected heading, ack status, SLA breach text, blockquote call-to-action, and timeline row.
- Update: docs/TESTING_GUIDE.md::Inbox Acknowledgement CLI + docs/development/TEST_SUITE_INDEX.md::Inbox acknowledgement entry — document the new selector/log path next to the existing SLA/history/snippet tests after capturing fresh pytest logs.
- Record: docs/fix_plan.md::DEBUG-SIM-LINES-DOSE-001.F1 Attempts + inbox/response_dose_experiments_ground_truth.md::Maintainer Status — summarize the 2026-01-23T021945Z scan (hours since inbound/outbound, SLA breach, escalation note path) and reiterate that Maintainer <2> has not acknowledged the drop.
- Capture: python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords ack --keywords acknowledged --keywords confirm --keywords received --keywords thanks --sla-hours 2.0 --fail-when-breached --history-jsonl $ARTIFACT_ROOT/inbox_history/inbox_sla_watch.jsonl --history-markdown $ARTIFACT_ROOT/inbox_history/inbox_sla_watch.md --status-snippet $ARTIFACT_ROOT/inbox_status/status_snippet.md --escalation-note $ARTIFACT_ROOT/inbox_status/escalation_note.md --output $ARTIFACT_ROOT/inbox_sla_watch | tee $ARTIFACT_ROOT/inbox_sla_watch/check_inbox.log.
- Validate: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_note_emits_call_to_action -q | tee $ARTIFACT_ROOT/logs/pytest_escalation_note.log && pytest tests/test_generic_loader.py::test_generic_loader -q | tee $ARTIFACT_ROOT/logs/pytest_loader.log (re-run the loader guard once the CLI/tests pass).

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T021945Z && mkdir -p "$ARTIFACT_ROOT"/logs "$ARTIFACT_ROOT"/inbox_history "$ARTIFACT_ROOT"/inbox_status "$ARTIFACT_ROOT"/inbox_sla_watch.
2. Edit plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py: add argparse options `--escalation-note` (Path) and `--escalation-recipient` (default "Maintainer <2>"), implement `write_escalation_note(results: dict, Path, str)` mirroring the status-snippet style but with sections for Summary Metrics, SLA Watch, Action Items, Proposed Message (blockquote referencing the recipient, request pattern, hours since inbound), and Timeline; call it from main() after history/snippet writers.
3. Ensure the escalation note gracefully handles cases with no SLA info or when ack is already detected (emit "No escalation required" instead of a breach warning) and reuses `sanitize_for_markdown` for table/blockquote text.
4. Update tests/tools/test_check_inbox_for_ack_cli.py by appending `test_escalation_note_emits_call_to_action`; reuse `create_inbox_file`, run the CLI with the new flags, and assert the Markdown output includes the heading, ack status line, "SLA Breach" text, a blockquote referencing the recipient/request pattern, and a timeline row.
5. pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_note_emits_call_to_action -q | tee "$ARTIFACT_ROOT/logs/pytest_escalation_note.log".
6. pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log" to confirm the existing SLA/history/snippet coverage still passes (optional if the targeted run already covers everything but preferred for regression confidence).
7. pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log" to keep the legacy loader guard current.
8. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords ack --keywords acknowledged --keywords confirm --keywords received --keywords thanks --sla-hours 2.0 --fail-when-breached --history-jsonl "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.jsonl" --history-markdown "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.md" --status-snippet "$ARTIFACT_ROOT/inbox_status/status_snippet.md" --escalation-note "$ARTIFACT_ROOT/inbox_status/escalation_note.md" --output "$ARTIFACT_ROOT/inbox_sla_watch" | tee "$ARTIFACT_ROOT/inbox_sla_watch/check_inbox.log" (expect exit 2 while SLA is still breached; keep the JSON/MD outputs).
9. Update docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md with the new selector/log path, then refresh docs/fix_plan.md Attempts + inbox/response_dose_experiments_ground_truth.md Maintainer Status at the bottom to cite the 2026-01-23T021945Z JSON/MD/snippet/escalation note.
10. Stage the updated scripts/tests/docs plus the new artifacts under $ARTIFACT_ROOT and leave everything else untouched.

Pitfalls To Avoid
- Do not touch production modules outside plans/active/DEBUG-SIM-LINES-DOSE-001/bin/.
- Keep Markdown writers idempotent (overwrite rather than append) and ensure ASCII-only output.
- Tests must never read the real inbox; tmp_path fixtures only.
- Preserve the existing keyword detection/actor parsing logic; the escalation note should build on scan results rather than re-reading files.
- Ensure CLI exits with code 2 only when --fail-when-breached is set and SLA breach conditions still hold.
- Do not move or delete previous reports directories or history logs; add a new timestamp folder instead.
- Maintain the loader pytest guard even though code changes are tooling-only.
- Avoid committing the maintainer inbox contents; only link to them in docs.

If Blocked
- Capture the failing command output via `tee` into $ARTIFACT_ROOT/logs/blocker.log, note the error plus attempted command inside docs/fix_plan.md Attempts History and galph_memory, and pause for supervisor guidance before changing scope.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:573 — DEBUG-SIM-LINES-DOSE-001.F1 Attempts/TODO state and SLA context.
- docs/TESTING_GUIDE.md:19 — Inbox acknowledgement CLI testing matrix to update with the new selector/logs.
- docs/development/TEST_SUITE_INDEX.md:13 — Test suite entry enumerating every CLI selector.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:1 — CLI implementation surface to extend.
- tests/tools/test_check_inbox_for_ack_cli.py:1 — Existing SLA/history/snippet tests to mirror for the escalation note.
- inbox/response_dose_experiments_ground_truth.md:196 — Maintainer Status section referencing the latest wait metrics/snippet links.

Next Up (optional)
1. If the escalation note is accepted, draft the actual Maintainer <1> follow-up referencing the new Markdown snippet.
2. If Maintainer <2> still does not reply after another SLA window, add automation to send scheduled reminders or summarize multi-run history for a Maintainer <3> escalation.

Doc Sync Plan (Conditional)
- After code/tests succeed, run `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_note_emits_call_to_action -q | tee "$ARTIFACT_ROOT/logs/pytest_escalation_note_collect.log"`, attach the log to $ARTIFACT_ROOT, and update both docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md with the selector description + new log path before shipping.

Mapped Tests Guardrail
- `tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_note_emits_call_to_action` must collect (>0) during the Doc Sync Plan; treat any collection failure as a blocker before finishing the loop.

Normative Math/Physics
- Not applicable — this loop only touches maintainer-monitoring tooling and documentation.

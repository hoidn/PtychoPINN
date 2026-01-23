Summary: Add a status-snippet output path to the inbox acknowledgement CLI so we can drop a reusable Markdown block into the maintainer response while refreshing the SLA evidence for the new timestamped reports.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary, tests/tools/test_check_inbox_for_ack_cli.py::test_history_logging_appends_entries, tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach, tests/test_generic_loader.py::test_generic_loader
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::main — add a `--status-snippet` argument (plus helper) that writes a Markdown snippet summarizing generated timestamp, ack status, hours since inbound/outbound, SLA threshold/breach flag, ack files, notes, and a condensed timeline table; ensure helpers create parent dirs, sanitize table text, and do not mutate the scan results before JSON/Markdown writers run.
- Implement: tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary — craft a new pytest that runs the CLI with `--status-snippet` on a temp inbox, then asserts the snippet file exists and contains the expected metadata (Ack Detected line, SLA breach line, timeline row for Maintainer <2>).
- Update: docs/TESTING_GUIDE.md::Inbox Acknowledgement CLI — list the new `test_status_snippet_emits_wait_summary` selector with an explanation of the snippet output plus fresh log pointers under the 2026-01-23T015222Z artifacts.
- Update: docs/development/TEST_SUITE_INDEX.md::Inbox acknowledgement CLI entry — add the status-snippet selector/log references alongside the SLA and history tests.
- Document: docs/fix_plan.md::DEBUG-SIM-LINES-DOSE-001.F1 Attempts + inbox/response_dose_experiments_ground_truth.md::Maintainer Status — capture the 2026-01-23T015222Z scan results (hours since inbound/outbound, SLA breach, snippet path) and note that we are still waiting on Maintainer <2>.

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z && export LOG_DIR="$ARTIFACT_ROOT/logs" && export STATUS_DIR="$ARTIFACT_ROOT/inbox_status" && export HISTORY_DIR="$ARTIFACT_ROOT/inbox_history" && export SLA_DIR="$ARTIFACT_ROOT/inbox_sla_watch" && mkdir -p "$LOG_DIR" "$STATUS_DIR" "$HISTORY_DIR" "$SLA_DIR".
2. Edit plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py: extend argparse with `--status-snippet`, add `write_status_snippet(results, Path)` that writes the Markdown block described above (heading, key metrics, optional timeline table), call it from main() after the history writers, and ensure helper functions reuse the existing sanitize logic so links render correctly.
3. Update tests/tools/test_check_inbox_for_ack_cli.py by appending `test_status_snippet_emits_wait_summary`, reusing `create_inbox_file` to stage inbound/outbound messages, running the CLI with `--sla-hours 2.0 --status-snippet $tmp/status.md`, and asserting the snippet text contains "Maintainer Status Snapshot", the ack/no-ack sentence, the SLA breach note, and at least one timeline row.
4. pytest tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary -q | tee "$LOG_DIR/pytest_status_snippet.log".
5. pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_logging_appends_entries -q | tee "$LOG_DIR/pytest_check_inbox_history.log" && pytest tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach -q | tee "$LOG_DIR/pytest_check_inbox.log".
6. pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$LOG_DIR/pytest_loader.log" to keep the loader guard current.
7. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --sla-hours 2.0 --history-jsonl "$HISTORY_DIR/inbox_sla_watch.jsonl" --history-markdown "$HISTORY_DIR/inbox_sla_watch.md" --status-snippet "$STATUS_DIR/status_snippet.md" --output "$SLA_DIR" | tee "$SLA_DIR/check_inbox.log".
8. (Optional) Repeat Step 7 with `--fail-when-breached` into "$SLA_DIR/check_inbox_fail.log" and expect exit 2 while preserving the snippet output.
9. Update docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md to cover the new selector/logs (point both to `$LOG_DIR/pytest_status_snippet.log` and the new snippet feature description); cite the Artifact root so reviewers know where to look.
10. Append a DEBUG-SIM-LINES-DOSE-001.F1 entry in docs/fix_plan.md plus a new "Status as of 2026-01-23T015222Z" subsection in inbox/response_dose_experiments_ground_truth.md linking to `$STATUS_DIR/status_snippet.md`, the JSON/Markdown summaries, and the SLA numbers; reiterate that the acknowledgement is still pending.

Pitfalls To Avoid
- Do not weaken the Maintainer <2> + ack keyword rules when detecting acknowledgements.
- Keep the status snippet writer idempotent; headers should not duplicate if the file already exists.
- Ensure new helper functions live in the same script (no imports from production modules) and honor ASCII-only output.
- Tests must operate entirely inside tmp_path fixtures; never read the real inbox during pytest runs.
- Capture every pytest/CLI command output via tee so logs land under $LOG_DIR or $SLA_DIR.
- Do not edit user-managed inbox files except via append-only maintainer notes.
- Avoid deleting or rewriting older reports directories when creating the new timestamped artifacts.
- Leave the existing history log format untouched so previous evidence remains comparable.

If Blocked
- If the new CLI flag or tests fail, send stdout/stderr to "$LOG_DIR/blocker.log", note the command and failure in docs/fix_plan.md Attempts History plus galph_memory, and pause further changes until the supervisor re-scopes the focus.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:1 — CLI entry point to extend with the status snippet flag and helper.
- tests/tools/test_check_inbox_for_ack_cli.py:1 — Existing SLA/history tests to mirror when adding the new status-snippet test.
- docs/fix_plan.md:408 — Active DEBUG-SIM-LINES-DOSE-001.F1 Attempts History and TODO context.
- docs/TESTING_GUIDE.md:19 — Inbox acknowledgement CLI sections that must list every selector/log path.
- docs/development/TEST_SUITE_INDEX.md:13 — Test index entry for the inbox acknowledgement suite.
- inbox/response_dose_experiments_ground_truth.md:196 — Maintainer Status section to update with the new snippet and wait metrics.

Next Up (optional)
1. If Maintainer <2> replies, log the acknowledgement path and close DEBUG-SIM-LINES-DOSE-001.F1.
2. If silence persists past the next SLA window, draft an escalation note referencing the shared history/snippet artifacts.

Doc Sync Plan (Conditional)
- After all code/tests pass, run `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary -q | tee "$LOG_DIR/pytest_status_snippet_collect.log"`, then update docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md with the selector description plus the new log paths (full command + artifact references) before finishing the loop.

Mapped Tests Guardrail
- `tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary` must collect successfully per the Doc Sync Plan; treat any collection failure as a blocker until resolved.

Normative Math/Physics
- Not applicable — maintainer-monitoring tooling only.

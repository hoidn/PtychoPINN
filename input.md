Summary: Extend the inbox acknowledgement CLI with persistent history logging so we can show Maintainer <2> the full wait timeline and capture fresh SLA evidence under the new artifacts drop.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: tests/tools/test_check_inbox_for_ack_cli.py::test_history_logging_appends_entries, tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach, tests/test_generic_loader.py::test_generic_loader
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T014011Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::main — add `--history-jsonl`/`--history-markdown` flags, create helpers to append JSONL + Markdown history rows (generated UTC, ack status, hours-since inbound/outbound, SLA breach flag, ack files), ensure parent directories exist, and invoke them after summary emission so every run leaves an auditable timeline without disturbing the existing ack/SLA logic.
- Implement: tests/tools/test_check_inbox_for_ack_cli.py::test_history_logging_appends_entries — add a pytest that fabricates a temp inbox, runs the CLI twice (before/after injecting a maintainer ack) with the new history flags, and asserts JSONL accumulated two entries plus the Markdown log grew by two data rows while the second entry flips `ack_detected` to true.
- Update: docs/TESTING_GUIDE.md::Inbox Acknowledgement CLI — document the new history selector alongside the SLA test, including fresh artifact log pointers after this run.
- Update: docs/development/TEST_SUITE_INDEX.md::Tools — add the new `test_history_logging_appends_entries` selector details + log path under the inbox CLI section.
- Document: docs/fix_plan.md::DEBUG-SIM-LINES-DOSE-001.F1 Attempts — append a row for the 2026-01-23T014011Z run covering the history logging upgrade, the new CLI/test results, and the still-missing Maintainer <2> acknowledgement with SLA breach stats.
- Update: inbox/response_dose_experiments_ground_truth.md::Maintainer Status — extend the status block with a short summary of the new history log (hours since inbound, breach flag, artifact paths) so Maintainer <2> sees the wait evidence inline.

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T014011Z && export HISTORY_DIR="$ARTIFACT_ROOT/inbox_history" && mkdir -p "$ARTIFACT_ROOT" "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/inbox_sla_watch" "$HISTORY_DIR".
2. Edit plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py: extend argparse with `--history-jsonl` and `--history-markdown`, add `append_history_jsonl()`/`append_history_markdown()` helpers (create parent dirs, add Markdown header once, sanitize notes), and call them from `main()` after writing the JSON/Markdown summaries; history rows must capture generated UTC, ack boolean, hours since inbound/outbound, SLA breach state, and ack file names.
3. Update tests/tools/test_check_inbox_for_ack_cli.py by adding `test_history_logging_appends_entries`: reuse `create_inbox_file`, run the CLI twice with `--history-jsonl $tmp/history.jsonl --history-markdown $tmp/history.md`, and assert the JSONL has two entries (one ack false, one true) plus the Markdown table contains two data rows with the expected Yes/No fields; keep runtimes short.
4. pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_logging_appends_entries -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_history.log".
5. pytest tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox.log" to guard the earlier SLA behaviour.
6. pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_history_logging_appends_entries -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_history_collect.log" (Doc Sync guardrail).
7. pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log" for the baseline loader check.
8. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --sla-hours 2.0 --history-jsonl "$HISTORY_DIR/inbox_sla_watch.jsonl" --history-markdown "$HISTORY_DIR/inbox_sla_watch.md" --output "$ARTIFACT_ROOT/inbox_sla_watch" | tee "$ARTIFACT_ROOT/inbox_sla_watch/check_inbox.log" (captures the new summary plus appends the history files).
9. (Optional but encouraged) python .../check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --sla-hours 2.0 --fail-when-breached --output "$ARTIFACT_ROOT/inbox_sla_watch_fail" > "$ARTIFACT_ROOT/inbox_sla_watch_fail/check_inbox_fail.log" 2>&1 || test $? -eq 2 to prove the exit code path still works with the new history hooks.
10. Copy the refreshed inbox_scan_summary.{json,md}, the CLI stdout logs, and both history files into $ARTIFACT_ROOT/inbox_sla_watch/ alongside the jsonl/md history in $HISTORY_DIR.
11. Update docs/TESTING_GUIDE.md §Inbox Acknowledgement CLI with the new selector + artifact log path (2026-01-23T014011Z) and mention the history logging scope; mirror the same selector/log entry under docs/development/TEST_SUITE_INDEX.md.
12. Summarize this loop in docs/fix_plan.md (Attempts History) and in inbox/response_dose_experiments_ground_truth.md Maintainer Status referencing the new history files + SLA numbers so Maintainer <2> sees the wait evidence inline.

Pitfalls To Avoid
- Do not let the history helpers mutate or filter `results`; they should only append metadata and leave main summaries untouched.
- Ensure Markdown headers are written exactly once even when the file already exists; avoid duplicating the header on every run.
- Keep history rows UTC-based and sanitize any newline/pipe characters before writing to the Markdown table so rendering stays intact.
- Respect the existing acknowledgement rule (Maintainer <2> + ack keyword) — no heuristic relaxations.
- Never exit non-zero for SLA breaches unless `--fail-when-breached` is explicitly passed; the default command should still exit 0.
- Tests must operate entirely inside tmp_path fixtures; never scan the real inbox directory inside pytest.
- Do not relocate or delete existing artifacts under reports/2026-01-23T020500Z; append new evidence under the new timestamp only.
- Avoid committing large inbox artifacts twice; point docs to the consolidated $ARTIFACT_ROOT paths.

If Blocked
- If pytest or the CLI fails, capture the full command + stderr into "$ARTIFACT_ROOT/logs/blocker.log", summarize the failure inside docs/fix_plan.md Attempts History and galph_memory, and pause for supervisor guidance before altering the acknowledgement workflow.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:408 — Active DEBUG-SIM-LINES-DOSE-001.F1 attempts/TODO context for the inbox monitoring effort.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:1 — Current CLI implementation to extend with history logging.
- tests/tools/test_check_inbox_for_ack_cli.py:1 — Existing SLA watch tests to expand with the new history test.
- docs/TESTING_GUIDE.md:1 — Authoritative commands + selector documentation that must reference the new test/logs.
- docs/development/TEST_SUITE_INDEX.md:1 — Test index entry that must list both inbox CLI selectors/logs.

Next Up (optional)
1. If Maintainer <2> remains silent after this loop, draft an escalation note referencing the history log and SLA breach stats.
2. Once acknowledgement arrives, close DEBUG-SIM-LINES-DOSE-001 by updating docs/fix_plan.md and the maintainer inbox with the ack path.

Doc Sync Plan (Conditional)
- After both inbox CLI tests pass, run `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_history_logging_appends_entries -q` (Step 6) and embed the resulting log path inside docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md so the registry reflects the new selector.

Mapped Tests Guardrail
- `tests/tools/test_check_inbox_for_ack_cli.py::test_history_logging_appends_entries` must collect successfully via the Step 6 command before finishing the loop; treat collection failures as blockers.

Normative Math/Physics
- Not applicable — this task only touches maintainer-tooling scripts.

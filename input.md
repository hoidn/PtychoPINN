Summary: Add per-actor wait metrics to the inbox acknowledgement CLI so we can show Maintainer <2>/<3> coverage and capture a fresh SLA breach evidence drop.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: pytest tests/tools/test_check_inbox_for_ack_cli.py -q; pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T031500Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::scan_inbox + write_markdown_summary + write_status_snippet + write_escalation_note — emit an `ack_actor_stats` block that tracks last inbound timestamps, hours since inbound, inbound counts, and ack file lists per normalized actor (default Maintainer <2> only) and surface those stats in JSON/Markdown/status outputs instead of hard-coding "Maintainer <2>".
- Implement: tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_wait_metrics_cover_each_actor — add a regression that creates Maintainer <2>/<3> messages with different modified times, runs the CLI with both `--ack-actor` flags, and asserts the JSON summary exposes distinct wait metrics for each actor plus no ack detection unless keywords match.
- Update: docs/TESTING_GUIDE.md::Inbox acknowledgement CLI section + docs/development/TEST_SUITE_INDEX.md::Inbox acknowledgement entry + docs/fix_plan.md (Attempts) + inbox/response_dose_experiments_ground_truth.md + inbox/followup_dose_experiments_ground_truth_2026-01-23T031500Z.md — reference the per-actor wait tables, new selectors/log paths, and refreshed SLA breach evidence once the CLI run completes.
- Capture: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T031500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch}; python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords acknowledged --keywords confirm --keywords received --keywords thanks --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" --sla-hours 2.0 --fail-when-breached --history-jsonl "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.jsonl" --history-markdown "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.md" --history-dashboard "$ARTIFACT_ROOT/inbox_history/inbox_history_dashboard.md" --status-snippet "$ARTIFACT_ROOT/inbox_status/status_snippet.md" --escalation-note "$ARTIFACT_ROOT/inbox_status/escalation_note.md" --output "$ARTIFACT_ROOT/inbox_sla_watch" | tee "$ARTIFACT_ROOT/logs/check_inbox.log" (exit 2 expected due to SLA breach).
- Validate: pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log" && pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log".

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T031500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch}.
2. Modify `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py`: add an `ack_actor_stats` dict inside `scan_inbox`, capture per-actor wait metrics + ack files, include the new block in the JSON result, update `write_markdown_summary` with an "Ack Actor Coverage" table, and teach `write_status_snippet`/`write_escalation_note`/CLI stdout to read the new structure so Maintainer <2>/<3> waits are documented explicitly.
3. Extend `tests/tools/test_check_inbox_for_ack_cli.py` with `test_ack_actor_wait_metrics_cover_each_actor`, fabricating Maintainer <2>/<3> messages under tmp_path, invoking the CLI with both `--ack-actor` flags, and asserting the JSON summary exposes two distinct entries under `ack_actor_stats` with different `hours_since_last_inbound` values and matching inbound counts.
4. Run `pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log"` followed by `pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log"`; keep logs under `$ARTIFACT_ROOT/logs/`.
5. Execute the CLI command under Capture (exit 2 is OK) to refresh JSON/Markdown/history/dashboard/status/escalation outputs inside `$ARTIFACT_ROOT`, then append the new per-actor wait stats to `inbox/response_dose_experiments_ground_truth.md` and author `inbox/followup_dose_experiments_ground_truth_2026-01-23T031500Z.md` summarizing Maintainer <2>/<3> wait times + SLA breach.
6. Update `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` with the new selector (`test_ack_actor_wait_metrics_cover_each_actor`) + log paths (use `$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log` for the suite and `$ARTIFACT_ROOT/logs/pytest_ack_actor_wait_collect.log` for the collect-only run), refresh `docs/fix_plan.md` Attempts with the artifacts timestamp, and ensure `$ARTIFACT_ROOT/summary.md` contains the final Turn Summary block.

Pitfalls To Avoid
- Stay within `plans/active/DEBUG-SIM-LINES-DOSE-001`, `tests/tools/`, `docs/`, and `inbox/`; do not touch shipped production modules.
- Preserve backwards compatibility: default ack actor remains Maintainer <2> when `--ack-actor` is omitted and legacy JSON fields (`is_from_maintainer_2`) must stay intact.
- Keep `ack_actor_stats` deterministic (sorted by actor id) and ensure Markdown tables remain ASCII-friendly; no emoji or wide Unicode.
- Do not recompute historical artifacts; only append new entries under `reports/2026-01-23T031500Z/`.
- Ensure the CLI still works without `--history-jsonl`/`--history-dashboard`; guard optional arguments carefully.
- Tests must operate solely on tmp_path inbox fixtures (no reads from the real `inbox/`).
- When updating docs, cite the new log paths under the current timestamp and leave prior entries untouched.
- CLI run will exit 2 because `--fail-when-breached` is set—treat it as success; only treat other non-zero codes as blockers.
- When drafting the new follow-up, avoid deleting the prior follow-ups and clearly cite the per-actor wait hours pulled from the latest scan.

If Blocked
- Capture the failing command with `tee "$ARTIFACT_ROOT/logs/blocker.log"`, note stderr plus attempted command, and add a brief block reason under `docs/fix_plan.md` Attempts + `galph_memory` before pausing.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:7 — DEBUG-SIM-LINES-DOSE-001 summary plus new per-actor wait action items that must be updated after the run.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:500 — Waiting-clock + Markdown summary sections where "Maintainer <2>" is hard-coded.
- tests/tools/test_check_inbox_for_ack_cli.py:777 — Recent ack-actor + keyword regressions to mirror for the new per-actor wait metrics test.
- docs/TESTING_GUIDE.md:18 — Inbox acknowledgement CLI coverage list that needs the new selector/log references.
- docs/development/TEST_SUITE_INDEX.md:12 — Test catalog entry that must mention the per-actor wait metrics regression.
- inbox/response_dose_experiments_ground_truth.md:200 — Maintainer status section to extend with the new per-actor wait tables.

Next Up (optional)
1. If Maintainer <2> and <3> stay silent after the per-actor evidence drop, prep an escalation note addressed to Maintainer <3> using the CLI’s `--escalation-recipient` flag.
2. Consider scripting periodic runs (cron-style) so we log SLA breaches without manual intervention once acknowledgement finally lands.

Doc Sync Plan (Conditional)
- After the suite passes, run `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_wait_metrics_cover_each_actor -q | tee \"$ARTIFACT_ROOT/logs/pytest_ack_actor_wait_collect.log\"`, archive the log, and then update docs/TESTING_GUIDE.md + docs/development/TEST_SUITE_INDEX.md with the selector/log references under the new timestamp.

Mapped Tests Guardrail
- Ensure `tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_wait_metrics_cover_each_actor` collects (>0) before finishing; treat collection failures as blockers alongside the full-suite run listed above.

Normative Math/Physics
- Not applicable — maintainer-monitoring tooling only.

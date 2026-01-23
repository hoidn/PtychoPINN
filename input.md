Summary: Extend the inbox acknowledgement CLI with per-actor SLA severity/deadline fields and refresh the DEBUG-SIM-LINES-DOSE-001.F1 evidence + follow-up bundle so Maintainer <2>/<3> see the current breach state.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_metrics_include_deadline -q; pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T050500Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::scan_inbox — add per-actor `sla_deadline_utc`, `sla_breached`, `sla_breach_duration_hours`, `sla_severity`, and notes when `--sla-hours` is set, and thread those fields through write_markdown_summary/status_snippet/escalation_note plus CLI stdout so every Markdown/JSON surface shows Maintainer <2>/<3> wait severities.
- Implement (test): tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_metrics_include_deadline — fabricate Maintainer <2> (3.5h old) + Maintainer <3> (no inbound) messages, run the CLI with `--ack-actor` flags, and assert JSON/Markdown both expose per-actor deadlines, breach durations, severity labels, and "unknown" entries for actors without inbound mail.
- Update: docs/TESTING_GUIDE.md §Inbox Acknowledgement CLI, docs/development/TEST_SUITE_INDEX.md entry, docs/fix_plan.md Attempts history, inbox/response_dose_experiments_ground_truth.md (add 2026-01-23T024800Z/031500Z/040500Z/050500Z statuses), and author inbox/followup_dose_experiments_ground_truth_2026-01-23T050500Z.md referencing the new per-actor severity tables.
- Capture: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T050500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch}; python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords acknowledged --keywords confirm --keywords received --keywords thanks --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" --sla-hours 2.0 --fail-when-breached --history-jsonl "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.jsonl" --history-markdown "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.md" --history-dashboard "$ARTIFACT_ROOT/inbox_history/inbox_history_dashboard.md" --status-snippet "$ARTIFACT_ROOT/inbox_status/status_snippet.md" --escalation-note "$ARTIFACT_ROOT/inbox_status/escalation_note.md" --escalation-recipient "Maintainer <2>" --output "$ARTIFACT_ROOT/inbox_sla_watch" | tee "$ARTIFACT_ROOT/logs/check_inbox.log" (expect exit code 2 because `--fail-when-breached` stays enabled while waiting for ack).
- Validate: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_metrics_include_deadline -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log"; pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log".
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T050500Z/ (inbox_sla_watch/, inbox_status/, inbox_history/, logs/, summary.md, updated docs/inbox files referenced inside)

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T050500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch} so every log/summary lands under the new timestamp.
2. Update `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py`: in `scan_inbox()` compute per-actor SLA stats whenever `sla_hours` is provided (deadline, breach flag/duration, severity, and short notes for actors without inbound). Propagate the new fields into the stored `ack_actor_stats` dict, augment the Markdown tables (summary + status snippet + escalation note) with Deadline/Breach/Severity columns, and expand CLI stdout so the “Ack Actor Coverage” block prints the extra fields. Guard `None` cases so actors with zero inbound produce `deadline_utc=None`, `severity="unknown"`, and `notes="No inbound..."` instead of throwing.
3. Add `test_ack_actor_sla_metrics_include_deadline` to `tests/tools/test_check_inbox_for_ack_cli.py`: create a temp inbox with a Maintainer <2> inbound file 3+ hours old (no ack keywords) and no Maintainer <3> inbound, run the CLI with `--ack-actor` flags plus `--sla-hours 2.0`, and assert JSON `ack_actor_stats['maintainer_2']` shows `sla_deadline_utc`, `sla_breach_duration_hours` > 1, `sla_severity == "critical"` while `ack_actor_stats['maintainer_3']` reports `sla_severity == "unknown"`. Verify `inbox_scan_summary.md` contains the expanded table headers (Deadline/Breach/Severity/Notes).
4. Run `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_metrics_include_deadline -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log"` followed by `pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log"`; treat any failure as a blocker and skip the CLI run until the suite is green.
5. Execute the Capture command to re-scan `inbox/` with Maintainer <2>/<3> ack actors; keep all JSON/Markdown/snippet/history/log outputs under `$ARTIFACT_ROOT` and accept exit 2 from `--fail-when-breached`.
6. Refresh docs: add the new selector + log paths to `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md`, append a new Attempt entry in `docs/fix_plan.md` referencing `$ARTIFACT_ROOT`, extend `inbox/response_dose_experiments_ground_truth.md` with the missing 2026-01-23T024800Z/031500Z/040500Z/050500Z status sections, and drop `inbox/followup_dose_experiments_ground_truth_2026-01-23T050500Z.md` quoting the per-actor severity tables. Copy the Turn Summary (end of this loop) into `$ARTIFACT_ROOT/summary.md` once everything is complete.

Pitfalls To Avoid
- Do not modify production modules outside `plans/active/DEBUG-SIM-LINES-DOSE-001`, `tests/tools/`, `docs/`, and `inbox/`.
- Preserve backwards compatibility: when `--sla-hours` is omitted or only Maintainer <2> is configured, old outputs should remain unchanged.
- Always emit ISO8601 timestamps with timezone info; never drop the `+00:00` when computing deadlines.
- Keep the per-actor notes concise ASCII (no tabs/Unicode) so Markdown tables stay readable.
- Only the CLI run may exit 2 (due to `--fail-when-breached`); every pytest/doc command must exit 0.
- Don’t clobber the existing history JSONL/Markdown headers; append new rows but keep prior evidence intact.
- Ensure Maintainer <3> continues to show "N/A"/"unknown" when no inbound mail exists—no fake ack detection.
- Update every doc/test reference (GUIDE + INDEX + maintainer response) or the next loop will have to redo documentation.

If Blocked
- Stop immediately, capture the failing command + stderr with `tee "$ARTIFACT_ROOT/logs/blocker.log"`, and log the blocker in `docs/fix_plan.md` Attempts plus `galph_memory.md` so we can decide whether to escalate or retarget next turn.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:745 — Current DEBUG-SIM-LINES-DOSE-001.F1 scope plus the per-actor SLA severity next actions.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:360 — SLA watch + ack_actor_stats logic that needs the new per-actor severity data.
- docs/TESTING_GUIDE.md:19 — Inbox acknowledgement CLI selector docs to update with the new regression + log paths.
- docs/development/TEST_SUITE_INDEX.md:13 — Test registry entry where the new selector/log must be listed.
- inbox/response_dose_experiments_ground_truth.md:200 — Maintainer status table that must gain the missing 024800Z/031500Z/040500Z/050500Z sections.

Next Up (optional)
- If Maintainer <3> still stays silent after this escalation, scope an automated hourly cron wrapper around the CLI so we can capture breach duration deltas without manual reruns.

Doc Sync Plan (Conditional)
- After the code/test updates pass, run `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_metrics_include_deadline -q | tee "$ARTIFACT_ROOT/logs/pytest_sla_metrics_collect.log"` to archive the selector, then update `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` with both the suite and collect-only log paths.

Mapped Tests Guardrail
- Ensure `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_metrics_include_deadline -q` collects (>0) before finishing; treat collection failures as blockers.

Normative Math/Physics
- Not applicable — monitoring/tooling only; no changes to physics or forward-model specs.

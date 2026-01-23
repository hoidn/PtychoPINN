Summary: Surface a per-actor SLA summary inside the inbox acknowledgement CLI and capture a fresh evidence bundle that proves Maintainer <2> is still breaching the delivery window while Maintainer <3> remains within their override threshold.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_summary_flags_breach -q; pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T070500Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::scan_inbox — emit an `ack_actor_summary` structure that groups each monitored actor by severity (critical/warning/ok/unknown) with hours-since-inbound, threshold, deadline, and notes so the CLI, Markdown summary, status snippet, escalation note, and stdout can immediately list which actor is breaching which SLA override; keep the existing tables intact and ensure summary generation works whenever `--sla-hours` (and optional `--ack-actor-sla`) are set.
- Implement (test): tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_summary_flags_breach — craft a synthetic inbox where Maintainer <2> is 3.5 h stale (breach) and Maintainer <3> is 1.0 h stale (within a 4–6 h override), run the CLI with both `--sla-hours` and `--ack-actor-sla` flags, and assert the JSON `ack_actor_summary` buckets, Markdown “Ack Actor SLA Summary” section, and CLI stdout text all highlight the correct actor/category pairings.
- Update: docs/TESTING_GUIDE.md (add the new selector + artifact log under the Inbox CLI section), docs/development/TEST_SUITE_INDEX.md (same selector/log), docs/fix_plan.md Attempts (log the summary scope once shipped), inbox/response_dose_experiments_ground_truth.md (append a 2026-01-23T070500Z status block that cites the per-actor SLA summary + artifact links), and author inbox/followup_dose_experiments_ground_truth_2026-01-23T070500Z.md referencing the latest SLA evidence so Maintainer <2> can’t miss the breach callout.
- Capture: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T070500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch}; python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords acknowledged --keywords confirm --keywords received --keywords thanks --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" --sla-hours 2.5 --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" --fail-when-breached --history-jsonl "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.jsonl" --history-markdown "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.md" --history-dashboard "$ARTIFACT_ROOT/inbox_history/inbox_history_dashboard.md" --status-snippet "$ARTIFACT_ROOT/inbox_status/status_snippet.md" --escalation-note "$ARTIFACT_ROOT/inbox_status/escalation_note.md" --escalation-recipient "Maintainer <2>" --output "$ARTIFACT_ROOT/inbox_sla_watch" | tee "$ARTIFACT_ROOT/logs/check_inbox.log" (exit 2 expected because Maintainer <2> stays in breach).
- Validate: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_summary_flags_breach -q | tee "$ARTIFACT_ROOT/logs/pytest_ack_actor_summary.log"; pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log"; after code passes, run pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_summary_flags_breach -q | tee "$ARTIFACT_ROOT/logs/pytest_ack_actor_summary_collect.log" so the new selector is recorded.
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T070500Z/ (JSON/Markdown summaries, history/status/escalation outputs, pytest logs, follow-up note, response update)

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T070500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch} to stage the new drop.
2. Update plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py — augment scan_inbox with an `ack_actor_summary` dict (critical/warning/ok/unknown buckets plus entry metadata), thread it through write_json_summary consumers (Markdown summary, status snippet, escalation note, CLI stdout) so each output gains a concise “Ack Actor SLA Summary” section that spells out which actor is late versus within SLA, and keep newline/Markdown sanitization consistent.
3. Add tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_summary_flags_breach next to the other SLA tests; reuse create_inbox_file fixtures, run the CLI with both ack actors + overrides, and assert the JSON bucket contents, Markdown summary text, and stdout lines mention the correct actor names, thresholds, and severity words.
4. Run pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_summary_flags_breach -q | tee "$ARTIFACT_ROOT/logs/pytest_ack_actor_summary.log", followed by pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_summary_flags_breach -q | tee "$ARTIFACT_ROOT/logs/pytest_ack_actor_summary_collect.log" and pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log".
5. Execute the capture command above so JSON/Markdown/status/escalation/history outputs under "$ARTIFACT_ROOT" clearly indicate Maintainer <2> is "critical" (3.7h > 2.0h) while Maintainer <3> remains "ok" under the 6.0h override; ensure --fail-when-breached propagates exit code 2.
6. Update docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/fix_plan.md Attempts, inbox/response_dose_experiments_ground_truth.md (add a 2026-01-23T070500Z status, cite summary files + tarball SHA), and write inbox/followup_dose_experiments_ground_truth_2026-01-23T070500Z.md that quotes the new summary; then copy this loop’s Turn Summary into "$ARTIFACT_ROOT/summary.md" once everything passes.

Pitfalls To Avoid
- Keep the existing Ack Actor Coverage tables and history writers intact; the new summary is additive.
- Do not treat Maintainer <3> as acknowledged unless keywords match; the summary must honor the ack_actors + keywords filters.
- Preserve Markdown sanitization so summary bullets never leak pipes/newlines into tables.
- Ensure CLI stdout still exits 2 when `--fail-when-breached` is set; don’t mask the exit code when printing the new summary.
- Append to history JSONL/Markdown instead of truncating earlier entries (the new scan should become the latest row only).
- Record the `ack_actor_sla_hours` overrides in the JSON summary and mention them inside the maintainer docs/follow-up to keep provenance clear.
- Keep paths under plans/active/DEBUG-SIM-LINES-DOSE-001/; never touch production modules or user-owned data directories.
- Maintain deterministic ordering (critical → warning → ok → unknown) so tests aren’t flaky.

If Blocked
- Capture the failing command + stderr with `tee "$ARTIFACT_ROOT/logs/blocker.log"`, stop further edits, and log the blocker + error signature inside docs/fix_plan.md Attempts plus galph_memory.md so we can decide whether to escalate or pivot focus.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:700 — Current DEBUG-SIM-LINES-DOSE-001.F1 attempts + SLA instrumentation scope.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:320 — scan_inbox + downstream writers that need the new summary.
- tests/tools/test_check_inbox_for_ack_cli.py:1321 — Latest per-actor SLA test to mirror when adding the summary regression.
- docs/TESTING_GUIDE.md:150 — Inbox acknowledgement CLI selectors/log references that must mention the new summary test path.
- inbox/response_dose_experiments_ground_truth.md:198 — Maintainer status log to extend with the 2026-01-23T070500Z summary + artifact links.

Next Up (optional)
- If Maintainer <2> acknowledges after this drop, close DEBUG-SIM-LINES-DOSE-001.F1 by logging the reply and archiving the SLA monitor artifacts.

Doc Sync Plan (Conditional)
- After the new test passes, run `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_summary_flags_breach -q | tee "$ARTIFACT_ROOT/logs/pytest_ack_actor_summary_collect.log"`, then update docs/TESTING_GUIDE.md §Inbox acknowledgement CLI and docs/development/TEST_SUITE_INDEX.md with the selector + log path before finishing the loop.

Mapped Tests Guardrail
- Treat any failure of `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_summary_flags_breach -q` as a blocker—collection must succeed before shipping this loop.

Normative Math/Physics
- Not applicable; reference docs/TESTING_GUIDE.md for CLI behavior rather than the physics specs.

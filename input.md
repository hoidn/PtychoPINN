Summary: Add per-actor SLA severity trends to the inbox history dashboard and capture a fresh evidence bundle (2026-01-23T093500Z) that shows the sustained breach while keeping Maintainer <3> visibility.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_severity_trends -q; pytest tests/tools/test_check_inbox_for_ack_cli.py -q; pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::write_history_dashboard — aggregate `ack_actor_summary` data across JSONL history entries, compute per-actor severity counts/longest-wait/latest timestamps, and render a new "## Ack Actor Severity Trends" table (critical/warning/ok/unknown ordering, Markdown-sanitized, graceful when history lacks SLA data).
- Implement (test): tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_severity_trends — run the CLI twice against a synthetic inbox with history logging/dashboard enabled (Maintainer <2> >3h overdue, Maintainer <3> still unknown) and assert the dashboard emits the new table with the correct severity counts for both actors.
- Update: docs/TESTING_GUIDE.md (add the new history-dashboard subsection + selector/log), docs/development/TEST_SUITE_INDEX.md (same selector/log), docs/fix_plan.md Attempts (record the shipped per-actor trends drop), inbox/response_dose_experiments_ground_truth.md (append the 2026-01-23T093500Z status block citing the new dashboard), and author inbox/followup_dose_experiments_ground_truth_2026-01-23T093500Z.md summarizing the actor-trend evidence for Maintainers <2>/<3>.
- Capture: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch}; python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords acknowledged --keywords confirm --keywords received --keywords thanks --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" --sla-hours 2.5 --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" --fail-when-breached --history-jsonl "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.jsonl" --history-markdown "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.md" --history-dashboard "$ARTIFACT_ROOT/inbox_history/inbox_history_dashboard.md" --status-snippet "$ARTIFACT_ROOT/inbox_status/status_snippet.md" --escalation-note "$ARTIFACT_ROOT/inbox_status/escalation_note.md" --escalation-recipient "Maintainer <2>" --output "$ARTIFACT_ROOT/inbox_sla_watch" | tee "$ARTIFACT_ROOT/logs/check_inbox.log" (expect exit 2 until Maintainer <2> replies).
- Validate: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_severity_trends -q | tee "$ARTIFACT_ROOT/logs/pytest_history_dashboard_actor_severity.log"; pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_severity_trends -q | tee "$ARTIFACT_ROOT/logs/pytest_history_dashboard_actor_severity_collect.log"; pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log"; pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log".
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z/ (JSON/Markdown summaries, dashboard/status/escalation outputs, pytest logs, updated docs/inbox entries, turn summary)

How-To Map
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch}` to stage the artifact root referenced everywhere else.
2. Extend `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::write_history_dashboard` (and helper logic if needed) so it parses `ack_actor_summary` from each JSONL entry, tallies severity counts/longest waits/latest timestamps per actor, and emits a Markdown table sorted by severity (critical→unknown) with sanitized text.
3. Add `tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_severity_trends`: build a tmp inbox with Maintainer <2> inbound ~3.5h ago + Maintainer <1> response, run the CLI twice with `--sla-hours 2.5 --ack-actor ... --ack-actor-sla ... --history-jsonl ... --history-markdown ... --history-dashboard ...`, then assert the dashboard text includes "## Ack Actor Severity Trends" and rows for Maintainer 2/3 with the expected critical/unknown counts.
4. Run `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_severity_trends -q | tee "$ARTIFACT_ROOT/logs/pytest_history_dashboard_actor_severity.log"`, `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_severity_trends -q | tee "$ARTIFACT_ROOT/logs/pytest_history_dashboard_actor_severity_collect.log"`, `pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log"`, and `pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log"`.
5. Execute the capture command to regenerate `inbox_sla_watch/`, `inbox_history/`, `inbox_status/`, and CLI logs inside `$ARTIFACT_ROOT` (Maintainer <2> should stay critical; Maintainer <3> should remain unknown).
6. Update docs/TESTING_GUIDE.md + docs/development/TEST_SUITE_INDEX.md with the new selector/log, append the 2026-01-23T093500Z status section + artifact links inside inbox/response_dose_experiments_ground_truth.md, draft inbox/followup_dose_experiments_ground_truth_2026-01-23T093500Z.md referencing the actor-trend dashboard, refresh docs/fix_plan.md Attempts, and copy this loop’s Turn Summary into "$ARTIFACT_ROOT/summary.md".

Pitfalls To Avoid
- Keep history JSONL/Markdown append-only; never rewrite prior rows.
- Sanitize Markdown (`|`, newlines) so the new table does not corrupt formatting.
- Preserve deterministic severity ordering (critical, warning, ok, unknown) to avoid flaky tests.
- Handle entries without `ack_actor_summary` by emitting a "No per-actor data" message instead of failing.
- Do not reduce the CLI’s exit code behavior (`--fail-when-breached` must still exit 2 when triggered).
- Avoid touching production modules or user-owned dataset directories; stay inside plans/active/, docs/, and inbox/.
- Ensure Maintainer <3> remains in the `unknown` bucket until a real inbound message shows up (no fake ack data).
- Capture all logs via `tee` into `$ARTIFACT_ROOT/logs/` so evidence is auditable.
- Copy the exact Turn Summary block into both the CLI reply and `$ARTIFACT_ROOT/summary.md`.

If Blocked
- Stop immediately, capture the failing command + stderr with `tee "$ARTIFACT_ROOT/logs/blocker.log"`, update docs/fix_plan.md Attempts and galph_memory.md with the blocker + error signature, and wait for supervisor guidance instead of proceeding blindly.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:745 — F1 TODO + the new 2026-01-23T09:35Z entry scoping the per-actor history dashboard work.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:1501 — `write_history_dashboard` implementation that needs the per-actor severity table.
- tests/tools/test_check_inbox_for_ack_cli.py:1 — Existing inbox CLI regression tests to mirror when adding `test_history_dashboard_actor_severity_trends`.
- docs/TESTING_GUIDE.md:79 — Inbox acknowledgement CLI (History Dashboard) section to extend with the new selector/log.
- inbox/response_dose_experiments_ground_truth.md:200 — Maintainer status log to append with the 2026-01-23T093500Z actor-trend summary.

Next Up (optional)
1. Auto-send an escalation draft to Maintainer <3> referencing the actor-trend dashboard if this loop still shows a critical breach.
2. Wire the CLI outputs into a lightweight HTML status board so Maintainer evidence is scroll-free if the wait drags on.

Doc Sync Plan (Conditional)
- After code/tests pass, re-run `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_severity_trends -q | tee "$ARTIFACT_ROOT/logs/pytest_history_dashboard_actor_severity_collect.log"`, then update docs/TESTING_GUIDE.md §Inbox acknowledgement CLI and docs/development/TEST_SUITE_INDEX.md with the selector/log references before finishing.

Mapped Tests Guardrail
- Treat `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_severity_trends -q` as mandatory; if it collects 0 tests or errors, halt and flag the issue.

Normative Math/Physics
- Not applicable — follow docs/TESTING_GUIDE.md for authoritative CLI behavior (no physics equations touched).

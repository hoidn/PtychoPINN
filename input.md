Summary: Persist the per-actor SLA severity classification inside the inbox acknowledgement history logs, add regression coverage, and capture a fresh evidence bundle proving Maintainer <2> remains in critical breach while Maintainer <3> is still within their override threshold.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_history_tracks_severity -q; pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::{append_history_jsonl,append_history_markdown} — persist the `ack_actor_summary` output in both history files (JSONL: embed the structured summary alongside the existing entry fields; Markdown: add a “Ack Actor Severity” column that lists `[CRITICAL] Maintainer 2 (4.20h > 2.00h)` style entries, with pipes/newlines sanitized) so we can prove how long each actor has been breaching.
- Implement (test): tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_history_tracks_severity — craft a synthetic inbox where Maintainer <2> is 3.5 h stale (critical) and Maintainer <3> has no inbound (unknown), run the CLI with `--sla-hours 2.5`, `--ack-actor-sla "Maintainer <2>=2.0"`, `--ack-actor-sla "Maintainer <3>=6.0"`, and the history flags, then assert the JSONL row contains the critical/unknown summary plus that the Markdown line includes `[CRITICAL] Maintainer 2` and `[UNKNOWN] Maintainer 3`.
- Update: docs/TESTING_GUIDE.md (Inbox CLI section gains the new selector/log path), docs/development/TEST_SUITE_INDEX.md (same selector/log), docs/fix_plan.md Attempts (log the per-actor severity history scope once shipped), inbox/response_dose_experiments_ground_truth.md (append a 2026-01-23T083500Z status block citing the enhanced history evidence), and author inbox/followup_dose_experiments_ground_truth_2026-01-23T083500Z.md referencing the new history data.
- Capture: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch}; python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords acknowledged --keywords confirm --keywords received --keywords thanks --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" --sla-hours 2.5 --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" --fail-when-breached --history-jsonl "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.jsonl" --history-markdown "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.md" --history-dashboard "$ARTIFACT_ROOT/inbox_history/inbox_history_dashboard.md" --status-snippet "$ARTIFACT_ROOT/inbox_status/status_snippet.md" --escalation-note "$ARTIFACT_ROOT/inbox_status/escalation_note.md" --escalation-recipient "Maintainer <2>" --output "$ARTIFACT_ROOT/inbox_sla_watch" | tee "$ARTIFACT_ROOT/logs/check_inbox.log" (expect exit 2 because Maintainer <2> is still breaching).
- Validate: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_history_tracks_severity -q | tee "$ARTIFACT_ROOT/logs/pytest_ack_actor_history.log"; pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_history_tracks_severity -q | tee "$ARTIFACT_ROOT/logs/pytest_ack_actor_history_collect.log"; pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log"; pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log".
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/ (JSON/Markdown summaries, history/status/escalation outputs, pytest logs, follow-up note, response update)

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch} to stage the drop.
2. Update plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::{append_history_jsonl,append_history_markdown}; thread the `ack_actor_summary` dict through, store it verbatim in JSONL (plus keep backwards-compatible fields), and extend the Markdown table with a new “Ack Actor Severity” column that lists severity-tagged actor summaries joined by `<br>` (sanitized) so older scans immediately show which actor breached.
3. Add tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_history_tracks_severity near the other history tests: build a synthetic inbox (Maintainer <2> inbound 3.5h ago, Maintainer <3> absent), run the CLI with ack actors, overrides, `--history-jsonl`, and `--history-markdown`, then assert the JSONL entry contains `ack_actor_summary["critical"][0]["actor_id"] == "maintainer_2"` plus Markdown row text containing `[CRITICAL] Maintainer 2` and `[UNKNOWN] Maintainer 3`.
4. Execute pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_history_tracks_severity -q | tee "$ARTIFACT_ROOT/logs/pytest_ack_actor_history.log" followed by pytest --collect-only ... -q | tee "$ARTIFACT_ROOT/logs/pytest_ack_actor_history_collect.log"; then run pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log" and pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log".
5. Run the capture command so `$ARTIFACT_ROOT` contains the refreshed JSON/Markdown/status/escalation/history outputs (expect Maintainer <2> critical, Maintainer <3> unknown) plus `logs/check_inbox.log`.
6. Update docs/TESTING_GUIDE.md & docs/development/TEST_SUITE_INDEX.md with the new selector/log path, append the latest status section + history references to inbox/response_dose_experiments_ground_truth.md, author inbox/followup_dose_experiments_ground_truth_2026-01-23T083500Z.md summarizing the per-actor history breach, refresh docs/fix_plan.md Attempts accordingly, and copy this loop’s Turn Summary into "$ARTIFACT_ROOT/summary.md" before finishing.

Pitfalls To Avoid
- Append to history JSONL/Markdown; never rewrite prior entries.
- Keep history headers intact and sanitize pipes/newlines when embedding the severity text.
- Preserve deterministic severity ordering (critical, warning, ok, unknown) to avoid flaky tests.
- Don’t suppress the CLI’s exit code 2 when `--fail-when-breached` is set; the new logging must be additive.
- Ensure Maintainer <3> stays `unknown` until an inbound message exists; no hard-coded ack.
- Avoid touching production modules or user-owned datasets; work only within plans/active/ and docs/inbox.
- Keep Markdown width manageable (use `<br>` or `;` separators) so the new column remains readable.
- Record the new selector/log paths in docs/TESTING_GUIDE.md and TEST_SUITE_INDEX to keep the registry accurate.
- Copy this turn’s summary to both the CLI reply and "$ARTIFACT_ROOT/summary.md".

If Blocked
- Capture the failing command + stderr with `tee "$ARTIFACT_ROOT/logs/blocker.log"`, stop further edits, and log the blocker + error signature inside docs/fix_plan.md Attempts plus galph_memory.md so we can decide whether to escalate or pivot focus.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:745 — Current DEBUG-SIM-LINES-DOSE-001.F1 TODO + new per-actor severity history scope.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:910 — `append_history_jsonl` / `append_history_markdown` hooks to extend with severity data.
- tests/tools/test_check_inbox_for_ack_cli.py:269 — History logging tests to mirror when adding the new regression.
- docs/TESTING_GUIDE.md:21 — Inbox acknowledgement CLI selectors/log references that must mention the new history test path.
- inbox/response_dose_experiments_ground_truth.md:198 — Maintainer status log to extend with the 2026-01-23T083500Z summary + artifact links.

Next Up (optional)
- If Maintainer <2> acknowledges after this drop, close DEBUG-SIM-LINES-DOSE-001.F1 by logging the reply and archiving the SLA monitor artifacts.

Doc Sync Plan (Conditional)
- After the new history test passes, run `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_history_tracks_severity -q | tee "$ARTIFACT_ROOT/logs/pytest_ack_actor_history_collect.log"`, then update docs/TESTING_GUIDE.md §Inbox acknowledgement CLI and docs/development/TEST_SUITE_INDEX.md with the selector + log path before finishing the loop.

Mapped Tests Guardrail
- Treat any failure of `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_history_tracks_severity -q` as a blocker—collection must succeed before shipping this loop.

Normative Math/Physics
- Not applicable; reference docs/TESTING_GUIDE.md for CLI behavior rather than the physics specs.

Summary: Add per-actor SLA override support to the inbox acknowledgement CLI and capture a fresh evidence bundle that shows Maintainer <2> remains in breach while Maintainer <3> has a looser window.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_overrides_thresholds -q; pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T060500Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::scan_inbox — add a repeatable `--ack-actor-sla` flag, parse overrides into a normalized `{actor: hours}` map, thread them into `scan_inbox()` so each actor records `sla_threshold_hours`, deadline, breach duration, severity, and notes (overrides take precedence but fall back to the global `--sla-hours`); update the JSON parameters, Markdown summary/status/escalation tables, and CLI stdout Ack Actor Coverage block to show a Threshold column whenever overrides are in play.
- Implement (test): tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_overrides_thresholds — create a synthetic inbox where Maintainer <2> is 3.5 hours stale while Maintainer <3> is 1 hour stale, run the CLI with `--sla-hours 2.5 --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=4.0"`, and assert the JSON reports actor-specific thresholds/severity plus Markdown tables/CLI stdout include the new column.
- Update: docs/TESTING_GUIDE.md (Inbox acknowledgement CLI section), docs/development/TEST_SUITE_INDEX.md (new selector + log), docs/fix_plan.md Attempts history, inbox/response_dose_experiments_ground_truth.md (add 2026-01-23T060500Z status), and drop inbox/followup_dose_experiments_ground_truth_2026-01-23T060500Z.md referencing the per-actor SLA overrides.
- Capture: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T060500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch}; python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords acknowledged --keywords confirm --keywords received --keywords thanks --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" --sla-hours 2.5 --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" --fail-when-breached --history-jsonl "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.jsonl" --history-markdown "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.md" --history-dashboard "$ARTIFACT_ROOT/inbox_history/inbox_history_dashboard.md" --status-snippet "$ARTIFACT_ROOT/inbox_status/status_snippet.md" --escalation-note "$ARTIFACT_ROOT/inbox_status/escalation_note.md" --escalation-recipient "Maintainer <2>" --output "$ARTIFACT_ROOT/inbox_sla_watch" | tee "$ARTIFACT_ROOT/logs/check_inbox.log" (exit 2 is expected because Maintainer <2> stays in breach).
- Validate: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_overrides_thresholds -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log"; pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log".
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T060500Z/ (JSON/MD outputs, logs/, updated docs/inbox files)

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T060500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch} so all logs/summaries land under the new timestamped drop.
2. Update `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py`: add the `--ack-actor-sla` parser option, normalize entries via `normalize_actor_alias()`, pass the override dict into `scan_inbox()`, emit `sla_threshold_hours` per actor, include the override map under `results["parameters"]["ack_actor_sla_hours"]`, and extend Markdown/CLI tables with a “Threshold (hrs)” column when SLA data exists. Make sure overrides can be used even when the global `--sla-hours` differs and that actors without overrides inherit the global threshold.
3. Add `test_ack_actor_sla_overrides_thresholds` to `tests/tools/test_check_inbox_for_ack_cli.py`: fabricate Maintainer <2>/<3> messages with different ages, run the CLI with both `--sla-hours` and `--ack-actor-sla` flags, and assert the JSON/Markdown/stdout incorporate the new per-actor threshold column plus severity decisions (M2 breached at 2.0 h override, M3 ok at inherited 2.5–4.0 h window).
4. Run `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_overrides_thresholds -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log"` followed by `pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log"`.
5. Execute the Capture command to regenerate inbox evidence with the new overrides; keep `--fail-when-breached` enabled and archive JSON/Markdown/status/escalation/history outputs under `$ARTIFACT_ROOT`.
6. Refresh docs (`docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`, `docs/fix_plan.md`) and maintainer comms (`inbox/response_dose_experiments_ground_truth.md`, new `inbox/followup_dose_experiments_ground_truth_2026-01-23T060500Z.md`) so they cite the per-actor thresholds + latest logs. Copy this loop’s Turn Summary into `$ARTIFACT_ROOT/summary.md` when done.

Pitfalls To Avoid
- Do not touch production modules; limit edits to `plans/active/DEBUG-SIM-LINES-DOSE-001`, `tests/tools/`, `docs/`, and `inbox/`.
- Normalise actor aliases before storing thresholds so "Maintainer <2>" and `maintainer_2` map to the same override key.
- Overrides must work even if `--sla-hours` is omitted or set to a different number; default back to the global threshold instead of leaving fields unset.
- Keep Markdown tables backward compatible (ASCII pipes, no tabs) and ensure actors without inbound still show `sla_severity="unknown"` and `threshold` values.
- The CLI capture should be the only command that exits 2; all pytest/doc updates must exit 0.
- Preserve existing history JSONL/Markdown headers; append new rows rather than rewriting prior evidence.
- When updating docs, cite the new log paths under `.../2026-01-23T060500Z/logs/` so future loops can trace the selector quickly.
- In maintainer notes, continue referencing the tarball SHA and loader pytest log so Maintainer <2> has no excuse to reject the evidence.

If Blocked
- Stop immediately, capture the failing command + stderr via `tee "$ARTIFACT_ROOT/logs/blocker.log"`, and document the blocker in `docs/fix_plan.md` Attempts and `galph_memory.md` so we can decide whether to escalate or pivot focus.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:745 — DEBUG-SIM-LINES-DOSE-001.F1 TODO + new per-actor SLA override scope.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:279 — `scan_inbox()`/Markdown logic to extend with override support.
- tests/tools/test_check_inbox_for_ack_cli.py:1086 — Existing per-actor SLA regression to model when adding the override test.
- docs/TESTING_GUIDE.md:19 — Inbox acknowledgement CLI selectors that must mention the new override test/logs.
- docs/development/TEST_SUITE_INDEX.md:13 — Test registry entry listing each selector/log for the inbox CLI.
- inbox/response_dose_experiments_ground_truth.md:200 — Maintainer status timeline that needs the 2026-01-23T060500Z update plus per-actor threshold notes.

Next Up (optional)
- Once overrides land, consider wiring a cron-friendly wrapper so we can capture SLA deltas hourly without manual execution.

Doc Sync Plan (Conditional)
- After tests pass, run `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_overrides_thresholds -q | tee "$ARTIFACT_ROOT/logs/pytest_sla_override_collect.log"`, then update docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md with the new selector + log.

Mapped Tests Guardrail
- Ensure `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_overrides_thresholds -q` collects (>0); treat collection failures as blockers before finishing the loop.

Normative Math/Physics
- Not applicable — no changes to the physics spec; cite the spec via docs/TESTING_GUIDE.md references when describing CLI usage.

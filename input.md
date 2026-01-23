Summary: Add Maintainer <3> escalation brief support to the inbox CLI plus docs/tests, then capture the new maintainer evidence bundle for F1.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement
Branch: dose_experiments
Mapped tests:
- `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_brief_targets_blocker -q`
- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_brief_targets_blocker -q`
- `pytest tests/tools/test_check_inbox_for_ack_cli.py -q`
- `pytest tests/test_generic_loader.py::test_generic_loader -q`
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/

Do Now (hard validity contract):
- Checklist IDs: F1 (Maintainer acknowledgement follow-up)
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::write_escalation_brief
- Implement: tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_brief_targets_blocker
- Test: `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_brief_targets_blocker -q`
- Artifact target: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/`

How-To Map:
1. Create artifact dirs: `mkdir -p plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/{logs,inbox_sla_watch,inbox_status,inbox_history}`.
2. Extend `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py`:
   - Add CLI args `--escalation-brief`, `--escalation-brief-recipient` (default Maintainer <3>), `--escalation-brief-target` (default Maintainer <2>). Place near other escalation args.
   - Introduce `write_escalation_brief(results, output_path, recipient, target_actor, breach_timeline_lines=None, breach_timeline_data=None)` that builds a Maintainer <3> brief with sections: header, blocking actor snapshot (hours since inbound, SLA threshold, deadline, hours past SLA, severity, ack files), breach streak summary (current streak, breach start/latest scan if timeline data present), action items, and a proposed message targeted at Maintainer <3> referencing the blocking actor. Fallback to “data unavailable” text if stats/timeline missing.
   - Update `_build_actor_breach_timeline_section` to return both Markdown lines and an `active_breaches` dict (actor_id -> breach_start/latest_scan/current_streak/hours_past_sla/severity/label) so the new helper can quote streak counts. Adjust docstring and call sites accordingly.
   - Update `main()` to capture both the timeline lines and data; pass both to `write_status_snippet`, `write_escalation_note`, and the new `write_escalation_brief` when their flags are provided. Print a console line when the new brief file is written.
3. Add `test_escalation_brief_targets_blocker` to `tests/tools/test_check_inbox_for_ack_cli.py` that:
   - Builds a temp inbox with an inbound Maintainer <2> message ~5h old (no ack) and one fresh outbound from Maintainer <1>.
   - Runs the CLI with the new flags (`--escalation-brief`, `--escalation-brief-recipient`, `--escalation-brief-target`, `--history-jsonl`, `--history-markdown`, `--status-snippet`, `--sla-hours 2.0`, `--ack-actor` for Maintainers <2>/<3>, `--fail-when-breached`).
   - Asserts the brief file exists and includes the blocking actor snapshot, “Maintainer 2” severity text, “Ack Actor Breach Timeline” section, and references Maintainer <3> in the proposed message.
4. Docs/tests/CLI execution:
   - `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_brief_targets_blocker -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/logs/pytest_escalation_brief_collect.log`.
   - `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_brief_targets_blocker -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/logs/pytest_escalation_brief.log`.
   - `pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/logs/pytest_check_inbox_suite.log`.
   - `pytest tests/test_generic_loader.py::test_generic_loader -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/logs/pytest_loader.log`.
   - Run the CLI with full flags and overrides to capture the Maintainer <3> brief + updated snippet/note/history:
```
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py \
  --inbox inbox \
  --request-pattern dose_experiments_ground_truth \
  --keywords acknowledged --keywords confirm --keywords received --keywords thanks \
  --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" \
  --sla-hours 2.5 \
  --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" \
  --fail-when-breached \
  --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/inbox_sla_watch \
  --history-jsonl plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/inbox_history/inbox_sla_watch.jsonl \
  --history-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/inbox_history/inbox_sla_watch.md \
  --history-dashboard plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/inbox_history/inbox_history_dashboard.md \
  --status-snippet plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/inbox_status/status_snippet.md \
  --escalation-note plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/inbox_status/escalation_note.md \
  --escalation-recipient "Maintainer <2>" \
  --escalation-brief plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/inbox_status/escalation_brief_maintainer3.md \
  --escalation-brief-recipient "Maintainer <3>" \
  --escalation-brief-target "Maintainer <2>"
```
   - Update `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` with a new “Maintainer <3> Escalation Brief” selector entry referencing the log path above.
   - Append a 2026-01-23T12:35Z status block to `inbox/response_dose_experiments_ground_truth.md` summarizing the new Maintainer <3> escalation brief and link to the new CLI artifacts.
   - Draft `inbox/followup_dose_experiments_ground_truth_2026-01-23T123500Z.md` targeting Maintainer <3> (CC Maintainer <2>) that cites the new brief, breach streak, and explicit ask for next steps.
   - Update `docs/fix_plan.md` Attempts History with this new increment (Maintainer <3> escalation brief) and refresh TODO/Next Actions accordingly.

Pitfalls To Avoid:
1. Do not modify production packages under `ptycho*/`—all code stays in `plans/active/DEBUG-SIM-LINES-DOSE-001/` and docs/inbox files.
2. Keep CLI output Markdown sanitized (use `sanitize_for_markdown` / `_sanitize_md` helpers; no raw pipes/newlines in table cells).
3. The brief must tolerate missing history entries; include fallback text instead of crashing when timeline data absent.
4. Preserve existing CLI defaults (`--ack-actor` still defaults to Maintainer <2>; new flags must be optional and backward compatible).
5. Do not downgrade existing selectors—ensure all mapped pytest selectors collect/passes before finishing.
6. Keep Maintainer metadata (names, SLA thresholds) synchronized with FIX_PLAN and README; avoid hard-coding anything outside CLI args.
7. Do not reroute git history or delete the maintainer inbox contents; read/append only.
8. Ensure logs/CLI outputs land under the timestamped artifacts directory listed above.
9. Avoid mixing Maintainer <3> escalation content into the Maintainer <2> note; keep separate Markdown files per recipient.

If Blocked:
- Capture the failure (error text, command, timestamp) inside `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/logs/blocked.log`, note the issue in `docs/fix_plan.md` + `galph_memory.md`, and pause F1 until the dependency or missing data is resolved.

Findings Applied (Mandatory): No relevant findings in the knowledge base.

Pointers:
- `docs/fix_plan.md:1180` — states the Maintainer <3> escalation-template requirement and current F1 status.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:1` — inbox CLI needing new flags/helper.
- `tests/tools/test_check_inbox_for_ack_cli.py:1` — regression suite where the new test must live.
- `docs/TESTING_GUIDE.md:19` & `docs/development/TEST_SUITE_INDEX.md:1` — documentation that must list the new selector/log paths.
- `inbox/response_dose_experiments_ground_truth.md:1` — maintainer response log to update with the Maintainer <3> brief reference.

Next Up (optional): If Maintainer <3> confirms but Maintainer <2> still silent, plan a closure task that archives their response + reruns the CLI to capture the ack event.

Doc Sync Plan (Conditional): Use the new collect/log outputs to add a “Maintainer <3> Escalation Brief” subsection to `docs/TESTING_GUIDE.md` + `docs/development/TEST_SUITE_INDEX.md`, referencing `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_brief_targets_blocker -q` and logs under `reports/2026-01-23T123500Z/logs/` once tests pass.

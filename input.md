Summary: Automate the maintainer-status updates so every inbox scan generates the response block + follow-up note from the latest inbox_scan_summary.json without manual editing.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement
Branch: dose_experiments
Mapped tests:
- `pytest tests/tools/test_update_maintainer_status.py::test_cli_generates_followup -q`
- `pytest tests/tools/test_check_inbox_for_ack_cli.py -q`
- `pytest tests/test_generic_loader.py::test_generic_loader -q`
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/

Do Now (hard validity contract):
- Checklist IDs: F1 (Maintainer acknowledgement follow-up)
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/update_maintainer_status.py::main — new helper CLI that appends the maintainer response block and writes the next follow-up note from inbox_scan_summary.json
- Implement: tests/tools/test_update_maintainer_status.py::test_cli_generates_followup — regression covering the new script’s response + follow-up output
- Test: `pytest tests/tools/test_update_maintainer_status.py::test_cli_generates_followup -q`
- Test: `pytest tests/tools/test_check_inbox_for_ack_cli.py -q`
- Test: `pytest tests/test_generic_loader.py::test_generic_loader -q`
- Artifact target: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/`

How-To Map:
1. Prep working dirs: `mkdir -p plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/{logs,inbox_sla_watch,inbox_history,inbox_status}`.
2. Implement `update_maintainer_status.py` in `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/`:
   - argparse flags: `--scan-json`, `--status-title`, repeated `--artifact`, `--response-path`, `--followup-path`, optional `--to`/`--cc` recipients.
   - Load the JSON, format UTC timestamps + hours fields, and build a Markdown section containing SLA summary + Ack Actor table (hrs since inbound/outbound, counts, severity, threshold, notes).
   - Append the block (preceded by two newlines) to `--response-path` and write the follow-up note (header, summary, Ack Actor Follow-Up table, bullet list of artifact links + action items) to `--followup-path`, creating parent dirs as needed.
   - Emit a short log on stdout summarizing the block title and output paths; exit non-zero on missing inputs.
3. Add helpers inside the script (e.g., `_format_hours()`, `_render_ack_actor_table()`, `_build_followup_note()`) so the test can import and assert on deterministic formatting.
4. New tests under `tests/tools/test_update_maintainer_status.py`:
   - Build a fixture inbox_scan_summary.json mirroring the current schema (global SLA block plus two actors) and write a seed response file.
   - Invoke the script via `subprocess.run([sys.executable, script, ...], check=True)` writing into tmp paths.
   - Assert the response file ends with `### Status as of <ts> (<title>)`, includes the actor table rows for Maintainer 2/3, and lists each artifact path as a bullet.
   - Assert the follow-up note file contains the new timestamp, action items, and ack actor follow-up stats.
5. Guardrail: `pytest --collect-only tests/tools/test_update_maintainer_status.py::test_cli_generates_followup -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/logs/pytest_update_status_collect.log` to prove the selector collects.
6. Run `pytest tests/tools/test_update_maintainer_status.py -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/logs/pytest_update_status.log` to execute the new suite.
7. Run `pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/logs/pytest_check_inbox_suite.log` to ensure no regression in the inbox CLI helpers.
8. Run `pytest tests/test_generic_loader.py::test_generic_loader -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/logs/pytest_loader.log` for the longstanding data-loader guard.
9. Capture a fresh inbox scan with all options enabled, logging to `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/logs/check_inbox.log`:
```
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py \
  --inbox inbox \
  --request-pattern dose_experiments_ground_truth \
  --keywords acknowledged --keywords confirm --keywords received --keywords thanks \
  --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" \
  --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" \
  --sla-hours 2.5 --fail-when-breached \
  --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/inbox_sla_watch \
  --history-jsonl plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/inbox_history/inbox_sla_watch.jsonl \
  --history-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/inbox_history/inbox_sla_watch.md \
  --history-dashboard plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/inbox_history/inbox_history_dashboard.md \
  --status-snippet plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/inbox_status/status_snippet.md \
  --escalation-note plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/inbox_status/escalation_note.md \
  --escalation-brief plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/inbox_status/escalation_brief_maintainer3.md \
  --escalation-brief-recipient "Maintainer <3>" --escalation-brief-target "Maintainer <2>"
```
10. Run the new automation CLI (tee stdout to `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/logs/update_maintainer_status.log`):
```
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/update_maintainer_status.py \
  --scan-json plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/inbox_sla_watch/inbox_scan_summary.json \
  --status-title "Maintainer Status Automation" \
  --artifact plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/inbox_sla_watch/inbox_scan_summary.md \
  --artifact plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/inbox_history/inbox_history_dashboard.md \
  --artifact plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/inbox_status/status_snippet.md \
  --response-path inbox/response_dose_experiments_ground_truth.md \
  --followup-path inbox/followup_dose_experiments_ground_truth_2026-01-23T153500Z.md \
  --to "Maintainer <2>" --cc "Maintainer <3>"
```
11. Verify the response doc gained a `### Status as of 2026-01-23T15:35Z` block with ack actor tables plus artifact list, and the follow-up note cites the same metrics; stage both files for review.
12. Docs + bookkeeping: update `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` with the new selector/log path, append the 2026-01-23T153500Z entry to `docs/fix_plan.md`, and copy the maintainer evidence (snippets, briefs, follow-up note) into the artifacts directory.

Pitfalls To Avoid:
1. Keep `update_maintainer_status.py` non-production—no imports from shipped modules.
2. Always append to the response doc; never rewrite earlier status blocks.
3. Treat missing inbound/outbound timestamps as `—` rather than crashing.
4. Do not modify the existing SLA breach detection or ack logic in `check_inbox_for_ack.py`.
5. Tests must operate entirely within `tmp_path`; never touch the real `inbox/` or artifact directories.
6. Preserve ASCII-only output and sanitize artifact paths in Markdown tables.
7. Ensure the new CLI exits non-zero on invalid input so automation can gate future runs.
8. Keep the new follow-up note in `inbox/` but avoid overwriting historical notes.
9. Log every pytest/CLI run into the new artifact root for traceability.
10. Leave the user’s dirty working tree untouched outside the targeted files.

If Blocked: Capture the failure signature and command in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/logs/blocked.log`, add a quick note to `docs/fix_plan.md` Attempts History plus Galph_memory, then pause for guidance rather than force-changing maintainer artifacts.

Findings Applied (Mandatory): No relevant findings in the knowledge base.

Pointers:
- `docs/fix_plan.md:1323` — documents the newly scoped maintainer-status automation gap and required deliverables.
- `docs/TESTING_GUIDE.md:114` — lists the Inbox Acknowledgement CLI sections that need an additional selector entry for the automation helper’s regression test.

Next Up (optional): Once automation lands, continue running the inbox CLI on a cadence until Maintainer <2> or <3> responds so we can close F1 with an acknowledgement record.

Doc Sync Plan (Conditional): After tests pass, keep the `pytest --collect-only ...test_cli_generates_followup` log under `logs/pytest_update_status_collect.log`, then update `docs/TESTING_GUIDE.md` §Inbox CLI and `docs/development/TEST_SUITE_INDEX.md` with the new selector and artifact paths before concluding the loop.

Mapped Tests Guardrail: Step 5’s collect-only command ensures the new selector gathers exactly one node before running the full suite.

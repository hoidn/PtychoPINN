Summary: Run the cadence CLI once to capture a fresh maintainer-status snapshot (timestamp 2026-01-23T173500Z), append the new status block + follow-up note, and log the evidence path for F1.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement
Branch: dose_experiments
Mapped tests:
- `pytest --collect-only tests/tools/test_run_inbox_cadence.py::test_cadence_sequence_creates_artifacts -q`
- `pytest tests/tools/test_run_inbox_cadence.py::test_cadence_sequence_creates_artifacts -q`
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T173500Z/

Do Now (hard validity contract):
- Checklist IDs: F1 (Maintainer acknowledgement cadence loop)
- Implement: inbox/response_dose_experiments_ground_truth.md::Status as of 2026-01-23T173500Z — run `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_inbox_cadence.py` with `--timestamp 2026-01-23T173500Z` so it appends the status section and emits `inbox/followup_dose_experiments_ground_truth_2026-01-23T173500Z.md`, then review/adjust the appended block if needed.
- Test: `pytest tests/tools/test_run_inbox_cadence.py::test_cadence_sequence_creates_artifacts -q`
- Artifact target: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T173500Z/`

How-To Map:
1. Prep artifacts: `mkdir -p plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T173500Z/logs` so logs/collect outputs land in the right folder without disturbing prior notes.
2. Guard the mapped selector: `pytest --collect-only tests/tools/test_run_inbox_cadence.py::test_cadence_sequence_creates_artifacts -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T173500Z/logs/pytest_run_inbox_cadence_collect.log` — confirms the test is still discoverable.
3. Run the regression: `pytest tests/tools/test_run_inbox_cadence.py::test_cadence_sequence_creates_artifacts -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T173500Z/logs/pytest_run_inbox_cadence.log`.
4. Execute the real cadence loop (teeing stdout/stderr into `logs/run_inbox_cadence.log`):
```
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_inbox_cadence.py \
  --inbox inbox \
  --request-pattern dose_experiments_ground_truth \
  --keywords acknowledged --keywords confirm --keywords received --keywords thanks \
  --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" \
  --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" \
  --sla-hours 2.5 --fail-when-breached \
  --output-root plans/active/DEBUG-SIM-LINES-DOSE-001/reports \
  --timestamp 2026-01-23T173500Z \
  --response-path inbox/response_dose_experiments_ground_truth.md \
  --followup-dir inbox \
  --followup-prefix followup_dose_experiments_ground_truth \
  --status-title "Maintainer Status Automation" \
  --to "Maintainer <2>" --cc "Maintainer <3>" \
  --escalation-brief-recipient "Maintainer <3>" \
  --escalation-brief-target "Maintainer <2>" \
  --skip-followup-on-ack \
  | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T173500Z/logs/run_inbox_cadence.log
```
5. Inspect cadence metadata: `python - <<'PY'` (or `jq`) to print `ack_detected`, `followup_written`, and key wait metrics from `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T173500Z/cadence_metadata.json`; record the outcome for docs/fix_plan.
6. Verify deliverables: ensure `cadence_summary.md`, `inbox_status/{status_snippet,escalation_note,escalation_brief_*.md}`, and the new `inbox/followup_dose_experiments_ground_truth_2026-01-23T173500Z.md` exist; open `inbox/response_dose_experiments_ground_truth.md` to confirm the CLI appended a "### Status as of 2026-01-23T173500Z" block with updated metrics and artifact pointers (edit if formatting needs tightening).
7. Update `docs/fix_plan.md` Attempts History with a new 2026-01-23T173500Z entry summarizing the ack status, SLA deltas, test selector/log references, and the follow-up note path; note in the entry whether the cadence loop exited with 0 or 3.
8. If `ack_detected` flipped to true, mark DEBUG-SIM-LINES-DOSE-001.F1 complete inside `docs/fix_plan.md` (TODO checkbox + status text) and capture the acknowledging inbox file path; otherwise, mention the latest breach duration + outbound counts in the Attempts entry so Maintainer <3> has current evidence.
9. Stage the refreshed artifacts (new reports dir, response doc update, follow-up note, cadence metadata/summary); leave older timestep directories untouched for auditability.

Pitfalls To Avoid:
1. Do not run the cadence CLI without `--timestamp 2026-01-23T173500Z`; otherwise artifacts land in an unexpected folder and the How-To map becomes inaccurate.
2. Keep the ack-actor/SLA overrides exactly as specified—changing actors or thresholds would invalidate the trend comparisons in history files.
3. Never edit `cadence_metadata.json` by hand; regenerate it by rerunning the CLI if it looks wrong.
4. Append to `inbox/response_dose_experiments_ground_truth.md`; never overwrite or reorder prior status blocks.
5. If `--skip-followup-on-ack` causes exit code 3, do not manually write a follow-up note—document the skip in fix_plan and leave evidence in metadata.
6. Avoid deleting prior `inbox/followup_*.md` files; each timestamp is part of the audit trail.
7. Keep all paths ASCII; do not paste emoji or smart quotes into maintainer docs.
8. Capture every command’s stdout/stderr via `tee` into the artifacts directory so Maintainer <3> can audit the run.
9. If the CLI reports `ack_detected=true`, stop before sending additional reminders—just update the docs/fix_plan state.
10. Leave `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T163500Z` untouched; the new run should only populate the 173500Z directory.

If Blocked: Capture the failing command + stderr into `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T173500Z/logs/blocked.log`, summarize the issue + hypothesis in `docs/fix_plan.md` (F1 Attempts History) and ping me before retrying so we can adjust the plan or escalate to the maintainer.

Findings Applied (Mandatory): No relevant findings in the knowledge base.

Pointers:
- docs/fix_plan.md:1392 — Captures the cadence-orchestrator deliverables and explicitly calls for the real evidence run + updated maintainer docs.
- docs/TESTING_GUIDE.md:352 — Documents the Inbox Cadence CLI usage and required pytest selectors/log handling.
- docs/development/TEST_SUITE_INDEX.md:80 — Lists `tests/tools/test_run_inbox_cadence.py` selectors and the log expectations we must maintain.

Next Up (optional): If this run still shows no acknowledgement, schedule the next cadence execution (e.g., 2026-01-23T183500Z) and consider prepping a Maintainer <3> escalation follow-up referencing the new breach duration.

Mapped Tests Guardrail: Step 2 runs `pytest --collect-only tests/tools/test_run_inbox_cadence.py::test_cadence_sequence_creates_artifacts -q`, so the mapped selector is proven collectible before executing the test.

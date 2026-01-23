Summary: Ship a single cadence CLI so every maintainer follow-up run is one command that wires check_inbox, status automation, and evidence capture.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement
Branch: dose_experiments
Mapped tests:
- `pytest tests/tools/test_run_inbox_cadence.py::test_cadence_sequence_creates_artifacts -q`
- `pytest tests/tools/test_run_inbox_cadence.py::test_cadence_skips_followup_on_ack -q`
- `pytest tests/tools/test_check_inbox_for_ack_cli.py -q`
- `pytest tests/tools/test_update_maintainer_status.py -q`
- `pytest tests/test_generic_loader.py::test_generic_loader -q`
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T163500Z/

Do Now (hard validity contract):
- Checklist IDs: F1 (Maintainer acknowledgement follow-up cadence)
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_inbox_cadence.py::main — new non-production CLI that accepts inbox/request pattern/keyword/actor inputs, creates a timestamped reports directory, runs `check_inbox_for_ack.py` with all history/status/escalation outputs wired into subfolders, logs stdout/stderr, reads `inbox_scan_summary.json`, records metadata (ack flag, timestamps, artifact paths), and, unless ack was detected and `--skip-followup-on-ack` was passed, invokes `update_maintainer_status.py` with the proper artifact list to append the response block + follow-up note. Emit `cadence_metadata.json`, human-readable stdout summary, and exit codes (0 success, 3 ack+skipped follow-up, non-zero on failure).
- Implement: tests/tools/test_run_inbox_cadence.py::test_cadence_sequence_creates_artifacts — regression that builds a temp inbox + response doc, runs the new CLI with deterministic `--timestamp 2026-01-23T163500Z`, asserts the expected directory layout/logs/metadata, verifies the response doc gained a status block, and ensures the follow-up note landed at `<tmp>/followup_dose_experiments_ground_truth_2026-01-23T163500Z.md`.
- Implement: tests/tools/test_run_inbox_cadence.py::test_cadence_skips_followup_on_ack — fixture with an ACK keyword that triggers ack detection, run CLI with `--skip-followup-on-ack`, assert exit code 3, metadata shows `ack_detected: true` and `followup_written: false`, and confirm no follow-up file was created while cadence metadata + logs still appear.
- Test: `pytest tests/tools/test_run_inbox_cadence.py::test_cadence_sequence_creates_artifacts -q`
- Test: `pytest tests/tools/test_run_inbox_cadence.py::test_cadence_skips_followup_on_ack -q`
- Test: `pytest tests/tools/test_check_inbox_for_ack_cli.py -q`
- Test: `pytest tests/tools/test_update_maintainer_status.py -q`
- Test: `pytest tests/test_generic_loader.py::test_generic_loader -q`
- Artifact target: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T163500Z/`

How-To Map:
1. Prep workspace: `mkdir -p plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T163500Z/{logs,inbox_sla_watch,inbox_history,inbox_status}` so logs land under the target path.
2. Implement `run_inbox_cadence.py` under `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/`:
   - Args: `--inbox`, `--request-pattern`, repeatable `--keywords`, `--ack-actor`, `--ack-actor-sla actor=hrs`, `--sla-hours`, `--fail-when-breached`, `--output-root`, optional `--timestamp` override (default ISO8601 from UTC now), `--response-path`, `--followup-dir`, `--followup-prefix`, `--status-title`, `--to`, `--cc`, `--escalation-brief-recipient`, `--escalation-brief-target`, and `--skip-followup-on-ack`.
   - Derived paths: `<output-root>/<timestamp>/logs`, `/inbox_sla_watch`, `/inbox_history`, `/inbox_status`. Wire `check_inbox_for_ack.py` to `--output` (sla_watch), `--history-jsonl` (history/inbox_sla_watch.jsonl), `--history-markdown`, `--history-dashboard`, `--status-snippet`, `--escalation-note`, `--escalation-brief`, `--history JSONL` prerequisites, etc. Use `subprocess.run(..., check=True, capture_output=False)` redirected to log file via `tee` logic (open file handle and pass to `stdout`/`stderr`).
   - After scan, load `inbox_scan_summary.json`, compute follow-up path = `<followup-dir>/<followup-prefix>_<timestamp>.md`, build `cadence_metadata.json` containing the timestamp, ack flag, ack_actor_summary (if present), CLI parameters, log paths, follow-up path, and bools `followup_written`/`status_appended`.
   - If ack detected and `--skip-followup-on-ack` is set, skip `update_maintainer_status.py`, emit stdout message explaining skip, and exit 3.
   - Otherwise run `update_maintainer_status.py` with `--scan-json`, `--status-title`, `--artifact` (include `inbox_sla_watch/inbox_scan_summary.md`, `inbox_history/inbox_history_dashboard.md`, `inbox_status/status_snippet.md`, `inbox_status/escalation_note.md`, `inbox_status/escalation_brief_*.md` when present), `--response-path`, `--followup-path`, `--to`, `--cc`. Tee its stdout/stderr into `logs/update_maintainer_status.log` and set `followup_written=True` when the command succeeds.
   - Always write `cadence_metadata.json` + `cadence_summary.md` describing ack status, hours since inbound/outbound, follow-up file path, and exit code.
3. Tests (`tests/tools/test_run_inbox_cadence.py`):
   - Add helpers to create inbox files with adjustable mtimes (reuse logic from `tests/tools/test_check_inbox_for_ack_cli.py` via local helper to avoid cross-imports).
   - `test_cadence_sequence_creates_artifacts`: uses tmp inbox w/ inbound/outbound messages lacking ack keywords so ack_detected is False. Seed response doc with header. Run CLI with deterministic timestamp (`--timestamp 2026-01-23T163500Z`), keywords (acknowledged/confirm/received/thanks), ack actors `<2` and `<3>`, SLA hours 2.5 + overrides, `--fail-when-breached`, follow-up dir = tmp_path, prefix = `followup_dose_experiments_ground_truth`. Assert return code 0, directory tree exists with log files, JSON summary, status snippet, escalation note/brief, follow-up note path exists, response doc now ends with `### Status as of 2026-01-23T163500Z`, metadata JSON records ack false + followup true.
   - `test_cadence_skips_followup_on_ack`: create inbox file containing ack keyword from Maintainer <2> and pass `--skip-followup-on-ack`; expect exit code 3, metadata shows ack true, followup flag false, follow-up file absent, but logs + scan JSON exist.
4. Docs: add a new subsection to `docs/TESTING_GUIDE.md` under Inbox CLI describing the cadence driver test selectors + usage, and append to `docs/development/TEST_SUITE_INDEX.md` with summary + log paths (collect log goes under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T163500Z/logs/`).
5. Guardrail: `pytest --collect-only tests/tools/test_run_inbox_cadence.py::test_cadence_sequence_creates_artifacts -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T163500Z/logs/pytest_run_inbox_cadence_collect.log`.
6. Execute new suite: `pytest tests/tools/test_run_inbox_cadence.py -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T163500Z/logs/pytest_run_inbox_cadence.log`.
7. Inbox CLI regression: `pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T163500Z/logs/pytest_check_inbox_suite.log`.
8. Status automation guard: `pytest tests/tools/test_update_maintainer_status.py -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T163500Z/logs/pytest_update_status.log`.
9. Loader guard: `pytest tests/test_generic_loader.py::test_generic_loader -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T163500Z/logs/pytest_loader.log`.
10. Produce the real evidence bundle via the new CLI (tee stdout to logs/run_inbox_cadence.log):
```
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_inbox_cadence.py \
  --inbox inbox \
  --request-pattern dose_experiments_ground_truth \
  --keywords acknowledged --keywords confirm --keywords received --keywords thanks \
  --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" \
  --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" \
  --sla-hours 2.5 --fail-when-breached \
  --output-root plans/active/DEBUG-SIM-LINES-DOSE-001/reports \
  --timestamp 2026-01-23T163500Z \
  --response-path inbox/response_dose_experiments_ground_truth.md \
  --followup-dir inbox \
  --followup-prefix followup_dose_experiments_ground_truth \
  --status-title "Maintainer Status Automation" \
  --to "Maintainer <2>" --cc "Maintainer <3>" \
  --escalation-brief-recipient "Maintainer <3>" --escalation-brief-target "Maintainer <2>"
```
   - Capture resulting follow-up note + status block updates in Git, store CLI stdout/stderr inside `logs/run_inbox_cadence.log`, and copy `cadence_metadata.json` plus the JSON/MD outputs into the artifacts directory.
11. Update documentation + bookkeeping: append the new cadence run entry to `docs/fix_plan.md` Attempts History, cite ack status (if ack true, mark focus ready to close). Stage `inbox/response_*.md`, `inbox/followup_*.md`, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, new script/tests, and logs for review.

Pitfalls To Avoid:
1. Keep cadence script/tests under `plans/active/...` (non-production) and avoid touching `scripts/` or shipped modules.
2. Never overwrite historical status blocks—append new sections and verify headings include the timestamp.
3. Tests must rely on `tmp_path` inboxes; do not reuse the real `inbox/` contents or artifact directories.
4. Ensure all JSON/MD outputs remain ASCII and deterministic (explicit timestamps via `--timestamp`).
5. When ack is detected, honor `--skip-followup-on-ack` to avoid spamming Maintainer <2>; still record metadata + history artifacts.
6. Propagate failures from subprocesses (non-zero exit -> raise) so CI catches real issues; no silent `subprocess.run` ignoring codes.
7. Keep CLI options consistent with `docs/TESTING_GUIDE.md` (keywords, ack actors, SLA overrides) to avoid divergence between automation and documentation.
8. Do not delete or rewrite existing `inbox_history/*.jsonl` files outside the new timestamp scope; always create new directories under the provided output root.
9. Respect the dirty working tree—limit edits to files listed above and never reformat unrelated maintainer artifacts.
10. Ensure `cadence_metadata.json` reflects actual ack status and follow-up decision so Maintainer <3> can audit automation at a glance.

If Blocked: Log the command + stderr into `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T163500Z/logs/blocked.log`, summarize the failure + hypothesis in `docs/fix_plan.md` Attempts History, and ping me before retrying to avoid clobbering maintainer evidence.

Findings Applied (Mandatory): No relevant findings in the knowledge base.

Pointers:
- docs/fix_plan.md:1328 — captures the maintainer-status automation scope and the new cadence orchestrator requirements.
- docs/TESTING_GUIDE.md:19 — enumerates the inbox acknowledgement CLI selectors/templates that the cadence driver must keep in sync.
- docs/development/TEST_SUITE_INDEX.md:1 — reference layout for documenting new selectors/log paths alongside existing inbox CLI coverage.

Next Up (optional): Once cadence automation lands, rerun the CLI daily until Maintainer <2>/<3> acknowledges so we can close DEBUG-SIM-LINES-DOSE-001.

Doc Sync Plan (Conditional): After the new tests pass, append the cadence driver section + selector under `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md`, referencing the new log paths in `reports/2026-01-23T163500Z/`. Keep the collect-only log from Step 5 as evidence.

Mapped Tests Guardrail: Step 5 ensures `pytest --collect-only tests/tools/test_run_inbox_cadence.py::test_cadence_sequence_creates_artifacts -q` collects the new selector before the full suite.

Summary: Embed the breach timeline (actor streaks + hours past SLA) directly into the status snippet and escalation note so Maintainers <2>/<3> can review streak data without opening the history dashboard.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary -q; pytest tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary -q; pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_note_emits_call_to_action -q; pytest tests/tools/test_check_inbox_for_ack_cli.py -q; pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T113500Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::main + ::write_status_snippet + ::write_escalation_note — after appending history JSONL, load the entries, reuse `_build_actor_breach_timeline_section()` to generate the "Ack Actor Breach Timeline" block, and append that Markdown section to both the status snippet and escalation note whenever `--history-jsonl` is provided (skip entirely when history logging is disabled so current users are unaffected).
- Implement (tests): tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary and ::test_escalation_note_emits_call_to_action — extend each test to verify (a) no breach timeline section is emitted when `--history-jsonl` is omitted and (b) running the CLI with history logging enabled writes the new "Ack Actor Breach Timeline" table containing a Maintainer 2 row, ensuring the snippet/note pull streak data from the appended JSONL.
- Update: docs/TESTING_GUIDE.md (§Status Snippet / Escalation Note) and docs/development/TEST_SUITE_INDEX.md so both call out the embedded breach timeline behavior (same selectors/logs); append a 2026-01-23T11:35Z status block to inbox/response_dose_experiments_ground_truth.md, add inbox/followup_dose_experiments_ground_truth_2026-01-23T113500Z.md summarizing the new snippet/note outputs, and note the attempt in docs/fix_plan.md Attempts History.
- Capture: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T113500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch} && python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords acknowledged --keywords confirm --keywords received --keywords thanks --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" --sla-hours 2.5 --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" --fail-when-breached --history-jsonl "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.jsonl" --history-markdown "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.md" --history-dashboard "$ARTIFACT_ROOT/inbox_history/inbox_history_dashboard.md" --status-snippet "$ARTIFACT_ROOT/inbox_status/status_snippet.md" --escalation-note "$ARTIFACT_ROOT/inbox_status/escalation_note.md" --escalation-recipient "Maintainer <2>" --output "$ARTIFACT_ROOT/inbox_sla_watch" | tee "$ARTIFACT_ROOT/logs/check_inbox.log" (exit code 2 expected until Maintainer <2> replies).
- Validate: pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary -q | tee "$ARTIFACT_ROOT/logs/pytest_status_snippet_collect.log"; pytest tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary -q | tee "$ARTIFACT_ROOT/logs/pytest_status_snippet.log"; pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_note_emits_call_to_action -q | tee "$ARTIFACT_ROOT/logs/pytest_escalation_note.log"; pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log"; pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log".
- Artifacts: Archive the refreshed JSON/Markdown summaries, history dashboard/status/escalation outputs, pytest logs, updated docs/inbox diffs, and copy this turn summary into "$ARTIFACT_ROOT/summary.md".

How-To Map
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T113500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch}` to set the artifact context the commands reference.
2. Update `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py`:
   - After `append_history_jsonl()` returns, read the JSONL into memory (skip when the flag wasn’t passed) and reuse `_build_actor_breach_timeline_section()` to prepare Markdown lines.
   - Extend `write_status_snippet` and `write_escalation_note` signatures to accept the optional history entries or pre-rendered lines, append the "## Ack Actor Breach Timeline" table when data exists, and skip the section entirely when history logging is off so current behavior stays intact.
3. Modify `tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary` and `::test_escalation_note_emits_call_to_action` to run the CLI twice (first without history to assert the section is absent, then with `--history-jsonl` to assert the Maintainer 2 breach row is present) while keeping the rest of the assertions unchanged.
4. Run the mapped pytest commands (collect-only, both targeted selectors, full module, loader guard) teeing logs into `$ARTIFACT_ROOT/logs/` exactly as listed—halt immediately if any selector fails or collects zero tests.
5. Execute the inbox CLI command so `$ARTIFACT_ROOT` contains refreshed `inbox_sla_watch/`, `inbox_history/`, `inbox_status/`, and `logs/check_inbox.log` artifacts that now show the breach timeline embedded in the snippet and escalation note (exit 2 expected because the SLA is still breached).
6. Update `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` language for the Status Snippet/Escalation Note sections, append the new status block to `inbox/response_dose_experiments_ground_truth.md`, draft `inbox/followup_dose_experiments_ground_truth_2026-01-23T113500Z.md`, and capture any doc diffs plus this turn summary inside `$ARTIFACT_ROOT/summary.md`.

Pitfalls To Avoid
- Do not introduce the breach timeline section when `--history-jsonl` is absent; the snippet/note must remain compact for users who only run a one-off scan.
- Keep `_build_actor_breach_timeline_section()` as the single source of Markdown formatting so the dashboard/snippet/escalation outputs stay identical.
- Ensure history JSONL is read *after* the current run is appended so the newest entry shows up inside the snippet/note timeline.
- Sanitize actor labels/timestamps before inserting into tables to avoid breaking Markdown layout.
- Preserve existing CLI exit codes and ack detection behavior; this change is additive only.
- Avoid duplicating the breach timeline block or adding multiple "##" headers on repeated runs—writers must remain idempotent.
- Do not clobber existing history JSONL/Markdown contents; always append before reading.
- When updating docs/inbox responses, keep previous status blocks intact and clearly label the 2026-01-23T11:35Z entry.

If Blocked
- If the snippet/note tests fail to collect or the CLI exits unexpectedly, capture stderr/stdout to `$ARTIFACT_ROOT/logs/blocker.log`, log the issue in docs/fix_plan.md Attempts History and galph_memory.md, and pause for supervisor guidance rather than guessing.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:1033 — `write_status_snippet` body that now needs to append the breach timeline lines.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:1258 — `write_escalation_note` implementation where the new section must be inserted.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:1505 — `_build_actor_breach_timeline_section` helper already used by the history dashboard.
- tests/tools/test_check_inbox_for_ack_cli.py:374 — Status snippet regression test to extend with the history-enabled assertions.
- tests/tools/test_check_inbox_for_ack_cli.py:520 — Escalation note regression test to extend for the breach timeline block.
- docs/TESTING_GUIDE.md:40 — Status Snippet / Escalation Note documentation to refresh with the new behavior.
- inbox/response_dose_experiments_ground_truth.md:200 — Maintainer status log that needs the 2026-01-23T11:35Z update.

Next Up (optional)
1. If Maintainer <2> still ignores the bundle after this drop, prep an escalation template for Maintainer <3> that pulls breach streak data directly from the snippet.
2. Consider emitting a machine-readable breach timeline JSON alongside the Markdown so automation can alert maintainers without parsing text.

Mapped Tests Guardrail
- `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary -q` must collect exactly one test; stop immediately if it returns 0 or errors.

Hard Gate
- Do not mark DEBUG-SIM-LINES-DOSE-001.F1 as complete until the new snippet/escalation note show the breach timeline in `$ARTIFACT_ROOT/inbox_status/*.md` and `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary -q` plus the rest of the mapped selectors pass with collection logs archived.

Normative Math/Physics
- Not applicable; rely on docs/TESTING_GUIDE.md for the authoritative CLI/test behavior (no physics specs involved).

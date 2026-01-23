Summary: Generalize the inbox acknowledgement CLI so it can treat Maintainer <3> replies as valid acks via a new --ack-actor flag, fix the --keywords gate, and capture a fresh SLA evidence drop.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: pytest tests/tools/test_check_inbox_for_ack_cli.py -q; pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T024800Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::scan_inbox (plus parse_args/is_acknowledgement) — add a repeatable --ack-actor flag (default Maintainer <2>), extend actor detection to recognize Maintainer <3>, require ack actors to match before marking acknowledgements, and honor the user-provided --keywords list instead of the hard-coded substring list while keeping JSON/Markdown summaries in sync.
- Implement: tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_supports_multiple_inbound_maintainers + test_custom_keywords_enable_ack_detection — cover the new flag (ack from Maintainer <3> should only trigger when --ack-actor includes them) and prove custom keywords actually gate detection; keep fixtures isolated under tmp_path.
- Update: docs/TESTING_GUIDE.md::Inbox acknowledgement CLI sections + docs/development/TEST_SUITE_INDEX.md::Inbox acknowledgement entry — add the new selectors/log paths once pytest logs exist, and refresh docs/fix_plan.md Attempts plus inbox/followup_dose_experiments_ground_truth_2026-01-23T024800Z.md with the latest SLA metrics.
- Capture: python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords acknowledged --keywords confirm --keywords received --keywords thanks --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" --sla-hours 2.0 --fail-when-breached --history-jsonl $ARTIFACT_ROOT/inbox_history/inbox_sla_watch.jsonl --history-markdown $ARTIFACT_ROOT/inbox_history/inbox_sla_watch.md --history-dashboard $ARTIFACT_ROOT/inbox_history/inbox_history_dashboard.md --status-snippet $ARTIFACT_ROOT/inbox_status/status_snippet.md --escalation-note $ARTIFACT_ROOT/inbox_status/escalation_note.md --output $ARTIFACT_ROOT/inbox_sla_watch | tee $ARTIFACT_ROOT/logs/check_inbox.log (expect exit 2 while SLA is breached).
- Validate: pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee $ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log && pytest tests/test_generic_loader.py::test_generic_loader -q | tee $ARTIFACT_ROOT/logs/pytest_loader.log.

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T024800Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch}.
2. Update check_inbox_for_ack.py: define normalize_actor_alias() covering Maintainer <1>/<2>/<3>, expand detect_actor_and_direction() with Maintainer <3> regexes, plumb ack_actors through scan_inbox() + results["parameters"], have is_acknowledgement() require actor membership + keyword hits, and ensure summaries mention the configured actors.
3. Extend tests/tools/test_check_inbox_for_ack_cli.py with two new tests (one for multi-actor ack via Maintainer <3>, one proving custom keywords are honored) plus any helper utilities; keep CLI invocations under tmp_path and assert JSON ack_detected flips only when expected.
4. Run pytest for the CLI suite, then rerun the loader guard as listed under Validate; capture stdout/stderr into $ARTIFACT_ROOT/logs/.
5. Execute the CLI command under Capture (exit 2 is expected) to refresh JSON/MD/history/status/dashboard artifacts inside $ARTIFACT_ROOT and double-check ack_actors + keywords are recorded.
6. Draft `inbox/followup_dose_experiments_ground_truth_2026-01-23T024800Z.md` summarizing the new wait metrics + ack-actor coverage, and update docs/fix_plan.md Attempts along with inbox/response_dose_experiments_ground_truth.md so Maintainer <2> sees the refreshed SLA breach evidence.
7. Update docs/TESTING_GUIDE.md + docs/development/TEST_SUITE_INDEX.md with the new selectors/logs once pytest logs exist, and ensure plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T024800Z/summary.md mirrors the final Turn Summary.

Pitfalls To Avoid
- Do not touch shipping modules; keep all code/doc changes inside plans/active/DEBUG-SIM-LINES-DOSE-001, tests/tools/, and docs/.
- Maintain backwards compatibility: default behavior must still treat only Maintainer <2> as acks unless --ack-actor is provided.
- Normalize actor strings ("Maintainer <3>", "maintainer_3") so regex casing/spaces do not block detection.
- Honor --keywords exactly; do not reintroduce hidden hard-coded substrings that override user intent.
- Keep CLI output ASCII and small so inbox artifacts remain readable; avoid dumping large inbox file contents into new docs unless truncated.
- Tests must fabricate inbox entries under tmp_path only; never read actual `inbox/` so we avoid leaking maintainer data.
- The CLI run under Capture will exit 2 because --fail-when-breached is set; treat that as success and still archive the outputs.
- Do not delete prior follow-up or response files; create the 2026-01-23T024800Z follow-up alongside the existing ones.

If Blocked
- Capture the failing command + stderr with `tee $ARTIFACT_ROOT/logs/blocker.log`, note the error plus attempted command in docs/fix_plan.md Attempts and ping Galph before changing scope.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:570 — DEBUG-SIM-LINES-DOSE-001.F1 context + newly added ack-actor next steps.
- docs/TESTING_GUIDE.md:19 — Inbox acknowledgement CLI sections to expand with the new selectors/log references.
- docs/development/TEST_SUITE_INDEX.md:13 — Test catalog entry that must record the added selectors/logs.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:1 — CLI implementation to update with --ack-actor + keyword logic.
- tests/tools/test_check_inbox_for_ack_cli.py:1 — Existing SLA/history/snippet/escalation tests to mirror for the new cases.
- inbox/response_dose_experiments_ground_truth.md:200 — Maintainer status log that needs the 2026-01-23T024800Z update once the new scan finishes.

Next Up (optional)
1. If Maintainer <2> remains silent after the multi-actor monitoring, prep a Maintainer <3> escalation package referencing the new CLI evidence.
2. Automate periodic check_inbox_for_ack.py runs (cron/task runner) so SLA breaches are logged without manual intervention.

Doc Sync Plan (Conditional)
- After tests pass, run `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_supports_multiple_inbound_maintainers -q | tee "$ARTIFACT_ROOT/logs/pytest_ack_actor_collect.log"` and `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_custom_keywords_enable_ack_detection -q | tee "$ARTIFACT_ROOT/logs/pytest_keywords_collect.log"`, archive the logs under $ARTIFACT_ROOT, and then update docs/TESTING_GUIDE.md + docs/development/TEST_SUITE_INDEX.md with the selectors/log references.

Mapped Tests Guardrail
- Ensure both new selectors (`tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_supports_multiple_inbound_maintainers` and `::test_custom_keywords_enable_ack_detection`) collect (>0) before finishing; treat collection failures as blockers.

Normative Math/Physics
- Not applicable — this loop focuses on maintainer-monitoring tooling.

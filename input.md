Summary: Extend the inbox scan CLI with a maintainer-direction timeline and waiting-clock metadata, then re-run the scan + loader pytest so we can prove how long we've been waiting on Maintainer <2>.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T014900Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::scan_inbox — add sender/recipient detection helpers so each match records `actor`, `direction`, and whether it is inbound (Maintainer <2>) or outbound (Maintainer <1>), track `last_inbound_utc`/`last_outbound_utc`, compute hours-since metrics, emit a `timeline` array in the JSON, and render a matching "Timeline" + "Waiting Clock" section in the Markdown summary; keep ack detection rules unchanged (ack = Maintainer <2> + ack keyword) and run the CLI into $ARTIFACT_ROOT/inbox_check_timeline with stdout tee so we capture the new structure.
- Update: docs/fix_plan.md::DEBUG-SIM-LINES-DOSE-001.F1 Attempts History — log the refreshed scan timestamp, cite $ARTIFACT_ROOT/inbox_check_timeline outputs, record `ack_detected` plus the computed wait duration, and only mark the TODO complete if Maintainer <2> actually acknowledged.
- Update: inbox/response_dose_experiments_ground_truth.md::Status — append a "Status as of 2026-01-23T014900Z" section summarizing the new timeline stats (last inbound/outbound timestamps, hours since last Maintainer <2> message) and link directly to $ARTIFACT_ROOT/inbox_check_timeline/inbox_scan_summary.md.
- Pytest: pytest tests/test_generic_loader.py::test_generic_loader -q
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T014900Z/

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T014900Z && mkdir -p "$ARTIFACT_ROOT/inbox_check_timeline"
2. Edit plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py: add helpers to detect Maintainer <1>/<2> senders, extend scan_inbox() to annotate each match with actor/direction + append entries to a `timeline` list sorted by modified timestamp, roll up `last_inbound_utc`, `last_outbound_utc`, and hours-since metrics in the summary dict, and update write_markdown_summary() to emit "Waiting Clock" bullets and a `Timeline` table showing UTC timestamp, actor, direction, ack flag, and keywords.
3. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords acknowledged --keywords ack --keywords confirm --keywords received --keywords thanks --output "$ARTIFACT_ROOT/inbox_check_timeline" | tee "$ARTIFACT_ROOT/inbox_check_timeline/check_inbox.log"
4. Review $ARTIFACT_ROOT/inbox_check_timeline/inbox_scan_summary.json: if a Maintainer <2> ack is listed, copy the full inbox file into $ARTIFACT_ROOT/inbox_check_timeline/ack_<filename>; otherwise note `ack_detected: false` plus the reported hours-since metric for the docs updates.
5. Update docs/fix_plan.md Attempts History + the F1 TODO bullet with the new scan timestamp, wait duration, and acknowledgement status (cite the summary path either way) so the plan records the monitoring evidence.
6. Append the new "Status as of 2026-01-23T014900Z" section to inbox/response_dose_experiments_ground_truth.md showing last inbound/outbound times and linking to the Markdown summary.
7. pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/pytest_loader.log"
8. Capture a short note in $ARTIFACT_ROOT/inbox_check_timeline/README.md describing the commands you ran plus the waiting-clock values for quick reference.

Pitfalls To Avoid
- Do not relax the Maintainer <2> acknowledgement rule; ack still requires a Maintainer <2> message with a core ack keyword.
- Keep JSON/Markdown output backward-compatible (preserve existing keys) while appending the new waiting/timeline fields so earlier analyses remain valid.
- Never delete or rename inbox files—only read them and optionally copy snapshots into $ARTIFACT_ROOT.
- Maintain deterministic ordering in the new timeline table (sort by modified_utc ascending) to prevent noise across scans.
- Avoid copying the large tarball or datasets; only reference them.
- Record every command output under $ARTIFACT_ROOT to keep evidence auditable.
- Environment changes are frozen; do not install packages or modify system settings.
- Keep tarball SHA references unchanged; the waiting-clock update should not trigger new bundle work.
- Ensure pytest logs land in $ARTIFACT_ROOT/pytest_loader.log so the validation gate is traceable.
- If ack is still absent, clearly state "still waiting" in both docs updates to avoid confusion.

If Blocked
- If the CLI or pytest fails, capture stderr in $ARTIFACT_ROOT/blocker.log, note the failure signature + command in docs/fix_plan.md Attempts History, leave F1 unchecked, and halt for supervisor guidance rather than guessing at acknowledgement status.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:343 — Current DEBUG-SIM-LINES-DOSE-001.F1 context and Attempts History.
- docs/fix_plan.md:408 — Open TODO describing the acknowledgement dependency.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:1 — CLI to extend with timeline + waiting-clock fields.
- docs/TESTING_GUIDE.md:5-17 — Loader pytest selector and CLI testing guardrails referenced by AUTHORITATIVE_CMDS_DOC.
- docs/development/TEST_SUITE_INDEX.md:5-11 — Confirms tests/test_generic_loader.py coverage and selector expectations.

Next Up (optional)
- Draft a Maintainer <1> escalation note citing the waiting-clock metrics if Maintainer <2> remains silent after this run.

Mapped Tests Guardrail
- `pytest tests/test_generic_loader.py::test_generic_loader -q` already collects (>0) per docs/TESTING_GUIDE.md, so keep it as the validation gate.

Normative Math/Physics
- Not applicable — this loop only touches inbox monitoring tooling.

Summary: Refresh the DEBUG-SIM-LINES-DOSE-001.F1 inbox evidence and loader guard so we can either document Maintainer <2>'s acknowledgement or prove we are still waiting.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T013500Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::main — rerun the acknowledgement scan against `inbox/` and write fresh JSON/Markdown summaries under $ARTIFACT_ROOT/inbox_check, tee stdout to $ARTIFACT_ROOT/inbox_check/check_inbox.log, copy any newly detected Maintainer <2> reply into $ARTIFACT_ROOT/inbox_check/ for posterity, and update docs/fix_plan.md Attempts History + the F1 checklist to cite this scan (mark the checkbox complete only if `ack_detected: true`). If ack remains absent, explicitly log that state in docs/fix_plan.md and keep the TODO open.
- Update: inbox/response_dose_experiments_ground_truth.md::Maintainer Status — append a short "Status as of 2026-01-23T013500Z" section summarizing the inbox scan result and linking to $ARTIFACT_ROOT/inbox_check/inbox_scan_summary.md so Maintainer <2> sees the monitoring evidence.
- Pytest: pytest tests/test_generic_loader.py::test_generic_loader -q
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T013500Z/

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T013500Z && mkdir -p "$ARTIFACT_ROOT/inbox_check"
2. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords acknowledged --keywords ack --keywords confirm --keywords received --keywords thanks --output "$ARTIFACT_ROOT/inbox_check" | tee "$ARTIFACT_ROOT/inbox_check/check_inbox.log"
3. Inspect $ARTIFACT_ROOT/inbox_check/inbox_scan_summary.json. If it lists new Maintainer <2> files, copy them into $ARTIFACT_ROOT/inbox_check/ack_<timestamp>.md (preserve originals in inbox/) and note filename + keywords in docs/fix_plan.md; otherwise record `ack_detected: false` in the next Attempts History entry.
4. Open docs/fix_plan.md and add a new Attempts History row titled `2026-01-23T013500Z — DEBUG-SIM-LINES-DOSE-001.F1 (inbox scan refresh)` that summarizes the CLI output/log paths and the acknowledgement status; update the TODO checklist bullet: mark `[x]` and quote the ack path if Maintainer <2> replied, or keep `[ ]` but mention "still waiting as of 2026-01-23T013500Z" if not.
5. Append a "Status as of 2026-01-23T013500Z" section to inbox/response_dose_experiments_ground_truth.md that links to $ARTIFACT_ROOT/inbox_check/inbox_scan_summary.md, reiterates the tarball SHA, and states whether Maintainer <2> has acknowledged yet.
6. pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/pytest_loader.log"
7. Drop a short README-style note in $ARTIFACT_ROOT/inbox_check/README.md outlining the commands you ran and any acknowledgements observed so Galph can cite it in the summary.

Pitfalls To Avoid
- Stay inside plans/active/DEBUG-SIM-LINES-DOSE-001 and inbox/; do not touch production packages or shipped modules.
- Never delete or rewrite maintainer inbox files—copy them if you need context snapshots.
- Keep the CLI stdout tee so we always have $ARTIFACT_ROOT/inbox_check/check_inbox.log for audit.
- Do not mark F1 complete without an actual Maintainer <2> acknowledgement that includes From: Maintainer <2> plus ack keywords.
- Maintain deterministic ordering in the Markdown summary to avoid churn between scans.
- Capture pytest output to $ARTIFACT_ROOT/pytest_loader.log so we can prove the loader still works.
- Avoid `sudo`, package installs, or environment tweaks; Environment Freeze is in effect.
- If ack status flips to true, immediately cite the exact inbox filename + SHA in docs/fix_plan.md and inbox/response_dose_experiments_ground_truth.md so the maintainer can find it.
- Keep tarball paths and SHA hashes unchanged; reference existing verification logs instead of recomputing large archives.
- Respect non-UTF8 inbox content by relying on the CLI's tolerant reader—don't open files manually with strict decoding.

If Blocked
- If the script or pytest fails (e.g., inbox permissions, missing datasets), capture stderr in $ARTIFACT_ROOT/blocker.log, update docs/fix_plan.md Attempts History with the failure signature, leave F1 unchecked, and stop for supervisor guidance rather than guessing about acknowledgement status.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:7 — Focus context and maintainer request synopsis for DEBUG-SIM-LINES-DOSE-001.
- docs/fix_plan.md:395 — F1 checklist item describing the acknowledgement dependency.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:1 — Non-production inbox scan CLI you will rerun this loop.
- docs/TESTING_GUIDE.md:5 — Loader pytest selector that remains the authoritative validation gate.
- docs/development/TEST_SUITE_INDEX.md:6 — Registry entry confirming tests/test_generic_loader.py coverage expectations.

Next Up (optional)
- If Maintainer <2> still has not replied after this scan, prepare a Maintainer <1> escalation draft outlining the repeated follow-up evidence before the next loop.

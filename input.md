Summary: Automate Maintainer-2 acknowledgement checks for DEBUG-SIM-LINES-DOSE-001 and refresh the loader guard so we can document whether the drop has been accepted.
Focus: DEBUG-SIM-LINES-DOSE-001 — Legacy dose_experiments ground-truth bundle
Branch: dose_experiments
Mapped tests: pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T012900Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::main — add a non-production CLI that scans an inbox directory for files referencing `dose_experiments_ground_truth`, records filename/content matches plus acknowledgement keywords (acknowledg*, confirm*, received, thanks) in JSON + Markdown summaries, emits an `ack_detected` flag, limits previews to a few hundred chars, and writes outputs under $ARTIFACT_DIR/inbox_check while tee-ing stdout into $ARTIFACT_DIR/inbox_check/check_inbox.log.
- Update: docs/fix_plan.md::TODO + Attempts History — log this inbox-scan attempt (list script path, summary/log artifacts, and loader pytest result); if the scan reports `ack_detected: true`, mark F1 complete and cite the maintainer reply path, otherwise keep the checkbox open but note that evidence was refreshed on this timestamp.
- Pytest: pytest tests/test_generic_loader.py::test_generic_loader -q
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T012900Z/

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && ARTIFACT_DIR=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T012900Z && mkdir -p "$ARTIFACT_DIR/inbox_check"
2. Author plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py with: argparse interface (`--inbox`, `--request-pattern`, `--keywords` as repeatable flag, `--since` ISO8601 filter, `--output` dir); tolerant UTF-8 reading with errors='ignore'; skip directories; gather size + modified_utc metadata; detect matches when filename or file content contains the request pattern; capture keyword hits case-insensitively; treat an acknowledgement as detected when any hit contains `acknowledg`, `confirm`, or `received` and the text references Maintainer <2>; emit JSON (`inbox_scan_summary.json`) plus Markdown (`inbox_scan_summary.md`) with tables listing file, match reasons, keywords, preview (<=320 chars), and top-level status fields (`scanned`, `matches`, `ack_detected`, `generated_utc`).
3. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords acknowledged --keywords ack --keywords confirm --keywords received --keywords thanks --output "$ARTIFACT_DIR/inbox_check" | tee "$ARTIFACT_DIR/inbox_check/check_inbox.log"
4. Review $ARTIFACT_DIR/inbox_check/inbox_scan_summary.json; if `ack_detected` is true, copy the maintainer reply into $ARTIFACT_DIR/inbox_check/acknowledgement.md and mention it in docs/fix_plan.md; if false, note "awaiting Maintainer <2>" but still reference the summary + log artifacts.
5. Edit docs/fix_plan.md Attempts History to include this timestamped check and update the F1 checklist text so it references the new script + artifacts; only mark the checkbox complete once Maintainer <2> has responded in inbox/.
6. pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_DIR/pytest_loader.log"

Pitfalls To Avoid
- Keep the new CLI and all outputs inside plans/active/DEBUG-SIM-LINES-DOSE-001; do not touch scripts/ or ptycho/.
- Scan only the project inbox/ tree; avoid recursively walking user home directories.
- Handle non-UTF8 inbox files with errors='ignore' so the CLI never crashes on encoding noise.
- Limit Markdown previews to a few hundred characters to avoid leaking entire maintainer notes into artifacts.
- Sort matches deterministically (e.g., by filename) so repeating the scan produces stable diffs.
- Do not mark F1 complete unless an actual Maintainer <2> acknowledgement file is found.
- Preserve all existing inbox/ files; never rename or delete maintainer communications.
- The pytest selector must exactly match tests/test_generic_loader.py::test_generic_loader and its log must land in $ARTIFACT_DIR/.
- Ensure stdout from the CLI is tee'd to $ARTIFACT_DIR/inbox_check/check_inbox.log for traceability.
- Keep the script dependency-free beyond stdlib so it runs everywhere.

If Blocked
- If the CLI fails (e.g., inbox unreadable or encoding crash) or pytest errors, capture the stack trace in $ARTIFACT_DIR/blocker.log, mention the failure in docs/fix_plan.md Attempts History for F1, and stop for supervisor triage instead of guessing.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:1 — Active checklist + Attempts History for DEBUG-SIM-LINES-DOSE-001, including the F1 row.
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:1 — Current implementation plan scope/status for this initiative.
- inbox/response_dose_experiments_ground_truth.md:1 — Maintainer response that defines the expected delivery contents.
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/rehydration_check/rehydration_summary.md:1 — Latest tarball verification evidence to cite in reports.
- CLAUDE.md:8 — Maintainer identity mapping (Maintainer <1> ↔ <2>) for accurate inbox addressing.

Next Up (optional)
- Once Maintainer <2> acknowledges, mark DEBUG-SIM-LINES-DOSE-001.F1 complete in docs/fix_plan.md and archive the initiative per plan instructions.

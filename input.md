Summary: Ping Maintainer <2> about the delivered dose_experiments bundle, update the fix plan with the follow-up checklist, and rerun the loader guard for fresh evidence.
Focus: DEBUG-SIM-LINES-DOSE-001 — Legacy dose_experiments ground-truth bundle
Branch: dose_experiments
Mapped tests: pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T011900Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001
- Implement: inbox/followup_dose_experiments_ground_truth_2026-01-23T011900Z.md::Main — draft a short follow-up note from Maintainer <1> to Maintainer <2> summarizing the delivered bundle, the tarball rehydration evidence (rehydration_summary/log paths), and the latest pytest loader result; politely request acknowledgement or new asks and copy the final note into the artifacts directory for auditing.
- Update: docs/fix_plan.md::Status + TODOs — add an unchecked `DEBUG-SIM-LINES-DOSE-001.F1` checklist row (maintainer acknowledgement) plus an Attempts History entry logging this follow-up/test run so the roadmap shows that engineering work is complete and we are awaiting Maintainer <2>.
- Pytest: pytest tests/test_generic_loader.py::test_generic_loader -q
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T011900Z/

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && ARTIFACT_DIR=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T011900Z && mkdir -p "$ARTIFACT_DIR"
2. Update docs/fix_plan.md: note that we are awaiting Maintainer <2>, add the new F1 checklist row (follow-up + ack), and describe the follow-up/test evidence in Attempts History referencing the new inbox file + pytest log.
3. Write inbox/followup_dose_experiments_ground_truth_2026-01-23T011900Z.md with: greeting, delivery summary (bundle root + tarball SHA), verification links (`.../bundle_verification.*`, `.../rehydration_check/rehydration_summary.md`), latest pytest loader log path once rerun, and an explicit request for confirmation or additional needs.
4. Copy the follow-up note into "$ARTIFACT_DIR/followup_note.md" for archival (cp inbox/... "$ARTIFACT_DIR/followup_note.md").
5. Run pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_DIR/pytest_loader.log" so we have a fresh loader gate log.

Pitfalls To Avoid
- Keep scope non-production: touch only docs/fix_plan.md, inbox/, and artifact copies under plans/active/DEBUG-*/reports.
- Reference existing artifacts (bundle_verification.md, rehydration_summary.md, tarball SHA) instead of duplicating large data inside the inbox note.
- Use Maintainer identities from CLAUDE.md and clearly state roles (Maintainer <1> → Maintainer <2>).
- Do not rename or delete prior inbox/response files; append a new follow-up file with a timestamp.
- Ensure the follow-up lives in repo inbox (~/Documents/PtychoPINN/inbox/), not in Maintainer <2>'s workspace.
- The pytest run must target the exact selector provided and capture the log via tee into the artifacts directory.
- Document the new TODO + Attempts History row atomically so fix_plan stays consistent with the follow-up action.
- Avoid editing production modules or scripts under scripts/ or ptycho/; this loop is communication + evidence only.
- Keep dataset paths read-only; no copies/moves of the 270 MB bundle besides referencing existing files.
- When citing artifacts, use workspace-relative paths so Maintainer <2> can open them quickly.

If Blocked
- If the inbox note cannot be written (e.g., permission issue) or pytest fails, capture the error/output into "$ARTIFACT_DIR/blocker.log", add a `DEBUG-SIM-LINES-DOSE-001.F1` attempt note in docs/fix_plan.md describing the failure, and halt for supervisor triage.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:1 — Active roadmap entry + checklist for DEBUG-SIM-LINES-DOSE-001.
- inbox/response_dose_experiments_ground_truth.md:1 — Prior maintainer response to extend with the follow-up context.
- specs/data_contracts.md:5 — RawData NPZ schema referenced in the manifest/rehydration evidence.
- docs/TESTING_GUIDE.md:5 — Authoritative command for tests/test_generic_loader.py selector.
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/rehydration_check/rehydration_summary.md:1 — Latest tarball rehydration metrics to cite in the follow-up.

Next Up (optional)
- After Maintainer <2> acknowledges, mark DEBUG-SIM-LINES-DOSE-001 complete in docs/fix_plan.md and select the next roadmap initiative.

Summary: Sync Phase F documentation and registries with dense/test LSQML evidence so F2.4 can close cleanly.
Mode: Docs
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2 — Phase F pty-chi baseline execution
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T233500Z/phase_f_ptychi_baseline_f2_doc_sync/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2:
  - Setup: mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T233500Z/phase_f_ptychi_baseline_f2_doc_sync/{collect,docs} && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  - Implement: docs/TESTING_GUIDE.md::Phase F — add Phase F pty-chi selectors (dense/test run, dry-run, suite) and CLI command snippets with artifact paths captured under 2025-11-04T230000Z.
  - Implement: docs/development/TEST_SUITE_INDEX.md::tests/study/test_dose_overlap_reconstruction.py — register the Phase F module, list key selectors, and cite artifact evidence/log destinations.
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md::F2.4 — flip status to [x], document dense/test doc-sync evidence, and reference the new 2025-11-04T233500Z hub.
  - Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T233500Z/phase_f_ptychi_baseline_f2_doc_sync/collect/pytest_phase_f_cli_collect.log
  - Docs: Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T233500Z/phase_f_ptychi_baseline_f2_doc_sync/docs/summary.md with doc/registry changes and add Attempt #82 entry in docs/fix_plan.md linking to both the 230000Z evidence and the new doc-sync hub.

Priorities & Rationale:
- F2.4 in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:34 expects dense/test docs + registry sync before closing Phase F.
- docs/findings.md:8-17 (POLICY-001, CONFIG-001/002, DATA-001, OVERSAMPLING-001) demand we document compliance in knowledge-base artifacts.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:212-247 flags outstanding TODOs to update docs/TESTING_GUIDE.md and TEST_SUITE_INDEX.md once selectors are green.
- docs/TESTING_GUIDE.md:75-139 currently stops at Phase E selectors, so Phase F guidance is missing despite evidence landing.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/docs/summary.md:1-120 records dense/test logs we must reference in docs.

How-To Map:
- Export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md before each pytest command to satisfy CLI contract.
- Insert Phase F section immediately after the Phase E bullets in docs/TESTING_GUIDE.md, pointing to selectors (`-k "ptychi"`, single-node execution, dry-run) and CLI command `python -m studies.fly64_dose_overlap.reconstruction ... --split test`.
- Add a new table row in docs/development/TEST_SUITE_INDEX.md for `tests/study/test_dose_overlap_reconstruction.py`, summarizing purpose (“Phase F pty-chi orchestration”), listing selectors, and referencing artifact logs under `reports/2025-11-04T230000Z/...`.
- When flipping F2.4 to `[x]`, cite both dense/test evidence (230000Z hub) and new doc-sync artifacts (233500Z hub).
- Run pytest --collect-only after edits to ensure selectors still collect (expect 4 tests) and archive output to `collect/pytest_phase_f_cli_collect.log`.
- Summarize doc-update deltas plus pytest collection proof in the new summary.md and record Attempt #82 in docs/fix_plan.md with artifact links.

Pitfalls To Avoid:
- Do not edit reconstruction CLI code or tests beyond documentation scope; stay documentation-only this loop.
- Keep artifact paths relative (no absolute `/home/` references) when updating docs.
- Ensure collect-only log isn’t overwritten by prior runs—use the new 233500Z hub.
- Avoid altering Phase E documentation; append Phase F guidance separately.
- Do not run full LSQML again; this loop documents existing evidence only.
- Preserve `[x]` states for earlier plan rows when editing plan.md (touch F2.4 only).
- Respect environment freeze—no dependency installs or environment tweaks.
- Make sure pytest collect exits 0; capture and report any failure in summary + ledger if encountered.

If Blocked:
- If pytest --collect-only fails, stop further edits, capture stderr in summary.md, and log the failure plus remediation path in docs/fix_plan.md Attempt #82.
- If doc edits reveal inconsistencies with evidence (e.g., missing log files), document the gap in summary.md and mark F2.4 as `[P]` instead of `[x]`, flagging the missing artifact in docs/fix_plan.md.

Findings Applied (Mandatory):
- POLICY-001 (docs/findings.md:8) — Document PyTorch dependency expectations in Phase F section.
- CONFIG-001 (docs/findings.md:10) — Note that params.cfg bridge occurs before CLI runs; documentation must reinforce this.
- CONFIG-002 (docs/findings.md:11) — Clarify execution-config neutrality in docs/test index entries.
- DATA-001 (docs/findings.md:14) — Reference amplitude + complex64 requirements when summarizing dense/test dataset usage.
- OVERSAMPLING-001 (docs/findings.md:17) — Reiterate K≥C guardrail in study description.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:34
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:212
- docs/TESTING_GUIDE.md:75
- docs/development/TEST_SUITE_INDEX.md:1
- docs/fix_plan.md:55

Next Up (optional):
- After docs sync, advance to sparse/train LSQML execution to exercise skip telemetry under real missing-view conditions.

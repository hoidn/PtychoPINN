Summary: Align Phase D overlap documentation and plans with the new metrics bundle workflow so D4 can close cleanly.
Mode: Docs
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase D (Group-Level Overlap Views)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_overlap.py --collect-only -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T051200Z/phase_d_doc_sync/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.D:
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md::Phase D status — rewrite the Phase D section to mark D1–D3 complete, describe the metrics bundle + CLI evidence from Attempt #10, and link to the artifact hub for verification.
  - Doc: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md::Phase D coverage — update selectors/results to reflect landed tests (metrics manifest + spacing filter), mark the phase COMPLETE, and set plan row D4 to `[x]` once docs + ledger are synced.
  - Log: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation/summary.md — append doc-sync closure notes and produce a new summary in the doc_sync hub capturing what changed; add Attempt #11 to docs/fix_plan.md with the artifact path.
  - Validate: pytest tests/study/test_dose_overlap_overlap.py --collect-only -vv (tee to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T051200Z/phase_d_doc_sync/collect/pytest_collect.log)

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/plan.md:18 keeps D4 at `[P]` until documentation + ledger catch up with metrics bundle behavior.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:105 still describes Phase D as “awaiting implementation”, so readers can’t see the executed metrics bundle workflow.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:64 labels Phase D “(PLANNED)”, missing the RED→GREEN evidence from Attempt #10.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation/summary.md:117 lists doc/test sync as outstanding next actions that must now be resolved.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T051200Z/phase_d_doc_sync/{collect,docs}
- Edit implementation.md/test_strategy.md/plan.md using apply_patch; capture before/after notes in docs/summary.
- Update phase_d_cli_validation/summary.md and author doc_sync/summary.md describing each documentation change with pointers to tests + artifacts.
- pytest tests/study/test_dose_overlap_overlap.py --collect-only -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T051200Z/phase_d_doc_sync/collect/pytest_collect.log
- Append Attempt #11 entry to docs/fix_plan.md referencing the new doc_sync summary + collect log.

Pitfalls To Avoid:
- Do not alter overlap.py code paths—this loop is documentation-only.
- Keep CONFIG-001 guardrails prominent; no claims that imply params.cfg mutations in Phase D.
- Ensure test selectors documented actually exist (use `--collect-only` proof).
- Avoid duplicating large NPZ artifacts in reports; reference existing bundle instead.
- Maintain ASCII formatting in docs; no smart quotes or tabs.
- Cite artifact paths precisely (timestamped directories) when updating docs/fix_plan.md.
- Leave the CLI blocker note about Phase C generator intact unless resolved.
- Preserve findings references (CONFIG-001, DATA-001, OVERSAMPLING-001) in updated sections.

If Blocked:
- Capture the issue in plans/active/.../phase_d_doc_sync/summary.md, note which doc resisted updates, and log the obstacle in docs/fix_plan.md Attempt #11 with return criteria (e.g., conflicting edits on branch).

Findings Applied (Mandatory):
- CONFIG-001 — Documentation must continue to emphasize params.cfg bridge boundaries for overlap utilities.
- DATA-001 — Keep validator references so filtered NPZ outputs remain contract-compliant.
- OVERSAMPLING-001 — Reiterate K≥C reasoning in docs to prevent regressions in future phases.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/plan.md:18
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:105
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:64
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation/summary.md:117

Next Up (optional):
- Draft Phase D → Phase E handoff checklist once documentation is synchronized and D4 is marked complete.

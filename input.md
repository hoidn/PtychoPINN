Summary: Sync Phase E5 documentation and test registries with the new skip summary evidence so we can close the training runner integration.
Mode: Docs
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5 — training runner integration
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py --collect-only -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T084850Z/phase_e_training_e5_doc_sync/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5:
  - Implement: docs/TESTING_GUIDE.md::Section 2 (Study selectors) — document skip_summary.json expectations plus deterministic CLI command, and propagate the same evidence into `plans/active/.../implementation.md` and `plans/active/.../test_strategy.md` per tasks D1–D2.
  - Sync: docs/development/TEST_SUITE_INDEX.md::test_dose_overlap_training.py — add skip summary coverage details and point to the 2025-11-04T170500Z artifact hub.
  - Validate: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; pytest tests/study/test_dose_overlap_training.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T084850Z/phase_e_training_e5_doc_sync/collect/pytest_collect_final.log.
  - Doc: Append a completion addendum to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/docs/summary.md summarizing the updated docs and registry changes.

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/plan.md:21 keeps T5 `[P]` until the documentation sync lands.
- docs/TESTING_GUIDE.md:75 still lists Phase E selectors without the new skip summary requirement; needs refresh to match Attempt #25.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:125-173 references E4 evidence and future work; must be updated to show E5 completion and artifact links.
- docs/development/TEST_SUITE_INDEX.md:60 currently lacks mention of skip summary persistence in `test_dose_overlap_training.py`.
- docs/fix_plan.md:44 records Attempt #25 but flags documentation/registry sync as outstanding; this loop resolves that gap.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- Edit docs/TESTING_GUIDE.md Section 2 to add skip summary narrative plus updated selector snippet referencing `pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv` and CLI dry-run command with artifact path.
- Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md Phase E5 entry to mark tasks complete and cite `reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/`.
- Revise plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md Phase E section with new skip summary selectors, RED/GREEN/collect evidence, and dry-run logs.
- Amend docs/development/TEST_SUITE_INDEX.md row for `test_dose_overlap_training.py` to cover skip summary assertions and log locations.
- pytest tests/study/test_dose_overlap_training.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T084850Z/phase_e_training_e5_doc_sync/collect/pytest_collect_final.log
- Extend plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/docs/summary.md with doc sync completion notes (updated docs, collect log path, findings alignment).

Pitfalls To Avoid:
- Do not modify production code; limit edits to documentation, plan, and registry files noted above.
- Keep all new artifacts under the 2025-11-04T084850Z hub; no stray logs in repo root or tmp/ once finished.
- Preserve CONFIG-001 messaging—builder remains pure and skip logic lives in CLI layer.
- When documenting commands, keep deterministic flags (`--accelerator cpu`, `--deterministic`, `--num-workers 0`, `--dry-run`) to avoid misleading guidance.
- Ensure collect-only run actually executes after doc edits; if it fails, capture log and stop for supervisor guidance.
- Retain references to existing RED→GREEN evidence; do not delete prior log pointers from test_strategy.md.

If Blocked:
- Store failing collect-only output in `reports/2025-11-04T084850Z/phase_e_training_e5_doc_sync/collect/pytest_collect_error.log`, note the blocker in summary.md, and update docs/fix_plan.md Attempts with the obstruction before pausing.

Findings Applied (Mandatory):
- POLICY-001 — Documentation must continue to assert PyTorch backend as required for Phase E workflows.
- CONFIG-001 — Highlight that skip summary lives in the runner CLI while builder stays pure.
- DATA-001 — Reinforce canonical NPZ expectations when describing regeneration commands.
- OVERSAMPLING-001 — Note sparse skips stem from spacing threshold enforcement; no threshold relaxation allowed.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/plan.md:17
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:125
- docs/TESTING_GUIDE.md:86
- docs/development/TEST_SUITE_INDEX.md:60
- docs/fix_plan.md:44

Next Up (optional):
- If time remains after doc sync, prepare checklist to transition Phase E to aggregated gs2 training evidence (Phase E6).

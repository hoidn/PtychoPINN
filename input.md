Summary: Align Phase E5 job enumeration with Phase D view subdirectories so the baseline CLI run can succeed and document the skip behaviour for missing overlap datasets.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5 — training runner integration
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_matrix -vv; pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_skips_missing_view -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv; pytest tests/study/test_dose_overlap_training.py --collect-only -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5:
  - Test: Update `tests/study/test_dose_overlap_training.py::test_build_training_jobs_matrix` to mirror the actual Phase D layout (`dose_X/{view}/{view}_{split}.npz`) and author a new RED test `tests/study/test_dose_overlap_training.py::test_build_training_jobs_skips_missing_view` that exercises `build_training_jobs(..., allow_missing_phase_d=True)` when the sparse view directory is absent; capture both failures to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/red/pytest_build_training_jobs_red.log`.
  - Implement: studies/fly64_dose_overlap/training.py::build_training_jobs — point dense/sparse job paths at `dose_{dose}/{view}/{view}_{split}.npz`, add an `allow_missing_phase_d` switch that skips overlap jobs whose NPZs are missing (logging the omission), and update `main()` to enable the non-strict mode for CLI execution while preserving strict defaults for tests.
  - Validate: Run `pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_matrix -vv`, `pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_skips_missing_view -vv`, `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv`, and `pytest tests/study/test_dose_overlap_training.py --collect-only -vv`, teeing outputs under the new artifact hub (`red/` for pre-change failures, `green/` for post-change passes, `collect/` for discovery).
  - Run: Regenerate Phase C/D fixtures (`python -m studies.fly64_dose_overlap.generation ...`, `python -m studies.fly64_dose_overlap.overlap ...`) and rerun the CLI baseline with deterministic knobs (`python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_training_evidence --phase-d-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/real_run --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 --logger csv`), archiving stdout/logs + manifest to the real_run subdirectory.
  - Doc: Refresh `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/docs/summary.md`, mark plan/test_strategy E5 checklist rows accordingly, sync `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` with the new selector, and record Attempt #21 in `docs/fix_plan.md`.

Priorities & Rationale:
- docs/fix_plan.md:33-55 (Attempt #20) records the path mismatch and CLI failure; fixing that regression is prerequisite for closing Phase E5.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:28 keeps E5 `[P]` until real-run evidence plus manifest/log updates land.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:84-170 mandates deterministic CLI execution and coverage for missing overlap data scenarios before declaring Phase E complete.
- specs/data_contracts.md:190-260 + docs/findings.md (DATA-001) require we reference actual filtered NPZ locations that Phase D produces under per-view subdirectories.
- docs/DEVELOPER_GUIDE.md:68-104 + docs/findings.md (CONFIG-001, POLICY-001, OVERSAMPLING-001) continue to constrain runner wiring and gridsize semantics during the fix.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/{red,green,collect,docs,real_run}
- pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_matrix -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/red/pytest_build_training_jobs_red.log
- pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_skips_missing_view -vv 2>&1 | tee -a plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/red/pytest_build_training_jobs_red.log
- (after implementation) rerun both selectors with tee to `.../green/pytest_build_training_jobs_green.log`
- pytest tests/study/test_dose_overlap_training.py -k training_cli -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/green/pytest_training_cli_suite_green.log
- pytest tests/study/test_dose_overlap_training.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/collect/pytest_collect.log
- python -m studies.fly64_dose_overlap.generation --base-npz datasets/fly/fly001_transposed.npz --output-root tmp/phase_c_training_evidence 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/real_run/phase_c_generation.log
- python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_training_evidence --output-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/real_run/overlap_cli 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/real_run/phase_d_overlap.log
- python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_training_evidence --phase-d-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/real_run --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 --logger csv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/real_run/training_cli_real_run.log
- Archive lightning checkpoints/logs under `.../real_run/lightning_logs/` and note manifest diffs in summary.md

Pitfalls To Avoid:
- Do not leave the new `allow_missing_phase_d` defaulting to permissive mode; unit tests rely on strict validation unless explicitly disabled.
- Keep dataset fixtures DATA-001 compliant; no empty NPZ placeholders when exercising missing-view logic.
- Ensure CLI logging explains skipped overlap jobs so operators understand sparse failures without reading source.
- Avoid mutating global params.cfg in `build_training_jobs`; CONFIG-001 bridge must remain in execution helper.
- Capture RED logs before implementation so the artifact history demonstrates TDD sequencing.
- Keep file outputs inside the 2025-11-04T150500Z hub; clean `tmp/` of large intermediates after CLI run to avoid clutter.
- Force CPU execution (`--accelerator cpu`) and deterministic flags to prevent nondeterministic regressions in evidence.
- Do not downgrade spacing thresholds here; missing sparse datasets should be logged, not silently fabricated.

If Blocked:
- Store failing pytest/CLI logs under the new artifact hub (`red/` or `real_run/`) with clear filenames, summarize the blocker (path? sparse rejection?) in summary.md, and add a blocked Attempt entry to docs/fix_plan.md before stopping.

Findings Applied (Mandatory):
- POLICY-001 — Maintain PyTorch backend as mandatory while wiring CLI runner.
- CONFIG-001 — Keep the legacy bridge sequencing intact when rerouting dataset loading.
- DATA-001 — Respect canonical NPZ layout when adjusting job paths and fixtures.
- OVERSAMPLING-001 — Preserve gridsize/neighbor_count invariants while filtering jobs.

Pointers:
- docs/fix_plan.md:33
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:28
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:163
- studies/fly64_dose_overlap/training.py:150
- tests/study/test_dose_overlap_training.py:620

Doc Sync Plan: After GREEN, rerun `pytest tests/study/test_dose_overlap_training.py --collect-only -vv` (log to `collect/pytest_collect_final.log`), then update `docs/TESTING_GUIDE.md` §2 (Phase E selectors) and `docs/development/TEST_SUITE_INDEX.md` with the new skip-aware selector reference.

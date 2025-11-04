# Phase E5 Real-Run Evidence & Skip Summary Plan

## Context
- Initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 (Phase E5 training runner integration)
- Goal: Produce deterministic CLI execution evidence with skip reporting persisted to disk and close Phase E5 deliverables (docs/test registry/test strategy).
- Dependencies: Phase C/D artifact generation pipelines (reuse or regenerate), existing skip-aware manifest implementation from Attempt #23, policy findings CONFIG-001 / DATA-001 / OVERSAMPLING-001.

## Deliverables
1. CLI skip summary persisted as standalone JSON alongside manifest (artifact reference for downstream analytics).
2. Deterministic baseline CLI run (dose=1e3, view=baseline, gridsize=1) with stdout/logs/manifest archived under new hub.
3. TDD evidence (RED→GREEN + collect) for skip summary file requirement.
4. Documentation sync: test registry + testing guide + plan/test_strategy updates + fix plan Attempt #24 entry.

## Tasks
| ID | Description | State | Guidance |
| --- | --- | --- | --- |
| T1 | Author RED expectation for skip summary file in `tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging`. | [ ] | Force delete any prior `skip_summary.json`, run targeted pytest to capture RED log at `reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/red/pytest_training_cli_manifest_red.log`. |
| T2 | Implement skip summary persistence in `studies/fly64_dose_overlap/training.py::main` (write JSON file, populate manifest path). | [ ] | Ensure file lives at `<artifact_root>/skip_summary.json`, contains same list as manifest['skipped_views'], and manifest records relative path. Preserve CONFIG-001 purity for builder. |
| T3 | GREEN tests: rerun targeted selectors + CLI suite + collect proof. | [ ] | `pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv`, `pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_skips_missing_view -vv`, and `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv`; tee outputs under `green/` & `collect/`. |
| T4 | Deterministic CLI baseline run capturing skip summary + manifest + Lightning logs. | [ ] | If prior Phase C/D outputs stale, regenerate via generation/overlap CLIs; execute training CLI with deterministic flags, store stdout/logs/manifest/skip summary under `real_run/` and `cli/`. |
| T5 | Documentation + ledger sync. | [ ] | Update `summary.md`, plan/test_strategy, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, and record Attempt #24 in docs/fix_plan.md referencing this hub. |

## Exit Criteria
- Skip summary JSON exists with >=1 record when sparse view missing, referenced in summary.md and manifest.
- GREEN test logs + collect proof archived.
- Real-run CLI artifacts captured with deterministic flags and pointer logged in fix plan.
- Plan/test strategy mark Phase E5 deliverables COMPLETE.

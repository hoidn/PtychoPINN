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
| T1 | Author RED expectation for skip summary file in `tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging`. | [x] | RED log captured at `reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/red/pytest_training_cli_manifest_red.log` showing `AssertionError: skip_summary.json not found`. |
| T2 | Implement skip summary persistence in `studies/fly64_dose_overlap/training.py::main` (write JSON file, populate manifest path). | [x] | Skip summary now persisted beside manifest (Phase E5.5); see `training.py:692-731`. |
| T3 | GREEN tests: rerun targeted selectors + CLI suite + collect proof. | [x] | `green/` logs show selectors PASS (`pytest_training_cli_manifest_green.log`, `pytest_training_cli_skips_green.log`, `pytest_training_cli_suite_green.log`); collect proof stored at `collect/pytest_collect.log`. |
| T4 | Deterministic CLI baseline run capturing skip summary + manifest + Lightning logs. | [x] | Dry-run CLI evidence archived under `real_run/` (manifest, skip summary, stdout, Lightning logs). |
| T5 | Documentation + ledger sync. | [P] | Outstanding: document skip summary requirement in guides/test index and mark Phase E5 rows complete in implementation/test strategy (ledger entry captured in Attempt #25). |

## Exit Criteria
- Skip summary JSON exists with >=1 record when sparse view missing, referenced in summary.md and manifest.
- GREEN test logs + collect proof archived.
- Real-run CLI artifacts captured with deterministic flags and pointer logged in fix plan.
- Plan/test strategy mark Phase E5 deliverables COMPLETE.

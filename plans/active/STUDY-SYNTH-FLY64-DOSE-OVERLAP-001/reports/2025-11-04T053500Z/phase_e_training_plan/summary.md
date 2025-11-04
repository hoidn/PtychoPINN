# Phase E Training Plan Authoring Summary

**Timestamp:** 2025-11-04T053500Z  
**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase E — Train PtychoPINN  
**Mode:** Planning

## Objectives
- Capture the task breakdown for Phase E, bridging dense/sparse overlap datasets to deterministic PtychoPINN training runs.
- Align planned work with CONFIG-001 (legacy bridge), DATA-001 (dataset contract), and OVERSAMPLING-001 (grouping constraints) guardrails.
- Establish test infrastructure expectations before implementation per CLAUDE.md directives.

## Key Decisions
1. **Training Job Matrix:** Enumerate 3 doses × 2 overlap views plus gs1 baseline, yielding 6 grouped jobs and 3 baseline jobs. Jobs carry dataset paths (Phase C baseline vs Phase D overlap), StudyDesign metadata, and artifact output directories.
2. **Testing Strategy:** Tests will focus on job enumeration (counts, paths, metadata) and execution wiring via dependency injection (stub runner). Documented in `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md` Phase E section (to be updated in E1).
3. **Execution Wrapper:** `run_training_job` will accept injected runner (defaulting to `ptycho_train`) to keep tests fast; dry-run flag ensures verification without heavy training.
4. **CLI Expectations:** Provide CLI parity with Phase C/D modules, including `--artifact-root`, selectors (`--dose`, `--view`, `--gridsize`), and manifest emission for accountability.

## Artifacts
- `plan.md` (this directory) detailing tasks E1–E4 with guidance and references.
- Future evidence (logs, manifests, checkpoints) must live under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/<timestamp>/phase_e_training_run/`.

## Next Actions (Engineer Loop)
1. Update `test_strategy.md` Phase E section with the documented test matrix and dependency injection approach (E1).
2. Author `tests/study/test_dose_overlap_training.py` RED case covering `build_training_jobs` expectations (counts, metadata). Capture RED log under new artifact hub.
3. Implement `studies/fly64_dose_overlap/training.py::TrainingJob` + `build_training_jobs` to satisfy RED test and rerun (GREEN).
4. Update `docs/fix_plan.md` with Attempt #12 once artifacts/logs captured.

All artifacts for this planning pass reside at `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/`.

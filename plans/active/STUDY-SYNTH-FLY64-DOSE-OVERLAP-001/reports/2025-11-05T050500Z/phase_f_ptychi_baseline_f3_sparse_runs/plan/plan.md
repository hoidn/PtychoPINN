# Phase F3 — Sparse LSQML Execution

## Context
- Initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Phase Goal: Prove sparse overlap views (train/test) run end-to-end through the pty-chi LSQML pipeline after the greedy downsampling fix, with manifest telemetry exposing selection strategy + acceptance metrics.
- Dependencies:
  - Attempt #86 (Phase D7 greedy rescue) — sparse NPZs now emitted.
  - Phase F plan (reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md) F3 tasks.
  - Test strategy Phase F section updated 2025-11-05 to track sparse selectors.

## Checklist

| ID | Task | State | Guidance |
| --- | --- | --- | --- |
| F3.A | Author RED test highlighting missing `selection_strategy` metadata for sparse jobs | [ ] | Update `tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs` (or introduce `test_cli_reports_sparse_selection_strategy`) so it asserts manifest entries contain `selection_strategy` + acceptance metrics. Capture failure in `reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/red/pytest_phase_f_sparse_red.log`. |
| F3.B | Implement manifest/summary metadata surfacing (CLI + tests GREEN) | [ ] | Patch `studies/fly64_dose_overlap/reconstruction.py::main` (and supporting helpers) to inspect Phase D metadata (selection strategy, acceptance rate) when enumerating sparse jobs and persist to manifest/summary. Confirm GREEN run through pytest and collect-only logs under `.../green/` and `.../collect/`. |
| F3.C | Execute sparse/train LSQML run (num_epochs=100, deterministic) capturing CLI transcript + reconstruction log | [ ] | Command stored in `input.md`; capture CLI stdout/stderr under `cli/sparse_train.log`, reconstruction log under `real_run/dose_{dose}/sparse/train/reconstruction.log`, updated manifest JSON, and summary note with acceptance stats. |
| F3.D | Execute sparse/test LSQML run mirroring train settings | [ ] | Store artifacts under `real_run/dose_{dose}/sparse/test/`; include log, manifest snapshot, skip summary, and metrics snippet via `summary.md`. |
| F3.E | Documentation + registry sync | [ ] | Update `plans/.../implementation.md` Phase F section, `test_strategy.md`, `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`, and `docs/fix_plan.md` Attempt entry. Capture collect-only proof (`collect/pytest_phase_f_sparse_collect.log`) and doc summary under `.../docs/summary.md`. |

## Artifact Hub
`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/`

Subdirectories:
- plan/ — this file, checklist updates
- red/ — RED pytest logs (metadata missing)
- green/ — GREEN pytest logs after implementation
- collect/ — pytest --collect-only outputs
- cli/ — CLI transcripts (dry-run + real-run)
- real_run/ — per-dose/view/split LSQML logs
- docs/ — summary.md, registry update notes

## References
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md`
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md`
- `docs/TESTING_GUIDE.md` (§Phase F authoritative commands)
- `docs/findings.md` — POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001
- `docs/GRIDSIZE_N_GROUPS_GUIDE.md:143` — sparse spacing constraint (102.4 px)

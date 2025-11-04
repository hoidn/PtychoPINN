Summary: Capture skip metadata for missing Phase D sparse views and prove Phase F CLI handles it with TDD plus a sparse run.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2 — Phase F pty-chi baseline execution
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_skips_missing_phase_d -vv; pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2:
  - Setup: mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/{red,green,collect,cli,docs} && export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  - Validate (RED): pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_skips_missing_phase_d -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/red/pytest_sparse_skip_red.log
  - Implement: tests/study/test_dose_overlap_reconstruction.py::test_cli_skips_missing_phase_d — author RED test asserting sparse jobs without Phase D NPZs are skipped with manifest + skip summary metadata.
  - Implement: studies/fly64_dose_overlap/reconstruction.py::build_ptychi_jobs — when allow_missing=True, drop missing overlap jobs, append skip events, and surface them through CLI manifest/skip summary.
  - Validate: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_skips_missing_phase_d -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/green/pytest_sparse_skip_green.log
  - Verify: pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/green/pytest_phase_f_suite_green.log
  - Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.reconstruction --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/cli --dose 1000 --view sparse --split train --dry-run --allow-missing-phase-d 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/cli/dry_run_sparse.log
  - Docs: Summarize results + skip metadata in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/docs/summary.md and log Attempt #83 in docs/fix_plan.md with artifact links.

Priorities & Rationale:
- docs/fix_plan.md:31-56 — Phase F focus remains open; next milestone is sparse view handling before Phase G comparisons.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:228-260 — F2 selectors expect skip metadata parity with training CLI; add regression coverage.
- docs/TESTING_GUIDE.md:146-208 — Phase F instructions require manifest + skip summary evidence for new scenarios.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/docs/summary.md:101-154 — Dense/test run summary notes sparse runs outstanding pending skip tooling.

How-To Map:
- RED: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_skips_missing_phase_d -vv
- GREEN: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_skips_missing_phase_d -vv
- Suite: pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv
- CLI dry run: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.reconstruction --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/cli --dose 1000 --view sparse --split train --dry-run --allow-missing-phase-d
- Artifact sync: tee logs into red/green/collect/cli dirs; update docs/summary and docs/fix_plan.md after GREEN.

Pitfalls To Avoid:
- Do not edit Phase D/E generators; focus on Phase F CLI only.
- Keep skip reasons deterministic (no timestamps beyond manifest top-level).
- Avoid absolute paths in tests/CLI; rely on tmp_path fixtures or repo-relative resolution.
- Ensure new test fails before implementation to satisfy TDD.
- Do not introduce torch imports in tests; rely on mocks/stubs.
- Maintain CONFIG-001 separation — builder must stay pure.
- Record CLI environment via AUTHORITATIVE_CMDS_DOC; no ad-hoc env changes.
- Keep artifact tree confined to initiative reports directory; no root-level logs.
- Preserve existing manifest schema; extend with skip metadata without removing fields.
- Do not mark doc registries updated unless edits land this loop.

If Blocked:
- If sparse NPZ generation is missing entirely, capture manifest/skip_summary state, stash logs under artifacts, and mark Attempt #83 blocked in docs/fix_plan.md with next steps (regenerate Phase D sparse data).

Findings Applied (Mandatory):
- POLICY-001 — Phase F CLI assumes torch>=2.2 available; document skip behavior without removing dependency.
- CONFIG-001 — Keep reconstruction builder pure; skip metadata must not mutate params.cfg.
- CONFIG-002 — Execution configs stay isolated from manifest logic.
- DATA-001 — Manifest/test fixtures must continue asserting amplitude + complex64 requirements even when skipping jobs.
- OVERSAMPLING-001 — Skip reasoning should cite spacing guard when sparse view absent.

Pointers:
- docs/fix_plan.md:31
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:31
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:228
- docs/TESTING_GUIDE.md:146
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/docs/summary.md:101

Next Up (optional):
- Sparse/test LSQML run once skip instrumentation is green.

Doc Sync Plan:
- After GREEN, run pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/collect/pytest_phase_f_collect.log, then extend docs/TESTING_GUIDE.md Phase F section and TEST_SUITE_INDEX once skip test stabilized.

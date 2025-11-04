Summary: Stage Phase F by adding the RED pty-chi job manifest test and updating the study test strategy so LSQML scaffolding can proceed.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F0 — Phase F pty-chi baseline scaffolding
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_reconstruction.py::test_build_ptychi_jobs_manifest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F0:
  - Implement: tests/study/test_dose_overlap_reconstruction.py::test_build_ptychi_jobs_manifest — add the RED test expecting a 3×2+baseline LSQML manifest derived from Phase E artifacts (will currently fail because `build_ptychi_jobs` is a stub); capture log in red/pytest_phase_f_red.log.
  - Strategy: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md::Phase F — document planned selectors, RED/GREEN evidence requirements, and CLI invocation rules for LSQML runs.
  - Scaffold: studies/fly64_dose_overlap/__init__.py — import the forthcoming `reconstruction` module so tests can locate the stub (leave implementation to the GREEN loop but add placeholder raising NotImplementedError).
  - Validate: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py::test_build_ptychi_jobs_manifest -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline/red/pytest_phase_f_red.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline/

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md: F0 tasks demand test strategy + RED test before implementation.
- docs/findings.md: CONFIG-001/ DATA-001/ POLICY-001/ OVERSAMPLING-001 enforce reconstruction guardrails we must encode in test strategy + fixtures.
- specs/data_contracts.md:210-276 defines reconstruction NPZ expectations; RED test should cite these inputs.
- docs/TESTING_GUIDE.md:102-154 prescribes authoritative commands and reconstruction prerequisites — keep Phase F instructions aligned.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:Phase F outlines LSQML baseline requirements needing manifest-driven automation.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- Edit plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md Phase F section per template (selectors, evidence, findings alignment, artifact destinations).
- Create tests/study/test_dose_overlap_reconstruction.py with RED test asserting manifest length, dose/view combinations, and CLI arguments; allow current stub to raise NotImplemented which will trigger the failure.
- Add stub `build_ptychi_jobs` raising NotImplementedError in studies/fly64_dose_overlap/reconstruction.py and expose via __all__ in __init__.py.
- Run pytest tests/study/test_dose_overlap_reconstruction.py::test_build_ptychi_jobs_manifest -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline/red/pytest_phase_f_red.log

Pitfalls To Avoid:
- Do not touch Phase E training implementation beyond the new reconstruction stub.
- Keep reconstruction paths relative to Phase D/E artifact roots; no hard-coded tmp paths.
- Avoid importing cupy/tike in tests — use placeholders until GREEN loop.
- Ensure RED test uses deterministic fixture data (Phase E manifest) instead of triggering heavy CLI runs.
- Do not update docs/TESTING_GUIDE.md or TEST_SUITE_INDEX yet; defer until GREEN evidence exists.
- Leave long-running LSQML executions for later phases to respect loop budget.
- Keep AUTHORITATIVE_CMDS_DOC exported before pytest to satisfy governance checks.
- Do not run pty-chi script in this loop; focus on scaffolding + RED failure capture.

If Blocked:
- Capture failing pytest output to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline/red/pytest_phase_f_error.log and log blocker + stack trace in docs/summary.md, then notify supervisor via docs/fix_plan.md Attempts History.

Findings Applied (Mandatory):
- CONFIG-001 — Keep reconstruction stub free of params.cfg mutations until runner integrates update_legacy_dict.
- DATA-001 — RED test must reference canonical NPZ layout for diffraction/object paths.
- POLICY-001 — Document that PyTorch backend remains required when invoking LSQML pipeline.
- OVERSAMPLING-001 — Note K≥C guard in fixtures so sparse jobs remain validated once GREEN.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:34
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:188
- docs/TESTING_GUIDE.md:102
- specs/data_contracts.md:210
- docs/findings.md:1

Next Up (optional):
- Implement build_ptychi_jobs + CLI (F1.1–F1.3) once RED test captured.
- Schedule initial LSQML dry-run using new CLI.

Doc Sync Plan (Conditional): After GREEN implementation adds new selectors, rerun `pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv`, archive log under collect/, then update docs/TESTING_GUIDE.md §2 and docs/development/TEST_SUITE_INDEX.md with the new Phase F entry.

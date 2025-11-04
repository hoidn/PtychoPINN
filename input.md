Summary: Finalize Phase D overlap metrics bundle + CLI artifact workflow and record evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase D (Group-Level Overlap Views)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_metrics_manifest -vv; pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.D:
  - Implement: studies/fly64_dose_overlap/overlap.py::generate_overlap_views — add `metrics_bundle_path` JSON emission (train/test aggregated) and surface path so CLI can copy `metrics/<dose>/<view>.json`; update `main` to copy the bundle + overlap manifest into `--artifact-root`.
  - Test: tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_metrics_manifest — move to RED by asserting `metrics_bundle_path` exists and contains both train/test entries, then drive GREEN after implementation; keep `pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv` as regression selector.
  - CLI: If Phase C outputs are absent, run `python -m studies.fly64_dose_overlap.generation --output-root tmp/fly64_phase_c_cli`; then run `python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/fly64_phase_c_cli --output-root tmp/phase_d_overlap_views --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation/` teeing logs and preserving copied metrics + manifest.
  - Document: Refresh `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/{implementation.md,test_strategy.md}` Phase D sections, update `reports/2025-11-04T045500Z/phase_d_cli_validation/summary.md`, extend `docs/TESTING_GUIDE.md` study suite + `docs/development/TEST_SUITE_INDEX.md`, and append Attempt #9 in `docs/fix_plan.md` with artifact links.
  - Validating selector: pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation/

Priorities & Rationale:
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/plan.md:16-18 — Phase D charter still requires consolidated metrics bundles copied under reports hub and CLI artifact capture.
  - studies/fly64_dose_overlap/overlap.py:372 — generate_overlap_views currently lacks a metrics bundle file, preventing plan-required `metrics/<dose>/<view>.json`.
  - docs/TESTING_GUIDE.md:1 — Authoritative commands require selector documentation + collect-only proof once tests land.
  - specs/data_contracts.md:207 — DATA-001 validator must continue to guard filtered outputs after metrics restructure.
  - summary.md artifact (plans/active/.../phase_d_metrics_alignment/summary.md:196) — Next actions call for CLI smoke test + doc sync using new `--artifact-root`.

How-To Map:
  - export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  - mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation/{red,green,collect,cli,metrics}
  - pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_metrics_manifest -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation/red/pytest_metrics_bundle.log
  - Implement overlap.py + updated test; rerun `pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_metrics_manifest -vv` teeing to `green/pytest_metrics_bundle.log`, then `pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv` teeing to `green/pytest_spacing.log`.
  - pytest tests/study/test_dose_overlap_overlap.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation/collect/pytest_collect.log
  - If `tmp/fly64_phase_c_cli` missing, run `python -m studies.fly64_dose_overlap.generation --output-root tmp/fly64_phase_c_cli` teeing stdout to `cli/phase_c_generation.log`.
  - python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/fly64_phase_c_cli --output-root tmp/phase_d_overlap_views --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation/cli/phase_d_overlap.log
  - find plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation/metrics -maxdepth 3 -type f -print > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation/metrics/metrics_inventory.txt
  - rm -rf tmp/phase_d_overlap_views

Pitfalls To Avoid:
  - Keep overlap pipeline params.cfg-neutral (CONFIG-001); no legacy imports inside tests.
  - Maintain backward compatibility for CLI args; `--artifact-root` optional with no behavior change when omitted.
  - Do not drop existing per-split metrics unless bundle replaces them downstream; update manifest accordingly.
  - Tests must stay lightweight (synthetic arrays in tmp_path) and clean up generated files.
  - Copy only JSON artifacts into reports hub; avoid huge NPZ duplication under artifacts.
  - Document CLI commands and logs in summary.md before finalizing.
  - Ensure tmp directories removed only after artifacts copied; capture removal command output if it fails.
  - Avoid hardcoding absolute paths; use Path joins to build artifact paths.
  - Treat missing Phase C dataset as blocker unless CLI generation completes successfully.

If Blocked:
  - Capture stderr/stdout to `cli/blocker.log`, summarize the failure in summary.md, note blocker + return criteria in docs/fix_plan.md, and leave tmp outputs for investigation.

Findings Applied (Mandatory):
  - CONFIG-001 — maintain params.cfg bridge boundaries while touching overlap.py.
  - DATA-001 — validator enforces NPZ contract post-filtering; new metrics bundle must not bypass validation.
  - OVERSAMPLING-001 — neighbor_count 7 ≥ gridsize² 4; document acceptance stats in metrics bundle.

Pointers:
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/plan.md:16
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T041900Z/phase_d_metrics_alignment/summary.md:204
  - studies/fly64_dose_overlap/overlap.py:360
  - tests/study/test_dose_overlap_overlap.py:322
  - specs/data_contracts.md:207

Next Up (optional):
  - Draft Phase D → Phase E handoff checklist once metrics artifacts validated.

Doc Sync Plan:
  - After GREEN, add the metrics bundle selector to docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md; archive collect-only output and CLI logs under the new artifact hub before updating docs.

Mapped Tests Guardrail:
  - Confirm `pytest tests/study/test_dose_overlap_overlap.py --collect-only -vv` reports the updated metrics manifest test; if it fails to collect, halt, capture log under `collect/`, and mark attempt blocked in docs/fix_plan.md.

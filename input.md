Summary: Extend Phase F2 by fixing the script-unit test path and executing the dense/test LSQML run with fresh artifacts.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2 — Phase F pty-chi baseline execution
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/scripts/test_ptychi_reconstruct_tike.py::test_main_uses_cli_arguments -vv; pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv; pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2:
  - Setup: mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/{red,green,collect,cli,real_run,docs} && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  - Test: pytest tests/scripts/test_ptychi_reconstruct_tike.py::test_main_uses_cli_arguments -vv || true 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/red/pytest_ptychi_cli_input_red.log
  - Implement: tests/scripts/test_ptychi_reconstruct_tike.py::test_main_uses_cli_arguments — swap hard-coded `/home/ollie/Documents/PtychoPINN2/...` import for repo-relative discovery via `Path(__file__).resolve().parents[2]` so the selector runs from any clone.
  - Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_ptychi_reconstruct_tike.py::test_main_uses_cli_arguments -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/green/pytest_ptychi_cli_input_green.log
  - Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/green/pytest_phase_f_cli_exec_green.log
  - Suite: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/green/pytest_phase_f_cli_suite_green.log
  - Collect: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/collect/pytest_phase_f_cli_collect.log
  - Real Run: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.reconstruction --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/real_run --dose 1000 --view dense --split test --allow-missing-phase-d 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/cli/real_run_dense_test.log
  - Docs: Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/docs/summary.md with outcomes and add manifest/log references; sync docs/TESTING_GUIDE.md §Phase F and docs/development/TEST_SUITE_INDEX.md with the new script selector and dense/test evidence; record Attempt #81 in docs/fix_plan.md referencing artifact paths and plan/test strategy updates.

Priorities & Rationale:
- Phase F plan F2.4 (plan.md:34) mandates dense/test coverage to unblock comparisons.
- docs/findings.md POLICY-001/CONFIG-001 enforce PyTorch availability and params.cfg neutrality during CLI runs.
- specs/data_contracts.md §4 ensures regenerated NPZ inputs remain DATA-001 compliant before executing LSQML.
- docs/TESTING_GUIDE.md §4 requires capturing RED/GREEN pytest logs plus CLI transcripts for acceptance evidence.
- test_strategy.md Phase F section notes TODO to remove absolute script path and sync registries before closing F2.

How-To Map:
- Ensure tmp/phase_{c,d}_f2_cli directories exist from prior attempt; regenerate datasets with existing script if missing before running CLI.
- Use pathlib relative import inside the test: `script_path = Path(__file__).resolve().parents[2] / "scripts" / "reconstruction" / "ptychi_reconstruct_tike.py"` then load via importlib.
- After CLI run, verify `real_run/dose_1000/dense/test/ptychi.log` plus visualization exist; copy manifest + skip summaries alongside run output.
- Update docs by adding command snippets and artifact links; keep logs referenced via relative repo paths.
- Gather evidence for ledger: summarize RED→GREEN outcomes, CLI status code, and doc/test updates in docs/fix_plan.md Attempt #81.

Pitfalls To Avoid:
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` — stay within test file + docs.
- Keep argparse defaults untouched while enabling overrides; ensure test still verifies fallback behavior.
- Prevent accidental deletion of prior artifacts under reports/2025-11-04T210000Z; new evidence must live in the 230000Z hub.
- Maintain DATA-001 key/dtype expectations when touching synthetic payload generation scripts.
- Capture RED logs before implementation; do not overwrite with GREEN output.
- Avoid swallowing non-zero return codes in CLI run; surface failure and log manifest entry if encountered.
- Ensure pytest selectors run with AUTHORITATIVE_CMDS_DOC set and tee outputs to the artifacts directory.
- Document any skips or deselections in summary.md; justify if CLI run aborted early.
- Respect environment freeze — no pip installs or package upgrades.
- Keep tmp directories tidy after run to avoid cross-contamination between train/test splits.

If Blocked:
- If the CLI run fails due to missing `ptychi` dependency, capture traceback into docs/summary.md, record failure in docs/fix_plan.md Attempt #81, and halt further execution.
- If test path update still raises FileNotFoundError, revert to RED artifact, log issue in summary.md, and leave test change unapplied for follow-up guidance.

Findings Applied (Mandatory):
- POLICY-001 — Maintain torch dependency expectations; no optional gating for CLI entry.
- CONFIG-001 — Test update must not touch params.cfg; CLI run must rely on existing bridge sequence.
- CONFIG-002 — Execution configs remain isolated; avoid writing to params.cfg during script invocation.
- DATA-001 — Dense/test NPZ must keep amplitude diffraction + complex64 patches.
- OVERSAMPLING-001 — Dense view retains K≥C assumption; document any skip events in manifest.

Pointers:
- tests/scripts/test_ptychi_reconstruct_tike.py:1
- scripts/reconstruction/ptychi_reconstruct_tike.py:296
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:34
- docs/fix_plan.md:52
- docs/findings.md:2

Next Up (optional):
- 1. After dense/test succeeds, extend LSQML execution to sparse/train to exercise skip-reporting telemetry under real missing-view conditions.

Doc Sync Plan (Conditional):
- After GREEN selectors pass, run AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv 2>&1 | tee .../collect/pytest_phase_f_cli_collect.log (already in Do Now) and update docs/TESTING_GUIDE.md §Phase F plus docs/development/TEST_SUITE_INDEX.md with selector + artifact references.


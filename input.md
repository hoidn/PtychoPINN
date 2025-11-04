Summary: Normalize Phase E manifest bundle paths and capture real dose=1000 dense training bundles for Phase G.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (Phase E6 bundle evidence)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/

Do Now:
- Implement: studies/fly64_dose_overlap/training.py::main + tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path — ensure CLI manifest records `bundle_path` relative to each job’s artifact_dir, add the new RED→GREEN test asserting manifest fields and bundle presence, and keep skip_summary serialization unchanged.
- Validate: pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/red/pytest_training_cli_bundle_red.log (expect failure before implementation), then rerun after changes teeing GREEN output to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/green/pytest_training_cli_bundle_green.log.
- Validate: pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/green/pytest_training_cli_suite_green.log to confirm CLI regressions stay clear.
- Collect: pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/collect/pytest_training_cli_collect.log.
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/cli/dose1000_dense_gs2.log, then rerun with --view baseline --gridsize 1 teeing to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/cli/dose1000_baseline_gs1.log; copy resulting `tmp/phase_e_training_gs2/{pinn,baseline}` into the artifact hub’s `cli/` subdir for traceability.
- Summarize: Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/analysis/summary.md with manifest excerpts (bundle_path entries, skip counts), CLI outputs, and enumerate remaining gaps (e.g., sparse view bundles, higher doses).

Priorities & Rationale:
- docs/fix_plan.md:31 — Phase G comparisons remain blocked until real Phase E bundles exist; manifest normalization supports downstream automation.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268 — Phase E6 checklist calls for aggregated gs2 training evidence with deterministic runs and artifacts.
- specs/ptychodus_api_spec.md:239 — Specs mandate `wts.h5.zip` persistence; manifest must clearly point to archives for comparison tooling.
- docs/TESTING_GUIDE.md:101-140 — Authoritative Phase E test/CLI commands; ensure new selector integrates with documented workflow.
- docs/development/TEST_SUITE_INDEX.md:60 — Update registry once the new CLI bundle-path test exists to keep coverage map authoritative.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/{red,green,collect,cli,analysis,docs}
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/red/pytest_training_cli_bundle_red.log
- Apply code/test edits; rerun pytest selector teeing to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/green/pytest_training_cli_bundle_green.log
- pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/green/pytest_training_cli_suite_green.log
- pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/collect/pytest_training_cli_collect.log
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/cli/dose1000_dense_gs2.log
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/cli/dose1000_baseline_gs1.log
- cp -a tmp/phase_e_training_gs2/{pinn,baseline} plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/cli/
- Document manifest snippets + outstanding gaps in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T230500Z/phase_e_training_bundle_real_run/analysis/summary.md

Pitfalls To Avoid:
- Do not skip the RED phase for the new pytest selector; capture failure output before implementation.
- Keep manifest JSON stable; only adjust bundle_path serialization without altering skip_summary schema.
- Ensure bundle_path conversion uses artifact-relative paths to avoid embedding workstation-specific absolutes.
- Verify tmp/phase_c_f2_cli and tmp/phase_d_f2_cli exist before running CLI; if missing, note the gap instead of fabricating data.
- Maintain CONFIG-001 compliance—`execute_training_job` already bridges params.cfg; avoid additional global mutations.
- Leave tensorboard/MLflow settings untouched to respect Environment Freeze.
- Capture stderr/stdout via tee so CLI failures include actionable context in artifacts.
- Do not overwrite existing artifact directories; copy or sync into the new timestamped hub only.
- Avoid deleting prior attempt logs or tmp outputs; reference them if datasets are stale.
- No package/environment tweaks; surface ImportErrors via logs and mark blocked if encountered.

If Blocked:
- Preserve RED logs and CLI stderr in artifact hub, note failure mode in analysis summary + docs/fix_plan.md, and mark the attempt blocked pending dataset regeneration or environment fix.

Findings Applied (Mandatory):
- POLICY-001 — docs/findings.md:8; PyTorch dependency enforced, CLI runs must emit actionable errors if torch missing.
- CONFIG-001 — docs/findings.md:10; ensure legacy bridge remains single-call in run_training_job and manifest update does not bypass it.
- DATA-001 — docs/findings.md:14; use canonical NPZ paths from Phase C/D without ad-hoc edits.
- OVERSAMPLING-001 — docs/findings.md:17; note sparse view availability and acceptance metadata when summarizing outcomes.

Pointers:
- docs/fix_plan.md:31 — Status block describing Phase G dependency on Phase E bundles.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268 — Phase E6 checklist for aggregated training evidence.
- specs/ptychodus_api_spec.md:239 — Bundle persistence contract (`wts.h5.zip`).
- docs/TESTING_GUIDE.md:101-140 — Authoritative Phase E testing + CLI commands.
- docs/development/TEST_SUITE_INDEX.md:60 — Registry entry for Phase E training tests (needs update post-loop).

Next Up (optional): Re-run Phase G comparisons for dose=1000 dense/train once bundles exist.

Doc Sync Plan:
- After GREEN, refresh docs/TESTING_GUIDE.md §Phase E and docs/development/TEST_SUITE_INDEX.md with the new bundle-path selector; capture `pytest --collect-only` log under this loop’s artifacts and link in summary before closing.

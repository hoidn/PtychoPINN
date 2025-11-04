Summary: Execute Phase G dose_1000 comparisons end-to-end by adding execution manifest summaries and capturing real-run evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.G2 — Deterministic comparison runs
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_records_summary -vv; pytest tests/study/test_dose_overlap_comparison.py -k comparison -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/

Do Now:
- Implement: studies/fly64_dose_overlap/comparison.py::execute_comparison_jobs + tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_records_summary — extend the executor to record success/failure counts in the manifest and update the CLI summary, with a RED→GREEN pytest proving the new fields.
- Validate: pytest tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_records_summary -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/red/pytest_phase_g_executor_red.log (expect failure before implementation) then rerun after the change teeing to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/green/pytest_phase_g_executor_green.log.
- Collect: pytest tests/study/test_dose_overlap_comparison.py --collect-only -k comparison -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/collect/pytest_phase_g_collect.log.
- Capture: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.comparison --phase-c-root tmp/phase_c_f2_cli --phase-e-root tmp/phase_e_training_gs2 --phase-f-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/phase_f_cli_test --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/cli/dose1000_dense --dose 1000 --view dense | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/cli/dose1000_dense.log.
- Capture: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.comparison --phase-c-root tmp/phase_c_f2_cli --phase-e-root tmp/phase_e_training_gs2 --phase-f-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/phase_f_cli_test --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/cli/dose1000_sparse_train --dose 1000 --view sparse --split train | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/cli/dose1000_sparse_train.log.
- Summarize: Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/analysis/summary.md with return codes, metrics file inventory, and outstanding gaps; attach manifest paths.

Priorities & Rationale:
- docs/fix_plan.md:31 — Phase G2 execution remains outstanding; this loop delivers the first real comparisons.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/plan/plan.md:32 — G2.1 requires dense train/test runs with captured metrics and logs.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/inventory.md:165 — Only dose_1000 dense/sparse-train combinations are currently unblocked.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:252 — Phase G test strategy calls for RED→GREEN selector evidence plus real-run logs before registry updates.
- docs/findings.md:8 — POLICY-001 mandates torch availability; executor must surface any compare_models import failures promptly.
- docs/findings.md:10 — CONFIG-001 bridge must remain encapsulated inside compare_models invocation; our orchestration stays pure.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/{red,green,collect,cli,analysis,docs}
- pytest tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_records_summary -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/red/pytest_phase_g_executor_red.log
- Implement execution manifest summary + CLI updates (ensure new fields n_success/n_failed are persisted)
- pytest tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_records_summary -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/green/pytest_phase_g_executor_green.log
- pytest tests/study/test_dose_overlap_comparison.py -k comparison -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/green/pytest_phase_g_suite_green.log
- pytest tests/study/test_dose_overlap_comparison.py --collect-only -k comparison -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/collect/pytest_phase_g_collect.log
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.comparison --phase-c-root tmp/phase_c_f2_cli --phase-e-root tmp/phase_e_training_gs2 --phase-f-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/phase_f_cli_test --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/cli/dose1000_dense --dose 1000 --view dense | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/cli/dose1000_dense.log
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.comparison --phase-c-root tmp/phase_c_f2_cli --phase-e-root tmp/phase_e_training_gs2 --phase-f-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/phase_f_cli_test --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/cli/dose1000_sparse_train --dose 1000 --view sparse --split train | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/cli/dose1000_sparse_train.log
- ls plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/cli/**/metrics* > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/analysis/metrics_inventory.txt
- Write analysis/summary.md noting return codes, metrics CSV presence, comparison manifest paths, and any failed jobs.

Pitfalls To Avoid:
- Do not run the dense command without verifying `tmp/phase_d_f2_cli/dose_1000/dense` exists; capture ls output if missing.
- Avoid executing sparse/test until Phase F sparse_test reconstruction lands; filter with `--split train`.
- Do not swallow subprocess stderr—ensure logs capture stdout/stderr for each job.
- Keep executor pure; no direct imports of compare_models artifacts before CONFIG-001 runs inside the subprocess.
- Do not overwrite existing manifests; create new artifact roots per run.
- Capture RED failure output before implementation; do not skip TDD cadence.
- Ensure metrics CSVs/plots remain under the timestamped artifact hub; no tmp artifacts left behind.
- Respect timeout handling—do not remove the 600s guard in executor.
- Avoid changing dataset paths or study design constants in this loop.
- Do not edit Phase F artifacts; treat them as read-only inputs.

If Blocked:
- If compare_models fails (ImportError, missing checkpoint, etc.), capture the stderr in the CLI log, record the failing command, and log the issue in docs/fix_plan.md with artifact references; leave G2 marked blocked with the error signature.

Findings Applied (Mandatory):
- POLICY-001 — docs/findings.md:8; ensure torch-required path surfaces failure codes rather than skipping.
- CONFIG-001 — docs/findings.md:10; executor remains orchestration only, compare_models handles legacy bridge.
- DATA-001 — docs/findings.md:14; phase C/D NPZ paths follow canonical layout (patched_{split}.npz, view/{view}_{split}.npz).
- OVERSAMPLING-001 — docs/findings.md:17; sparse run summary must include acceptance metadata from manifests.

Pointers:
- docs/fix_plan.md:31 — Active focus status and outstanding G2 execution work.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/plan/plan.md:32 — G2 checklist expectations.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/inventory.md:165 — Ready vs blocked comparison conditions.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:252 — Phase G selectors and execution proof policy.
- scripts/compare_models.py:1 — Execution script invoked by the executor; treat as authoritative target.

Next Up:
- Run sparse/test comparison once Phase F sparse_test reconstruction is generated.

Doc Sync Plan:
- After GREEN tests, rerun `pytest tests/study/test_dose_overlap_comparison.py --collect-only -k comparison -vv` (log already planned) and update docs/TESTING_GUIDE.md §Phase G plus docs/development/TEST_SUITE_INDEX.md with the new selector and CLI commands, citing the new artifact logs.

Mapped Tests Guardrail: ensure `pytest tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_records_summary -vv` collects (new test) and remains in the selector list; rerun collect-only proof if the test name changes.

Summary: Implement Phase G comparison executor (non-dry-run path) and prove it with targeted pytest plus CLI evidence for dose_1000 conditions.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.G2 — Deterministic comparison runs
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_invokes_compare_models -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T173500Z/phase_g_execution_g2/

Do Now:
- Implement: studies/fly64_dose_overlap/comparison.py::execute_comparison_jobs + tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_invokes_compare_models — add an executor helper that shells out to `scripts/compare_models.py`, update the CLI to invoke it when `--dry-run` is false, and drive the change through a new RED→GREEN pytest.
- Validate: pytest tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_invokes_compare_models -vv (tee RED to reports/.../red/pytest_phase_g_executor_red.log, GREEN to .../green/pytest_phase_g_executor_green.log).
- Collect: pytest tests/study/test_dose_overlap_comparison.py --collect-only -k comparison -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T173500Z/phase_g_execution_g2/collect/pytest_phase_g_collect.log.
- Capture: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.comparison --phase-c-root tmp/phase_c_f2_cli --phase-e-root tmp/phase_e_training_gs2 --phase-f-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/phase_f_cli_test --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T173500Z/phase_g_execution_g2/cli --dose 1000 --view dense --split train --dry-run | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T173500Z/phase_g_execution_g2/cli/phase_g_cli_dry_run.log.
- Artifacts: Stage `reports/2025-11-05T173500Z/phase_g_execution_g2/{red,green,collect,cli,analysis}` before execution; drop pytest summaries, CLI transcript, and (if available) comparison summaries/return codes inside.

Priorities & Rationale:
- docs/fix_plan.md:31 — Active focus notes G2 execution outstanding; this loop must advance deterministic comparison runs.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/plan/plan.md:31 — G2.1 expects dense train/test executions with captured logs/metrics.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/inventory.md:8 — Inventory restricts feasible comparisons to dose_1000 (dense train/test, sparse train); executor must respect those prerequisites.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:248 — Phase G section demands a RED→GREEN selector plus CLI evidence before registering real-run commands.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T173500Z/phase_g_execution_g2/{red,green,collect,cli,analysis}
- pytest tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_invokes_compare_models -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T173500Z/phase_g_execution_g2/red/pytest_phase_g_executor_red.log
- Implement executor + CLI wiring, then rerun pytest tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_invokes_compare_models -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T173500Z/phase_g_execution_g2/green/pytest_phase_g_executor_green.log
- pytest tests/study/test_dose_overlap_comparison.py --collect-only -k comparison -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T173500Z/phase_g_execution_g2/collect/pytest_phase_g_collect.log
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.comparison --phase-c-root tmp/phase_c_f2_cli --phase-e-root tmp/phase_e_training_gs2 --phase-f-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/phase_f_cli_test --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T173500Z/phase_g_execution_g2/cli --dose 1000 --view dense --split train | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T173500Z/phase_g_execution_g2/cli/phase_g_cli_run.log
- Summarize exit codes + outstanding gaps in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T173500Z/phase_g_execution_g2/analysis/summary.md

Pitfalls To Avoid:
- Do not skip the RED phase; commit TDD by running the new pytest before implementation.
- Avoid invoking `compare_models.py` directly in tests—mock subprocess calls to keep tests fast and deterministic.
- Keep executor pure wrt CONFIG-001: no direct legacy imports before `update_legacy_dict` lives inside the called script.
- Capture subprocess return codes and stderr; do not swallow failures silently.
- Store all logs under the timestamped artifact hub; no ad-hoc tmp outputs left behind.
- Respect `POLICY-001` by assuming torch is installed—no optional imports or silent downgrades.
- When CLI execution fails due to missing prerequisites, record the error message instead of patching around it.
- Do not change dose/view lists yet; focus on executor wiring for ready conditions.

If Blocked:
- If required Phase C/E/F assets are missing, run the CLI command above, tee the failure into `analysis/blockers.log`, snapshot `ls` output of the missing directory into the same folder, and log the block in docs/fix_plan.md (Attempt history) referencing the captured error.

Findings Applied (Mandatory):
- POLICY-001 — Executor assumes PyTorch backend is available; failures must be surfaced if compare_models.py cannot import torch.
- CONFIG-001 — Comparisons rely on scripts that perform `update_legacy_dict`; ensure our orchestration does not bypass the bridge.
- DATA-001 — Job construction continues to point at canonical Phase C/D NPZ layouts (patched_{split}.npz, view/{view}_{split}.npz).
- OVERSAMPLING-001 — Preserve manifest metadata (selection_strategy, acceptance_rate) when logging sparse job execution.

Pointers:
- docs/fix_plan.md:31 — Phase G status + outstanding tasks.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/plan/plan.md:31 — G2 checklist expectations.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/inventory.md:8 — Ready vs blocked comparison conditions.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:248 — Active selectors + execution proof policy for Phase G.
- studies/fly64_dose_overlap/comparison.py:130 — CLI entry requiring executor integration.

Next Up (optional):
- G2.2 sparse/train CLI execution once executor scaffolding lands and dose_1000 assets verify.

Doc Sync Plan (conditional):
- After GREEN tests, rerun `pytest tests/study/test_dose_overlap_comparison.py --collect-only -k comparison -vv` (log already captured) and update `docs/TESTING_GUIDE.md` §Phase G plus `docs/development/TEST_SUITE_INDEX.md` with the new selector + CLI command, citing evidence paths once executor produces real runs.

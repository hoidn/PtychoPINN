Summary: Ship dense Phase G pipeline evidence with automated artifact inventory and full metrics verification in the 2025-11-09T210500Z hub.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T210500Z/phase_g_dense_full_execution_real_run/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — add deterministic `analysis/artifact_inventory.txt` emission. TDD: extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` to fail red when the inventory file is missing, capture the RED log at `$HUB/red/pytest_orchestrator_dense_exec_inventory_fail.log`, then implement the new helper and confirm GREEN.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv (log to `$HUB/green/pytest_orchestrator_dense_exec_inventory_fix.log`).
- Execute: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
- Verify: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json
- Document: Update `$HUB/summary/summary.md`, `docs/fix_plan.md`, and (if new lessons) `docs/findings.md`; ensure MS-SSIM/MAE deltas and provenance are captured with artifact links.

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T210500Z/phase_g_dense_full_execution_real_run
3. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/red/pytest_orchestrator_dense_exec_inventory_fail.log
4. (after implementation) pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_inventory_fix.log
5. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
6. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json
7. cp "$HUB"/analysis/artifact_inventory.txt "$HUB"/analysis/artifact_inventory_check.txt  # optional copy for quick diff

Pitfalls To Avoid
- Do not skip the RED → GREEN test cycle; capture both logs in hub/red and hub/green.
- Keep `artifact_inventory.txt` sorted with POSIX-style relative paths; don’t emit absolute paths or nondeterministic ordering.
- Always run the pipeline with `--clobber` to clear partial Phase C outputs; otherwise prepare_hub will raise.
- Ensure `AUTHORITATIVE_CMDS_DOC` remains exported for every pytest and pipeline invocation (CONFIG-001 guard expects it).
- Do not terminate long-running training/reconstruction subprocesses; if they fail, archive logs and document blockers.
- Watch disk usage when running the dense pipeline; if storage issues appear, pause and record the condition before retrying.
- Preserve `_metadata` fields when inspecting NPZs—never modify Phase C outputs in-place.

If Blocked
- If the new test cannot be made RED (e.g., it already passes), document why in `$HUB/summary/summary.md`, capture the attempted log, and note the gap in docs/fix_plan.md before proceeding.
- If pipeline execution fails, move the failing log to `$HUB/red/`, summarize the error signature in summary.md, update docs/fix_plan.md with a blocked status, and leave the hub intact for analysis.
- If verifier reports invalid artifacts, archive the JSON report + offending files, record findings in summary.md, and stop before marking the loop done.

Findings Applied (Mandatory)
- POLICY-001 — PyTorch dependency policy honored; ensure environment keeps torch>=2.2 installed.
- CONFIG-001 — Always bridge configs via `update_legacy_dict` (already handled in orchestrator, but confirm in summary).
- DATA-001 — Validate all NPZs produced by the pipeline conform to the data contract; note compliance in summary.
- TYPE-PATH-001 — Normalize paths (inventory + verifier) to avoid string/Path mismatches.
- OVERSAMPLING-001 — Dense view relies on K > C; confirm summary highlights adherence.
- STUDY-001 — Capture MS-SSIM/MAE deltas against baseline/PtyChi in summary.
- PHASEC-METADATA-001 — Ensure metadata compliance guard runs and report status in summary.

Pointers
- docs/fix_plan.md:4
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T210500Z/phase_g_dense_full_execution_real_run/plan/plan.md:1
- tests/study/test_phase_g_dense_orchestrator.py:977
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:1
- docs/TESTING_GUIDE.md:1
- docs/findings.md:8
- specs/data_contracts.md:1

Next Up (optional)
- If dense run completes quickly, queue sparse-view pipeline execution plan for the next loop.

Doc Sync Plan (Conditional)
- If the orchestrator test selector name changes or new tests are added, run `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv | tee "$HUB"/collect/pytest_collect.log` after GREEN and update `docs/TESTING_GUIDE.md` §2 plus `docs/development/TEST_SUITE_INDEX.md`.

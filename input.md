Summary: Ship dense Phase G evidence by adding inventory validation to the verifier under TDD and rerunning the end-to-end pipeline into the 2025-11-10T093500Z hub.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_artifact_inventory_blocks_missing_entries -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_artifact_inventory_passes_with_complete_bundle -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T093500Z/phase_g_dense_full_execution_real_run/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::main — add `validate_artifact_inventory` (POSIX-relative path checks, required bundle entries) and integrate into the verifier flow. Drive TDD by first authoring RED tests in `tests/study/test_phase_g_dense_artifacts_verifier.py` that invoke the CLI against (a) missing-inventory and (b) complete-inventory hubs, capturing the RED log at `$HUB/red/pytest_artifact_inventory_fail.log`, then implement the helper and confirm GREEN at `$HUB/green/pytest_artifact_inventory_fix.log`.
- Validate: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_artifact_inventory_blocks_missing_entries -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_artifact_inventory_passes_with_complete_bundle -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv (capture outputs in `$HUB/red/` then `$HUB/green/pytest_orchestrator_dense_exec_inventory_fix.log`).
- Execute: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
- Verify: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json
- Document: Update `$HUB/summary/summary.md` with MS-SSIM/MAE deltas, metadata compliance, verifier summary, and log references; refresh `docs/fix_plan.md` Attempts History and add any durable lessons to `docs/findings.md`.

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T093500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}
4. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_artifact_inventory_blocks_missing_entries -vv | tee "$HUB"/red/pytest_artifact_inventory_fail.log
5. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_artifact_inventory_passes_with_complete_bundle -vv | tee "$HUB"/green/pytest_artifact_inventory_fix.log
6. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_inventory_fix.log
7. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
8. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json
9. pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect.log

Pitfalls To Avoid
- Do not skip the RED stage for the new verifier tests; archive the failing log before implementing the helper.
- Keep `analysis/artifact_inventory.txt` strictly POSIX-relative with deterministic sorting; reject absolute paths or backslashes.
- Ensure `AUTHORITATIVE_CMDS_DOC` is exported for every pytest/pipeline run so CONFIG-001 guards stay active.
- Avoid touching legacy Phase C NPZs; pipeline must regenerate outputs with `--clobber` instead of manual edits.
- Document any pipeline failures immediately in summary.md and stop rather than rerunning silently.
- Watch disk usage during Phase E/F; abort gracefully if storage issues arise and record the block.
- Keep new pytest selectors isolated to this module to avoid long runtimes beyond the targeted guard.

If Blocked
- If verifier tests cannot fail (e.g., helper already present), capture the attempted log, explain the reason in `$HUB/summary/summary.md`, and update `docs/fix_plan.md` with blocked status before proceeding.
- If pipeline execution aborts, move the failing CLI log to `$HUB/red/`, summarize the error signature in summary.md, and mark the attempt blocked in `docs/fix_plan.md`.
- If verifier reports missing artifacts post-run, keep the hub intact, archive the report, and stop for triage.

Findings Applied (Mandatory)
- POLICY-001 — PyTorch dependency policy stays enforced; leave torch>=2.2 installed.
- CONFIG-001 — Ensure legacy bridge (`update_legacy_dict`) remains before TensorFlow modules; note compliance in summary.
- DATA-001 — Validate generated NPZs adhere to the data contract; record results via verifier + summary.
- TYPE-PATH-001 — Inventory helper must normalize to POSIX relative paths; reject absolute/malformed entries.
- OVERSAMPLING-001 — Dense overlap parameters stay aligned with design; mention in summary while reviewing outputs.
- STUDY-001 — Capture MS-SSIM/MAE deltas vs Baseline/PtyChi in documentation.
- PHASEC-METADATA-001 — Confirm metadata compliance section renders with real dataset results and reference in summary.

Pointers
- docs/fix_plan.md:4
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T093500Z/phase_g_dense_full_execution_real_run/plan/plan.md:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:1
- tests/study/test_phase_g_dense_orchestrator.py:977
- specs/data_contracts.md:1
- docs/TESTING_GUIDE.md:1
- docs/findings.md:8

Next Up (optional)
- Prepare sparse-view Phase G rerun plan once dense pipeline artifacts are verified and documented.

Doc Sync Plan (Conditional)
- After GREEN, run `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect.log`, then update `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` with the new selectors.

Mapped Tests Guardrail
- The selectors above must collect >0 tests; if collection fails, first author/repair the test module before implementation progress.

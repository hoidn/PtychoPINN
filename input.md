Summary: Capture dense Phase G CLI evidence with a new collect-only smoke test and post-run validation on a clean hub.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_prepare_hub_clobbers_previous_outputs -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T230500Z/phase_g_dense_cli_execution/

Do Now (hard validity contract):
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands — add a collect-only smoke test that loads main(), runs with --collect-only into a tmp hub, asserts command text matches expected substrings, and verifies no Phase C outputs are created.
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_prepare_hub_clobbers_previous_outputs -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T230500Z/phase_g_dense_cli_execution --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T230500Z/phase_g_dense_cli_execution/cli/phase_g_dense_pipeline.log
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): python - <<'PY' >> plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T230500Z/phase_g_dense_cli_execution/analysis/validate_and_summarize.log
from pathlib import Path
from run_phase_g_dense import validate_phase_c_metadata, summarize_phase_g_outputs
hub = Path("plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T230500Z/phase_g_dense_cli_execution")
validate_phase_c_metadata(hub)
summarize_phase_g_outputs(hub)
PY

How-To Map:
  1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  2. mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T230500Z/phase_g_dense_cli_execution/{analysis,cli,collect,green,red,summary}
  3. Run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T230500Z/phase_g_dense_cli_execution/green/pytest_collect_only_green.log`
  4. Run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_prepare_hub_clobbers_previous_outputs -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T230500Z/phase_g_dense_cli_execution/green/pytest_prepare_hub_clobber_green.log`
  5. Run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T230500Z/phase_g_dense_cli_execution/green/pytest_metadata_guard_green.log`
  6. Run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T230500Z/phase_g_dense_cli_execution/green/pytest_summarize_green.log`
  7. Capture selector coverage with `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T230500Z/phase_g_dense_cli_execution/collect/pytest_phase_g_orchestrator_collect.log`
  8. Execute the dense run with `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T230500Z/phase_g_dense_cli_execution --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T230500Z/phase_g_dense_cli_execution/cli/phase_g_dense_pipeline.log`
  9. Post-run, execute the embedded validation script from Do Now step to log `validate_phase_c_metadata` + `summarize_phase_g_outputs` results into `analysis/validate_and_summarize.log`
  10. Update summary/summary.md with Turn Summary, metrics snapshot, and key pointers once outputs exist.

Pitfalls To Avoid:
  - Do not force clobber logic outside prepare_hub; rely on helper for cleanup and archive path emission.
  - Keep collect-only test fully isolated—no real filesystem side effects beyond tmp_path.
  - Ensure AUTHORITATIVE_CMDS_DOC remains exported for all pytest and CLI commands.
  - Watch for long-running CLI; abort on first failure and capture blocker log instead of rerunning blindly.
  - Validate that metrics_summary.json/md exist after summarization; log a blocker if missing.
  - Avoid hard-coded absolute paths in tests; derive from tmp_path to maintain portability.
  - Maintain TYPE-PATH-001 compliance by normalizing paths inside tests before assertions.
  - Leave prior evidence archived; never delete archive/ directories generated by prepare_hub.
  - Verify new test is deterministic—mock or patch time-based components if they surface in output.
  - Keep pytest output logs under the hub directories named above; no stray files at repo root.

If Blocked:
  - Stop immediately, save failing pytest/CLI output to `analysis/blocker.log`, annotate summary.md with blocker context, and document the block in docs/fix_plan.md + galph_memory.md before ending the loop.

Findings Applied (Mandatory):
  - POLICY-001 — Dense pipeline may trigger PyTorch baselines; ensure environment honors mandatory torch dependency.
  - CONFIG-001 — Orchestrator already bridges params.cfg; collect-only test must not bypass this contract.
  - DATA-001 — Post-run validation must confirm NPZ metadata contract on fresh outputs.
  - TYPE-PATH-001 — Normalize hub paths in tests and validation scripts to prevent string/path regressions.
  - OVERSAMPLING-001 — Preserve dense overlap parameters; report any divergence if CLI output suggests mismatch.

Pointers:
  - docs/findings.md:8
  - docs/findings.md:10
  - docs/findings.md:14
  - docs/findings.md:21
  - docs/findings.md:17
  - docs/TESTING_GUIDE.md:215
  - docs/development/TEST_SUITE_INDEX.md:62
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:137
  - tests/study/test_phase_g_dense_orchestrator.py:380
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:210

Next Up (optional):
  - Extend collect-only smoke coverage to sparse view once dense evidence is captured.

Doc Sync Plan (Conditional):
  - After GREEN, re-run `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv` (log already captured) and update docs/TESTING_GUIDE.md §2 plus docs/development/TEST_SUITE_INDEX.md with the new collect-only test entry.
